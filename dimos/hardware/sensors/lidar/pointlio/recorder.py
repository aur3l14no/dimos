# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record Point-LIO odometry + lidar into a memory2 SQLite db.

A ``Recorder`` that records its In ports under their own names
(``pointlio_odometry`` / ``pointlio_lidar``). By default each lidar frame uses
the latest odometry pose for compatibility with existing recordings. Mapping
blueprints can enable ``require_exact_pose_stamp`` to wait briefly for, and only
attach, an odometry pose with the exact same sensor timestamp.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from pydantic import Field

from dimos.core.stream import In
from dimos.memory2.module import OnExisting, Recorder, RecorderConfig, pose_setter_for
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


class PointlioRecorderConfig(RecorderConfig):
    # Append into a populated db (keep other streams); replace only our own.
    on_existing: OnExisting = OnExisting.APPEND
    # Require the pose attached to a lidar observation to come from odometry
    # with the exact same sensor timestamp. Legacy recorders keep latest-pose
    # behavior unless they opt in.
    require_exact_pose_stamp: bool = False
    exact_pose_wait_seconds: float = Field(default=0.1, ge=0.0)
    pose_buffer_size: int = Field(default=256, gt=0)


class PointlioRecorder(Recorder):
    config: PointlioRecorderConfig

    pointlio_odometry: In[Odometry]
    pointlio_lidar: In[PointCloud2]

    _last_odom_pose: Pose | None = None
    _poses_by_stamp: OrderedDict[int, Pose]
    _pose_waiters: dict[int, asyncio.Event]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._poses_by_stamp = OrderedDict()
        self._pose_waiters = {}

    @staticmethod
    def _stamp_key(ts: float | None) -> int | None:
        if ts is None:
            return None
        return round(ts * 1_000_000_000)

    @pose_setter_for("pointlio_odometry")
    async def _odom_pose(self, msg: Odometry) -> Pose | None:
        pose = getattr(msg, "pose", None)
        self._last_odom_pose = getattr(pose, "pose", None) if pose is not None else None
        if not self.config.require_exact_pose_stamp:
            return self._last_odom_pose

        key = self._stamp_key(msg.ts)
        if key is not None and self._last_odom_pose is not None:
            self._poses_by_stamp[key] = self._last_odom_pose
            self._poses_by_stamp.move_to_end(key)
            while len(self._poses_by_stamp) > self.config.pose_buffer_size:
                self._poses_by_stamp.popitem(last=False)
            waiter = self._pose_waiters.pop(key, None)
            if waiter is not None:
                waiter.set()
        return self._last_odom_pose

    @pose_setter_for("pointlio_lidar")
    async def _lidar_pose(self, msg: PointCloud2) -> Pose | None:
        if not self.config.require_exact_pose_stamp:
            return self._last_odom_pose

        key = self._stamp_key(msg.ts)
        if key is None:
            return None
        pose = self._poses_by_stamp.get(key)
        if pose is not None:
            return pose

        waiter = self._pose_waiters.setdefault(key, asyncio.Event())
        try:
            await asyncio.wait_for(waiter.wait(), timeout=self.config.exact_pose_wait_seconds)
        except TimeoutError:
            return None
        finally:
            self._pose_waiters.pop(key, None)
        return self._poses_by_stamp.get(key)
