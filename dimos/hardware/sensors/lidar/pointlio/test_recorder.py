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

import asyncio
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest

from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


@pytest.fixture
def recorder_factory() -> Generator[Callable[..., PointlioRecorder], None, None]:
    recorders: list[PointlioRecorder] = []

    def create(**kwargs: Any) -> PointlioRecorder:
        recorder = PointlioRecorder(**kwargs)
        recorders.append(recorder)
        return recorder

    yield create

    for recorder in recorders:
        recorder._close_module()


def _cloud(ts: float) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_id="mid360_link",
        timestamp=ts,
    )


@pytest.mark.asyncio
async def test_exact_pose_waits_for_out_of_order_odometry(
    recorder_factory: Callable[..., PointlioRecorder],
) -> None:
    recorder = recorder_factory(
        require_exact_pose_stamp=True,
        exact_pose_wait_seconds=0.1,
    )
    pose = Pose(1.0, 2.0, 3.0)

    pending = asyncio.create_task(recorder._lidar_pose(_cloud(42.0)))
    await asyncio.sleep(0)
    await recorder._odom_pose(Odometry(ts=42.0, pose=pose))

    assert await pending is pose


@pytest.mark.asyncio
async def test_exact_pose_rejects_mismatched_stamp(
    recorder_factory: Callable[..., PointlioRecorder],
) -> None:
    recorder = recorder_factory(
        require_exact_pose_stamp=True,
        exact_pose_wait_seconds=0.001,
    )

    await recorder._odom_pose(Odometry(ts=41.0, pose=Pose()))
    assert await recorder._lidar_pose(_cloud(42.0)) is None


@pytest.mark.asyncio
async def test_exact_pose_buffer_is_bounded(
    recorder_factory: Callable[..., PointlioRecorder],
) -> None:
    recorder = recorder_factory(require_exact_pose_stamp=True, pose_buffer_size=2)

    for ts in (1.0, 2.0, 3.0):
        await recorder._odom_pose(Odometry(ts=ts, pose=Pose(ts, 0.0, 0.0)))

    assert list(recorder._poses_by_stamp) == [2_000_000_000, 3_000_000_000]
