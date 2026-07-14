#!/usr/bin/env python3
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

"""Sensor-only PointLIO recorder for remote-controlled G1 mapping.

The Unitree handheld controller owns locomotion. This blueprint deliberately
does not instantiate keyboard teleop, MovementManager, or G1HighLevelDdsSdk.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder


def _default_recording_dir() -> Path:
    override = os.getenv("DIMOS_G1_RECORDING_DIR")
    if override:
        return Path(override).expanduser()
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%Z")
    return RECORDINGS_DIR / f"g1-pointlio-{stamp}"


_RECORDING_DIR = _default_recording_dir()


unitree_g1_pointlio_record = autoconnect(
    PointLio.blueprint(
        frame_id="world",
        host_ip=os.getenv("DIMOS_POINTLIO_HOST_IP", "192.168.123.164"),
        lidar_ip=os.getenv("DIMOS_POINTLIO_LIDAR_IP", "192.168.123.120"),
        pointcloud_freq=5.0,
        scan_publish_en=False,
        deskewed_scan_publish_en=True,
    ).remappings(
        [
            (PointLio, "deskewed_lidar", "pointlio_lidar"),
            (PointLio, "lidar_odometry", "pointlio_odometry"),
        ]
    ),
    PointlioRecorder.blueprint(db_path=str(_RECORDING_DIR / "mem2.db")),
).global_config(n_workers=6, robot_model="unitree_g1")


def main() -> None:
    _RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Recording G1 PointLIO data to {_RECORDING_DIR / 'mem2.db'}")
    print("Locomotion remains under the Unitree handheld controller.")
    coordinator = ModuleCoordinator.build(unitree_g1_pointlio_record)
    coordinator.loop()


if __name__ == "__main__":
    main()
