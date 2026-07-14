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

"""Sensor-only G1 relocalization validation.

Publishes the coherent deskewed PointLIO scan/odometry pair, accumulated map,
loaded static map, merged map, and relocalization TF. It deliberately contains
no G1 control module, planner, MovementManager, or cmd_vel path.
"""

from __future__ import annotations

import os
from pathlib import Path

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.relocalization.module import RelocalizationModule
from dimos.navigation.cmu_nav.frames import FRAME_ODOM

_MAP_FILE = os.getenv("DIMOS_G1_MAP_FILE")

unitree_g1_relocalization_view = (
    autoconnect(
        PointLio.blueprint(
            host_ip=os.getenv("DIMOS_POINTLIO_HOST_IP", "192.168.123.164"),
            lidar_ip=os.getenv("DIMOS_POINTLIO_LIDAR_IP", "192.168.123.120"),
            pointcloud_freq=5.0,
            scan_publish_en=False,
            deskewed_scan_publish_en=True,
        ),
        RayTracingVoxelMap.blueprint(
            voxel_size=0.08,
            emit_every=0,
            global_emit_every=5,
        ),
        RelocalizationModule.blueprint(
            map_file=_MAP_FILE,
            publish_loaded_map=True,
            fitness_threshold=0.45,
            use_carving=False,
            live_frame=FRAME_ODOM,
        ),
    )
    .remappings(
        [
            (RayTracingVoxelMap, "lidar", "deskewed_lidar"),
            (RayTracingVoxelMap, "odometry", "lidar_odometry"),
        ]
    )
    .global_config(n_workers=8, robot_model="unitree_g1")
)


def main() -> None:
    if not _MAP_FILE:
        raise RuntimeError("DIMOS_G1_MAP_FILE must point to a .pc2.lcm static map")
    path = Path(_MAP_FILE).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Relocalizing G1 against {path}")
    print("Sensor-only validation: no robot control module is active.")
    coordinator = ModuleCoordinator.build(unitree_g1_relocalization_view)
    coordinator.loop()


if __name__ == "__main__":
    main()
