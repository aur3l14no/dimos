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
from typing import Any

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.visualization.vis_module import vis_module

_VOXEL_SIZE = 0.08


def _render_global_map(msg: PointCloud2) -> Any:
    return msg.to_rerun(voxel_size=_VOXEL_SIZE)


def _static_g1_body(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.25, 0.20, 0.6],
            centers=[[0.0, 0.0, -0.6]],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/mid360_link"),
    ]


def _mapping_rerun_blueprint() -> Any:
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name="G1 PointLIO live mapping",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(
                plane=rr.components.Plane3D.XY.with_distance(0.0),
            ),
        ),
        rrb.TimePanel(state="collapsed"),
    )


_RERUN_CONFIG = {
    "blueprint": _mapping_rerun_blueprint,
    "visual_override": {
        "world/global_map": _render_global_map,
        "world/pointlio_lidar": None,
        "world/lidar": None,
        "world/local_map": None,
    },
    "max_hz": {
        "world/global_map": 1.0,
    },
    "memory_limit": "64MB",
    "static": {
        "world/robot_body": _static_g1_body,
    },
}


def _default_recording_dir() -> Path:
    override = os.getenv("DIMOS_G1_RECORDING_DIR")
    if override:
        return Path(override).expanduser()
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%Z")
    return RECORDINGS_DIR / f"g1-pointlio-{stamp}"


_RECORDING_DIR = _default_recording_dir()


unitree_g1_pointlio_record = (
    autoconnect(
        vis_module(viewer_backend=global_config.viewer, rerun_config=_RERUN_CONFIG),
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
        RayTracingVoxelMap.blueprint(
            voxel_size=_VOXEL_SIZE,
            emit_every=0,
            global_emit_every=5,
        ),
        PointlioRecorder.blueprint(db_path=str(_RECORDING_DIR / "mem2.db")),
    )
    .remappings(
        [
            (RayTracingVoxelMap, "lidar", "pointlio_lidar"),
            (RayTracingVoxelMap, "odometry", "pointlio_odometry"),
        ]
    )
    .global_config(n_workers=9, robot_model="unitree_g1")
)


def main() -> None:
    _RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Recording G1 PointLIO data to {_RECORDING_DIR / 'mem2.db'}")
    print("Publishing the accumulated global map to DimOS Viewer.")
    print("Locomotion remains under the Unitree handheld controller.")
    coordinator = ModuleCoordinator.build(unitree_g1_pointlio_record)
    coordinator.loop()


if __name__ == "__main__":
    main()
