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

"""Standalone G1 navigation with PointLIO, ray tracing, and A* replanning."""

import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import HeightCostConfig
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.g1_rerun import (
    g1_pointlio_costmap,
    g1_pointlio_ground_z,
    g1_pointlio_pelvis_transform,
    g1_pointlio_static_body,
)
from dimos.visualization.vis_module import vis_module

assert G1.height_clearance is not None and G1.width_clearance is not None

_VOXEL_SIZE = 0.08
_OVERHEAD_SAFETY_MARGIN = 0.2
_MAX_STEP_HEIGHT = 0.10
_SAFE_RADIUS_MARGIN = 0.6
_ROTATION_DIAMETER = 0.8


def _render_global_map(msg: PointCloud2) -> Any:
    return msg.to_rerun(voxel_size=_VOXEL_SIZE)


def _render_loaded_map(msg: PointCloud2) -> Any:
    return msg.to_rerun(voxel_size=_VOXEL_SIZE, colors=[70, 130, 255])


def _render_merged_map(msg: PointCloud2) -> Any:
    return msg.to_rerun(voxel_size=_VOXEL_SIZE, colors=[80, 220, 140])


def _render_costmap(msg: OccupancyGrid) -> Any:
    return g1_pointlio_costmap(msg)


def _render_path(msg: Path) -> Any:
    return msg.to_rerun(z_offset=g1_pointlio_ground_z() + 0.3)


def _g1_pointlio_rerun_blueprint() -> Any:
    # Rerun is an optional viewer dependency, so load it only when the layout is built.
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name="G1 PointLIO navigation",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(
                plane=rr.components.Plane3D.XY.with_distance(g1_pointlio_ground_z()),
            ),
        ),
        rrb.TimePanel(state="collapsed"),
    )


_rerun_config = {
    "blueprint": _g1_pointlio_rerun_blueprint,
    "visual_override": {
        "world/global_map": _render_global_map,
        "world/loaded_map": _render_loaded_map,
        "world/merged_map": _render_merged_map,
        "world/global_costmap": _render_costmap,
        "world/navigation_costmap": _render_costmap,
        "world/path": _render_path,
        "world/lidar": None,
        "world/deskewed_lidar": None,
        "world/local_map": None,
        "world/lidar_odometry": g1_pointlio_pelvis_transform,
    },
    "max_hz": {
        "world/path": 0,
    },
    "memory_limit": "64MB",
    "static": {
        "world/lidar_odometry/g1": g1_pointlio_static_body,
    },
}

unitree_g1_pointlio_nav = (
    autoconnect(
        vis_module(viewer_backend=global_config.viewer, rerun_config=_rerun_config),
        PointLio.blueprint(
            pointcloud_freq=5.0,
            scan_publish_en=False,
            deskewed_scan_publish_en=True,
            host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
            lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
        ),
        RayTracingVoxelMap.blueprint(
            voxel_size=_VOXEL_SIZE,
            emit_every=0,
            global_emit_every=5,
        ),
        CostMapper.blueprint(
            config=HeightCostConfig(
                resolution=_VOXEL_SIZE,
                can_pass_under=G1.height_clearance + _OVERHEAD_SAFETY_MARGIN,
                can_climb=_MAX_STEP_HEIGHT,
            ),
            initial_safe_radius_meters=G1.width_clearance + _SAFE_RADIUS_MARGIN,
        ),
        ReplanningAStarPlanner.blueprint(
            robot_width=G1.width_clearance,
            robot_rotation_diameter=_ROTATION_DIAMETER,
        ),
        MovementManager.blueprint(),
        G1HighLevelDdsSdk.blueprint(),
    )
    .remappings(
        [
            (RayTracingVoxelMap, "lidar", "deskewed_lidar"),
            (RayTracingVoxelMap, "odometry", "lidar_odometry"),
        ]
    )
    .global_config(n_workers=10, robot_model="unitree_g1")
)
