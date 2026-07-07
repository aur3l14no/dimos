#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.navigation.cmu_nav.main import cmu_nav_rerun_config, create_cmu_nav
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.g1.blueprints.primitive.unitree_g1_onboard import _unitree_g1_onboard
from dimos.robot.unitree.g1.config import G1, G1_LOCAL_PLANNER_PRECOMPUTED_PATHS
from dimos.robot.unitree.g1.g1_rerun import (
    g1_odometry_tf_override,
    g1_static_robot,
)
from dimos.visualization.vis_module import vis_module

unitree_g1_nav_onboard = (
    autoconnect(
        _unitree_g1_onboard,
        # _unitree_g1_onboard already includes FastLio2. autoconnect de-duplicates
        # modules by keeping the later blueprint, so this overrides that FastLio2
        # config for nav without changing the primitive/onboard default.
        FastLio2.blueprint(
            host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
            lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
            scan_publish_en=False,
            registered_scan_publish_en=True,
        ),
        create_cmu_nav(
            planner="simple",
            vehicle_height=G1.height_clearance,
            max_speed=0.6,
            far_planner={
                "is_static_env": False,
            },
            terrain_analysis={
                "obstacle_height_threshold": 0.01,
                "ground_height_threshold": 0.01,
                "sensor_range": 40,  # meters
            },
            local_planner={
                "paths_dir": str(G1_LOCAL_PLANNER_PRECOMPUTED_PATHS),
                "publish_free_paths": False,
            },
            simple_planner={
                # FastLio2 publishes odom -> mid360_link (no separate body/sensor alias).
                "body_frame": "mid360_link",
                "sensor_frame": "mid360_link",
                "cell_size": 0.2,
                "obstacle_height_threshold": 0.10,
                "inflation_radius": 0.5,
                "lookahead_distance": 2.0,
                "replan_rate": 5.0,
                "replan_cooldown": 2.0,
            },
        ),
        MovementManager.blueprint(),
        vis_module(
            viewer_backend=global_config.viewer,
            rerun_config=cmu_nav_rerun_config(
                {
                    "visual_override": {"world/odometry": g1_odometry_tf_override},
                    "static": {"world/tf/robot": g1_static_robot},
                    "memory_limit": "1GB",
                },
                vis_throttle=0.5,
            ),
        ),
    )
    .remappings(
        [
            # Planner owns way_point — disconnect MovementManager's click relay
            (MovementManager, "way_point", "_mgr_way_point_unused"),
        ]
    )
    .global_config(n_workers=12, robot_model="unitree_g1")
)
