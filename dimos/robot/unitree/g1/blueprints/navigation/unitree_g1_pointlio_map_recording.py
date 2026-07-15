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

"""Record coherent G1 PointLIO scans while previewing the ray-traced map."""

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.lidar.pointlio.recorder import PointlioRecorder
from dimos.mapping.costmapper import CostMapper
from dimos.memory2.module import OnExisting
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_pointlio_nav import (
    unitree_g1_pointlio_nav,
)
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk

unitree_g1_pointlio_map_recording = (
    autoconnect(
        unitree_g1_pointlio_nav.disabled_modules(
            CostMapper,
            ReplanningAStarPlanner,
            MovementManager,
            G1HighLevelDdsSdk,
        ),
        PointlioRecorder.blueprint(
            db_path=RECORDINGS_DIR / "g1_pointlio_map.db",
            on_existing=OnExisting.BACKUP,
            root_frame="odom",
            record_tf=True,
            stream_remapping={
                "pointlio_lidar": "lidar",
                "pointlio_odometry": "lidar_odometry",
            },
            drop_unposed=True,
            require_exact_pose_stamp=True,
        ),
    )
    .remappings(
        [
            (PointlioRecorder, "pointlio_lidar", "deskewed_lidar"),
            (PointlioRecorder, "pointlio_odometry", "lidar_odometry"),
        ]
    )
    .global_config(n_workers=6, robot_model="unitree_g1")
)
