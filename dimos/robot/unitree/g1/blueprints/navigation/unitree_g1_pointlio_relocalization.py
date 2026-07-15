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

"""G1 PointLIO navigation localized against a recorded global map."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.relocalization.module import RelocalizationModule
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_pointlio_nav import (
    unitree_g1_pointlio_nav,
)
from dimos.robot.unitree.g1.config import G1

assert G1.width_clearance is not None

_VOXEL_SIZE = 0.08
_ROTATION_DIAMETER = 0.8


unitree_g1_pointlio_relocalization = (
    autoconnect(
        unitree_g1_pointlio_nav,
        # The base navigation stack only emits the persistent global map. This
        # variant also emits the cropped local view used for matching.
        RayTracingVoxelMap.blueprint(
            voxel_size=_VOXEL_SIZE,
            emit_every=1,
            global_emit_every=5,
        ),
        RelocalizationModule.blueprint(
            publish_loaded_map=True,
            matching_map="local_map",
            fitness_threshold=0.45,
            live_frame="odom",
            merge_voxel_size=_VOXEL_SIZE,
            require_map_file=True,
            lock_after_first=True,
            gravity_aligned_4dof=True,
            min_local_points=25_000,
        ),
        # Keep every goal entry point fail-closed until the first merged map is
        # available. This also covers the planner's set_goal RPC.
        ReplanningAStarPlanner.blueprint(
            robot_width=G1.width_clearance,
            robot_rotation_diameter=_ROTATION_DIAMETER,
            require_navigation_enabled=True,
        ),
    )
    .remappings(
        [
            (CostMapper, "merged_costmap_ready", "localization_ready"),
            (ReplanningAStarPlanner, "navigation_enabled", "localization_ready"),
        ]
    )
    .global_config(n_workers=11, robot_model="unitree_g1")
)
