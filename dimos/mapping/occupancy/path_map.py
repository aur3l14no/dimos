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

import numpy as np

from dimos.mapping.occupancy.gradient import GradientStrategy, gradient, voronoi_gradient
from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.mapping.occupancy.operations import overlay_occupied, smooth_occupied
from dimos.mapping.occupancy.types import NavigationStrategy
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid


def _preserve_traversal_costs(
    clearance_map: OccupancyGrid, terrain_map: OccupancyGrid
) -> OccupancyGrid:
    """Keep non-lethal terrain costs alongside obstacle-clearance costs."""
    known = terrain_map.grid != CostValues.UNKNOWN
    clearance_map.grid[known] = np.maximum(
        clearance_map.grid[known],
        terrain_map.grid[known],
    )
    return clearance_map


def make_navigation_map(
    occupancy_grid: OccupancyGrid,
    robot_width: float,
    strategy: NavigationStrategy,
    gradient_strategy: GradientStrategy,
) -> OccupancyGrid:
    half_width = robot_width / 2
    gradient_distance = 1.5

    if strategy == "simple":
        costmap = simple_inflate(occupancy_grid, half_width)
    elif strategy == "mixed":
        costmap = smooth_occupied(occupancy_grid)
        costmap = simple_inflate(costmap, half_width)
        costmap = overlay_occupied(costmap, occupancy_grid)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Occupancy values 1..99 are traversable terrain costs, not obstacle
    # seeds. Only lethal cells are inflated and converted to hard obstacles;
    # the graded terrain costs are overlaid onto the clearance field below.
    if gradient_strategy == "gradient":
        clearance_map = gradient(
            costmap,
            obstacle_threshold=CostValues.OCCUPIED,
            max_distance=gradient_distance,
        )
    elif gradient_strategy == "voronoi":
        clearance_map = voronoi_gradient(
            costmap,
            obstacle_threshold=CostValues.OCCUPIED,
            max_distance=gradient_distance,
        )
    else:
        raise ValueError(f"Unknown gradient strategy: {gradient_strategy}")

    return _preserve_traversal_costs(clearance_map, costmap)
