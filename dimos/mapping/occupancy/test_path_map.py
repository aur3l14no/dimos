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


import cv2
import numpy as np
import pytest

from dimos.mapping.occupancy.gradient import GradientStrategy
from dimos.mapping.occupancy.path_map import make_navigation_map
from dimos.mapping.occupancy.types import NavigationStrategy
from dimos.mapping.occupancy.visualizations import visualize_occupancy_grid
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.utils.data import get_data


@pytest.mark.parametrize("strategy", ["simple", "mixed"])
def test_make_navigation_map(occupancy: OccupancyGrid, strategy: NavigationStrategy) -> None:
    expected = cv2.imread(get_data(f"make_navigation_map_{strategy}.png"), cv2.IMREAD_COLOR)
    robot_width = 0.4

    og = make_navigation_map(occupancy, robot_width, strategy=strategy, gradient_strategy="voronoi")

    result = visualize_occupancy_grid(og, "rainbow")
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.parametrize("gradient_strategy", ["gradient", "voronoi"])
def test_navigation_map_preserves_nonlethal_terrain_costs(
    gradient_strategy: GradientStrategy,
) -> None:
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[2, 2] = CostValues.UNKNOWN
    grid[4, 4] = 60
    occupancy = OccupancyGrid(grid, resolution=0.1)

    navigation = make_navigation_map(
        occupancy,
        robot_width=0.0,
        strategy="simple",
        gradient_strategy=gradient_strategy,
    )

    assert navigation.grid[4, 4] == 60
    assert navigation.grid[2, 2] == CostValues.UNKNOWN
    known_free = np.ones_like(navigation.grid, dtype=bool)
    known_free[2, 2] = False
    known_free[4, 4] = False
    assert np.all(navigation.grid[known_free] == CostValues.FREE)
    assert not np.any(navigation.grid == CostValues.OCCUPIED)


@pytest.mark.parametrize("gradient_strategy", ["gradient", "voronoi"])
def test_navigation_map_only_uses_lethal_cells_as_obstacle_seeds(
    gradient_strategy: GradientStrategy,
) -> None:
    grid = np.zeros((15, 15), dtype=np.int8)
    grid[7, 3] = 60
    grid[7, 11] = CostValues.OCCUPIED
    occupancy = OccupancyGrid(grid, resolution=0.1)

    navigation = make_navigation_map(
        occupancy,
        robot_width=0.0,
        strategy="simple",
        gradient_strategy=gradient_strategy,
    )

    assert navigation.grid[7, 3] == 60
    assert navigation.grid[7, 11] == CostValues.OCCUPIED
