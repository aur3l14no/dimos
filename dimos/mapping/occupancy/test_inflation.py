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

from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.mapping.occupancy.visualizations import visualize_occupancy_grid
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.utils.data import get_data


def test_inflation(occupancy) -> None:
    expected = cv2.imread(get_data("inflation_simple.png"), cv2.IMREAD_COLOR)

    og = simple_inflate(occupancy, 0.2)

    result = visualize_occupancy_grid(og, "rainbow")
    np.testing.assert_array_equal(result.data, expected)


def test_inflation_uses_metric_radius_and_custom_obstacle_threshold() -> None:
    grid = np.zeros((15, 15), dtype=np.int8)
    grid[7, 7] = 50
    occupancy = OccupancyGrid(
        grid=grid,
        resolution=float(np.float32(0.08)),
    )

    inflated = simple_inflate(occupancy, 0.33, obstacle_threshold=50)

    assert inflated.grid[7, 11] == CostValues.OCCUPIED
    assert inflated.grid[7, 12] == CostValues.FREE


def test_inflation_does_not_include_cells_below_custom_threshold() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = 49
    occupancy = OccupancyGrid(grid=grid, resolution=0.08)

    inflated = simple_inflate(occupancy, 0.33, obstacle_threshold=50)

    np.testing.assert_array_equal(inflated.grid, grid)


def test_inflation_includes_exact_float32_radius_boundary() -> None:
    grid = np.zeros((13, 13), dtype=np.int8)
    grid[6, 6] = CostValues.OCCUPIED
    occupancy = OccupancyGrid(
        grid=grid,
        resolution=float(np.float32(0.05)),
    )

    inflated = simple_inflate(occupancy, 0.2)

    assert inflated.grid[6, 10] == CostValues.OCCUPIED
    assert inflated.grid[6, 11] == CostValues.FREE
