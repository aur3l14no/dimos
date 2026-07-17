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
from scipy import ndimage

from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid


def simple_inflate(
    occupancy_grid: OccupancyGrid,
    radius: float,
    obstacle_threshold: int = CostValues.OCCUPIED,
) -> OccupancyGrid:
    """Inflate obstacles by a given radius (binary inflation).

    Args:
        radius: Inflation radius in meters
        obstacle_threshold: Values at or above this threshold are obstacles.

    Returns:
        New OccupancyGrid with inflated obstacles
    """
    if radius < 0:
        raise ValueError("Inflation radius must be non-negative")
    if occupancy_grid.resolution <= 0:
        raise ValueError("Occupancy-grid resolution must be positive")

    # Use the requested metric radius when drawing the kernel. Rounding the
    # radius up first over-inflates by as much as one full grid cell and is
    # especially surprising when a float32 resolution lies just below an
    # exact decimal value (for example 0.079999998 instead of 0.08).
    radius_cells = radius / occupancy_grid.resolution
    cell_extent = int(np.ceil(radius_cells))

    # Get grid as numpy array
    grid_array = occupancy_grid.grid

    # Create circular kernel for binary inflation
    y, x = np.ogrid[-cell_extent : cell_extent + 1, -cell_extent : cell_extent + 1]
    radius_squared = radius_cells**2
    radius_tolerance = max(1e-9, radius_squared * 1e-6)
    kernel = (x**2 + y**2 <= radius_squared + radius_tolerance).astype(np.uint8)

    # Find occupied cells
    occupied_mask = grid_array >= obstacle_threshold

    # Binary inflation
    inflated = ndimage.binary_dilation(occupied_mask, structure=kernel)
    result_grid = grid_array.copy()
    result_grid[inflated] = CostValues.OCCUPIED

    # Create new OccupancyGrid with inflated data using numpy constructor
    return OccupancyGrid(
        grid=result_grid,
        resolution=occupancy_grid.resolution,
        origin=occupancy_grid.origin,
        frame_id=occupancy_grid.frame_id,
        ts=occupancy_grid.ts,
    )
