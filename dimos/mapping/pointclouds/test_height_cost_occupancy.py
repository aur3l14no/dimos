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

import numpy as np

from dimos.mapping.pointclouds.occupancy import height_cost_occupancy
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def _flat_floor_with_hole(*, resolution: float, hole_radius: float) -> PointCloud2:
    coordinates = np.arange(-2.0, 2.0 + resolution / 2, resolution)
    x, y = np.meshgrid(coordinates, coordinates)
    observed = np.hypot(x, y) >= hole_radius
    points = np.column_stack(
        (
            x[observed],
            y[observed],
            np.full(np.count_nonzero(observed), -1.2),
        )
    )
    return PointCloud2.from_numpy(points, frame_id="odom")


def test_height_cost_does_not_turn_unknown_boundary_into_obstacle() -> None:
    costmap = height_cost_occupancy(
        _flat_floor_with_hole(resolution=0.08, hole_radius=0.8),
        resolution=0.08,
        can_climb=0.10,
        ignore_noise=0.05,
        smoothing=1.0,
    )

    assert np.any(costmap.grid == CostValues.UNKNOWN)
    assert np.any(costmap.grid == CostValues.FREE)
    assert not np.any(costmap.grid > CostValues.FREE)


def test_height_cost_still_detects_observed_step() -> None:
    resolution = 0.08
    coordinates = np.arange(-1.0, 1.0 + resolution / 2, resolution)
    x, y = np.meshgrid(coordinates, coordinates)
    z = np.where(x < 0.0, 0.0, 0.3)
    cloud = PointCloud2.from_numpy(
        np.column_stack((x.ravel(), y.ravel(), z.ravel())),
        frame_id="odom",
    )

    costmap = height_cost_occupancy(
        cloud,
        resolution=resolution,
        can_climb=0.10,
        ignore_noise=0.05,
        smoothing=1.0,
    )

    assert np.any(costmap.grid == CostValues.OCCUPIED)
