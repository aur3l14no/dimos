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

from dataclasses import asdict

from dimos_lcm.std_msgs import Bool
import numpy as np
from pydantic import Field
from reactivex import combine_latest, operators as ops

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.pointclouds.occupancy import (
    OCCUPANCY_ALGOS,
    HeightCostConfig,
    OccupancyConfig,
)
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


class Config(ModuleConfig):
    algo: str = "height_cost"
    config: OccupancyConfig = Field(default_factory=HeightCostConfig)
    # for robots that cant see directly below themself
    initial_safe_radius_meters: float = 0.0


class CostMapper(Module):
    config: Config
    global_map: In[PointCloud2]
    merged_map: In[PointCloud2]
    global_costmap: Out[OccupancyGrid]
    merged_costmap_ready: Out[Bool]

    @rpc
    def start(self) -> None:
        super().start()

        self.register_disposable(
            combine_latest(
                self.global_map.observable(),  # type: ignore[no-untyped-call]
                self.merged_map.observable().pipe(ops.start_with(None)),  # type: ignore[no-untyped-call,arg-type]
            ).subscribe(self._on_map_pair)
        )

    def _on_map_pair(self, pair: tuple[PointCloud2, PointCloud2 | None]) -> None:
        global_map, merged_map = pair
        grid = self._calculate_costmap(merged_map if merged_map is not None else global_map)
        self.global_costmap.publish(grid)
        if merged_map is not None:
            self.merged_costmap_ready.publish(Bool(True))

    @rpc
    def stop(self) -> None:
        super().stop()

    # @timed()  # TODO: fix thread leak in timed decorator
    def _calculate_costmap(self, msg: PointCloud2) -> OccupancyGrid:
        occupancy_function = OCCUPANCY_ALGOS[self.config.algo]
        grid = occupancy_function(msg, **asdict(self.config.config))
        self._apply_initial_safe_radius(grid)
        return grid

    def _apply_initial_safe_radius(self, grid: OccupancyGrid) -> None:
        radius_meters = self.config.initial_safe_radius_meters
        if radius_meters <= 0 or grid.grid.size == 0:
            return

        resolution = grid.resolution
        origin_x = grid.origin.position.x
        origin_y = grid.origin.position.y

        rows, columns = np.ogrid[: grid.grid.shape[0], : grid.grid.shape[1]]
        cell_world_x = columns * resolution + origin_x
        cell_world_y = rows * resolution + origin_y
        distance_squared_meters = cell_world_x**2 + cell_world_y**2

        # Half-cell tolerance: a cell counts as inside if any part of it overlaps
        # the disc. Avoids floating-point boundary flakiness from radius/resolution.
        effective_radius_meters = radius_meters + resolution * 0.5
        safe_mask = distance_squared_meters <= effective_radius_meters**2
        grid.grid[safe_mask] = 0
