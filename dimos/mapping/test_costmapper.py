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

from collections.abc import Generator

import numpy as np
import pytest

from dimos.mapping.costmapper import CostMapper
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


@pytest.fixture
def cost_mapper() -> Generator[CostMapper, None, None]:
    module = CostMapper()
    yield module
    module._close_module()


def _cloud(z: float) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray([[0.0, 0.0, z]], dtype=np.float32),
        frame_id="odom",
        timestamp=1.0,
    )


def test_merged_costmap_readiness_is_published_after_calculation(
    monkeypatch: pytest.MonkeyPatch,
    cost_mapper: CostMapper,
) -> None:
    calculated_from: list[PointCloud2] = []
    ready: list[bool] = []
    costmaps: list[OccupancyGrid] = []

    def calculate(msg: PointCloud2) -> OccupancyGrid:
        calculated_from.append(msg)
        return OccupancyGrid(width=1, height=1, frame_id=msg.frame_id)

    monkeypatch.setattr(cost_mapper, "_calculate_costmap", calculate)
    cost_mapper.global_costmap.subscribe(costmaps.append)
    cost_mapper.merged_costmap_ready.subscribe(lambda msg: ready.append(msg.data))
    live, merged = _cloud(0.0), _cloud(1.0)

    cost_mapper._on_map_pair((live, None))
    assert calculated_from == [live]
    assert ready == []

    cost_mapper._on_map_pair((live, merged))
    assert calculated_from == [live, merged]
    assert len(costmaps) == 2
    assert ready == [True]
