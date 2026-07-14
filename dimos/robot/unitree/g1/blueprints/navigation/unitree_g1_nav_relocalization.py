#!/usr/bin/env python3
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

"""Standalone G1 PointLIO navigation with static-map relocalization."""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.relocalization.module import RelocalizationModule
from dimos.navigation.cmu_nav.frames import FRAME_ODOM
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_pointlio_nav import (
    unitree_g1_pointlio_nav,
)

unitree_g1_nav_relocalization = autoconnect(
    unitree_g1_pointlio_nav,
    RelocalizationModule.blueprint(
        map_file=os.getenv("DIMOS_G1_MAP_FILE"),
        publish_loaded_map=True,
        fitness_threshold=0.45,
        live_frame=FRAME_ODOM,
    ),
)
