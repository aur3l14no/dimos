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

"""Python interface to the Rust voxel ray-tracing mapper."""

from __future__ import annotations

__all__ = ["VoxelRayMapper", "local_bounds"]

try:
    from dimos_voxel_ray_tracing import (
        VoxelRayMapper,
        local_bounds,
    )
except ImportError as e:
    raise ImportError(
        "dimos_voxel_ray_tracing is unavailable. The ray-tracing map tools require "
        "the Nix-built PyO3 extension from this checkout. Build RayTracingVoxelMap "
        "with its configured build command, then expose "
        "dimos/mapping/ray_tracing/rust/result/lib/python3.12/site-packages to the "
        "Python environment running `dimos map`."
    ) from e
