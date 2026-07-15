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

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.mapping.utils.cli.map import (
    _observation_pose_registered,
    _prepare_raytrace_frame,
    _resolve_registration,
)
from dimos.memory2.type.observation import Observation
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def _obs(
    points: NDArray[np.float32],
    *,
    pose: tuple[float, float, float, float, float, float, float] | None,
    frame_id: str = "mid360_link",
) -> Observation[PointCloud2]:
    return Observation(
        id=1,
        ts=10.0,
        pose=pose,
        _data=PointCloud2.from_numpy(points, frame_id=frame_id, timestamp=10.0),
    )


@pytest.fixture
def origin_observation() -> Observation[PointCloud2]:
    return _obs(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )


def test_observation_pose_registration_accepts_world_origin_and_preserves_pose(
    origin_observation: Observation[PointCloud2],
) -> None:
    world, cloud_frame, register = _resolve_registration(
        origin_observation,
        mode="observation-pose",
        requested_world="odom",
        tf_buf=None,
        tf_tolerance=None,
    )

    [registered] = list(
        _observation_pose_registered(
            iter([origin_observation]),
            world_frame=world,
            child_frame=cloud_frame or "sensor",
        )
    )

    assert register is not None
    assert registered.pose_tuple == origin_observation.pose_tuple
    assert registered.data.frame_id == "odom"
    np.testing.assert_allclose(registered.data.points_f32(), [[1.0, 0.0, 0.0]])


def test_observation_pose_registration_requires_named_world_frame(
    origin_observation: Observation[PointCloud2],
) -> None:
    with pytest.raises(ValueError, match="--frame is required"):
        _resolve_registration(
            origin_observation,
            mode="observation-pose",
            requested_world=None,
            tf_buf=None,
            tf_tolerance=None,
        )


def test_raytrace_frame_corrects_endpoints_and_ray_origin() -> None:
    class TranslationGraph:
        def correction_at(self, ts: float) -> Transform:
            return Transform(
                translation=Vector3(10.0, 0.0, 0.0),
                rotation=Quaternion(),
                frame_id="odom",
                child_frame_id="odom_raw",
                ts=ts,
            )

    obs = _obs(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        pose=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    _, _, register = _resolve_registration(
        obs,
        mode="observation-pose",
        requested_world="odom",
        tf_buf=None,
        tf_tolerance=None,
    )
    prepared = _prepare_raytrace_frame(
        obs,
        world_frame="odom",
        graph=TranslationGraph(),  # type: ignore[arg-type]
        register=register,
    )

    assert prepared is not None
    points, origin = prepared
    np.testing.assert_allclose(points, [[13.0, 0.0, 0.0]])
    assert origin == (12.0, 0.0, 0.0)


def test_raytrace_world_endpoints_receive_only_pgo_correction() -> None:
    class TranslationGraph:
        def correction_at(self, ts: float) -> Transform:
            return Transform(
                translation=Vector3(10.0, 0.0, 0.0),
                rotation=Quaternion(),
                frame_id="odom",
                child_frame_id="odom_raw",
                ts=ts,
            )

    obs = _obs(
        np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
        pose=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        frame_id="odom",
    )

    prepared = _prepare_raytrace_frame(
        obs,
        world_frame="odom",
        graph=TranslationGraph(),  # type: ignore[arg-type]
        register=None,
    )

    assert prepared is not None
    points, origin = prepared
    np.testing.assert_allclose(points, [[15.0, 0.0, 0.0]])
    assert origin == (12.0, 0.0, 0.0)
