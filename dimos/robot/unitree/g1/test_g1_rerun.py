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
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.robot.unitree.g1.g1_rerun import (
    g1_pointlio_ground_z,
    g1_pointlio_pelvis_transform,
)


def test_pointlio_pelvis_transform_matches_g1_mount_geometry() -> None:
    yaw = 0.7
    mount_pitch = 0.04014257279586953
    pelvis_from_mid360_translation = np.array([-0.00368, 0.00003, 0.46018])
    cos_pitch = np.cos(mount_pitch)
    sin_pitch = np.sin(mount_pitch)
    pelvis_from_mid360_rotation = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ]
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    world_from_pelvis_rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ground_z = g1_pointlio_ground_z()
    world_from_pelvis_translation = np.array([1.5, -2.0, ground_z + 0.74])
    world_from_mid360_rotation = world_from_pelvis_rotation @ pelvis_from_mid360_rotation
    world_from_mid360_translation = (
        world_from_pelvis_translation + world_from_pelvis_rotation @ pelvis_from_mid360_translation
    )

    # PointLIO reports the raw orientation of the physically upside-down sensor.
    upside_down = np.diag([1.0, -1.0, -1.0])
    raw_orientation = Quaternion.from_rotation_matrix(world_from_mid360_rotation @ upside_down)
    odometry = Odometry(
        frame_id="odom",
        pose=Pose(
            *world_from_mid360_translation,
            *raw_orientation.to_tuple(),
        ),
    )

    transform = g1_pointlio_pelvis_transform(odometry)

    translation = transform.translation.as_arrow_array().to_pylist()[0]
    assert translation == pytest.approx(world_from_pelvis_translation)
    actual_quaternion = np.asarray(transform.quaternion.as_arrow_array().to_pylist()[0])
    expected_orientation = Quaternion.from_rotation_matrix(world_from_pelvis_rotation)
    expected_quaternion = np.asarray(
        [
            expected_orientation.x,
            expected_orientation.y,
            expected_orientation.z,
            expected_orientation.w,
        ]
    )
    assert abs(float(np.dot(actual_quaternion, expected_quaternion))) == pytest.approx(1.0)
    assert ground_z == pytest.approx(-1.20018)
    assert translation[2] - ground_z == pytest.approx(0.74)
