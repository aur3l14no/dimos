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

"""G1-specific Rerun visual helpers (robot dimensions, TF overrides)."""

from __future__ import annotations

from typing import Any

import numpy as np

from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.visualization.rerun.urdf_robot import (
    UrdfRobotJointStateRerunFactory,
    UrdfRobotStaticRerunFactory,
)

G1_RERUN_ROOT = "world/odom/g1"
G1_RERUN_URDF = "g1_urdf/g1.fixed.urdf"

# Rest-pose pelvis -> Mid-360 transform from g1.urdf. Keep this explicit because
# the URDF visualization dependency (yourdfpy) is unavailable on Linux aarch64.
_G1_MID360_PITCH = 0.04014257279586953
_G1_PELVIS_TO_MID360 = np.array(
    [
        [np.cos(_G1_MID360_PITCH), 0.0, np.sin(_G1_MID360_PITCH), -0.00368],
        [0.0, 1.0, 0.0, 0.00003],
        [-np.sin(_G1_MID360_PITCH), 0.0, np.cos(_G1_MID360_PITCH), 0.46018],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
_G1_MID360_TO_PELVIS = np.linalg.inv(_G1_PELVIS_TO_MID360)
_G1_MID360_UPSIDE_DOWN = np.diag([1.0, -1.0, -1.0])
_G1_NOMINAL_PELVIS_Z = 0.74

# Classic costmap palette, indexed by grid value + 1:
# transparent unknown, blue free, orange occupied, red lethal.
_COSTMAP_LOOKUP_TABLE = np.zeros((102, 4), dtype=np.uint8)
_COSTMAP_LOOKUP_TABLE[0] = (0, 0, 0, 0)
_COSTMAP_LOOKUP_TABLE[1] = (72, 73, 129, 255)
_COSTMAP_LOOKUP_TABLE[2:101] = (255, 140, 0, 255)
_COSTMAP_LOOKUP_TABLE[101] = (220, 30, 30, 255)


def g1_costmap(grid: Any, z_offset: float = 0.02) -> Any:
    """Render an OccupancyGrid with the classic costmap palette.

    The default z_offset lifts the mesh 2cm off the floor plane to avoid
    z-fighting with the ground.
    """
    return grid.to_rerun(color_lookup_table=_COSTMAP_LOOKUP_TABLE, z_offset=z_offset)


def g1_urdf_static_robot(root_path: str = G1_RERUN_ROOT) -> UrdfRobotStaticRerunFactory:
    """Create a static Rerun logger for the G1 URDF visual meshes."""
    return UrdfRobotStaticRerunFactory(urdf_path=G1_RERUN_URDF, root_path=root_path)


def g1_urdf_joint_state(root_path: str = G1_RERUN_ROOT) -> UrdfRobotJointStateRerunFactory:
    """Create a Rerun JointState converter for the G1 URDF."""
    return UrdfRobotJointStateRerunFactory(urdf_path=G1_RERUN_URDF, root_path=root_path)


def g1_static_robot(rr: Any) -> list[Any]:
    """Static G1 humanoid wireframe box attached to the sensor TF frame.

    Half-sizes are ~50x40x120 cm (the G1 humanoid), and the box is
    centered 0.6m below the sensor (lidar mounted at head height).
    """
    return [
        rr.Boxes3D(
            half_sizes=[0.25, 0.20, 0.6],
            centers=[[0, 0, -0.6]],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/sensor"),
    ]


def g1_pointlio_static_body(rr: Any) -> list[Any]:
    """Static G1 wireframe expressed in the pelvis frame.

    The box extends from the nominal ground contact at -0.74m to the Mid-360
    mount at +0.46m. Its entity path must be a child of the dynamic pelvis
    transform produced by :func:`g1_pointlio_pelvis_transform`.
    """
    return [
        rr.Boxes3D(
            half_sizes=[0.25, 0.20, 0.6],
            centers=[[0, 0, -0.14]],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        )
    ]


def g1_pointlio_pelvis_transform(odom: Any) -> Any:
    """Convert upside-down Mid-360 odometry into the G1 pelvis pose."""
    import rerun as rr

    world_from_mid360 = np.eye(4)
    world_from_mid360[:3, :3] = odom.orientation.to_rotation_matrix() @ _G1_MID360_UPSIDE_DOWN
    world_from_mid360[:3, 3] = (odom.x, odom.y, odom.z)
    world_from_pelvis = world_from_mid360 @ _G1_MID360_TO_PELVIS
    orientation = Quaternion.from_rotation_matrix(world_from_pelvis[:3, :3])
    return rr.Transform3D(
        translation=world_from_pelvis[:3, 3].tolist(),
        rotation=rr.Quaternion(xyzw=[orientation.x, orientation.y, orientation.z, orientation.w]),
    )


def g1_pointlio_ground_z() -> float:
    """Nominal ground height in PointLIO's Mid-360 boot frame."""
    return -(float(_G1_PELVIS_TO_MID360[2, 3]) + _G1_NOMINAL_PELVIS_Z)


def g1_pointlio_costmap(grid: Any) -> Any:
    """Render a PointLIO costmap on the G1's nominal ground plane."""
    return g1_costmap(grid, z_offset=g1_pointlio_ground_z() + 0.02)


def g1_odometry_tf_override(odom: Any) -> Any:
    """Publish odometry as a TF frame so sensor_scan/path/robot can reference it.

    The z is zeroed because point clouds already have the full init_pose
    transform applied (ground at z≈0). Using the raw odom.z (= mount height)
    would double-count the vertical offset.
    """
    import rerun as rr

    tf = rr.Transform3D(
        translation=[odom.x, odom.y, 0.0],
        rotation=rr.Quaternion(
            xyzw=[
                odom.orientation.x,
                odom.orientation.y,
                odom.orientation.z,
                odom.orientation.w,
            ]
        ),
        parent_frame="tf#/map",
        child_frame="tf#/sensor",
    )
    return [
        ("tf#/sensor", tf),
    ]
