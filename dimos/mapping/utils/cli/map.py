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

from collections.abc import Callable, Iterable, Iterator
from functools import partial
import math
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, Literal, cast

import rerun as rr
import rerun.blueprint as rrb
import typer

# Heavy dimos imports (mapping/memory2 → torch, transformers, open3d) are
# deferred into the function bodies below so that `dimos --help` — which imports this
# module just to register the `map` subcommand — stays fast. See test_cli_startup.py.
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from dimos.mapping.loop_closure.pgo import PoseGraph
    from dimos.memory2.stream import Stream
    from dimos.memory2.type.observation import Observation
    from dimos.msgs.geometry_msgs.Transform import Transform
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

PATH_THICKNESS = 0.01
# Pin pattern (from dimos/memory2/vis/space/rerun.py): thin vertical line
# from each marker with the label floating at the top so multi-marker
# labels never overlap the boxes.
MARKER_STEM = 1.0

# Conventional world frames tried in order when --frame isn't given.
_WORLD_FRAMES = ("world", "map", "odom")

RegistrationMode = Literal["tf", "observation-pose"]


def _detect_world(tf_buf: Any, cloud_frame: str, ts: float) -> str | None:
    """Pick the first conventional world frame that resolves the cloud frame via tf."""
    if cloud_frame in _WORLD_FRAMES:
        return cloud_frame
    if tf_buf is not None:
        for cand in _WORLD_FRAMES:
            if tf_buf.get(cand, cloud_frame, time_point=ts) is not None:
                return cand
    return None


def _observation_pose_transform(
    obs: Observation[Any], *, world_frame: str, child_frame: str | None = None
) -> Transform | None:
    """Return the exact sensor pose stored on an observation.

    A missing pose or an uninitialized/non-finite quaternion is not a usable
    registration. The world origin is valid and must not be mistaken for a
    placeholder pose.
    """
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.geometry_msgs.Transform import Transform
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    pose = obs.pose_tuple
    if pose is None or not all(math.isfinite(value) for value in pose):
        return None
    x, y, z, qx, qy, qz, qw = pose
    if qx * qx + qy * qy + qz * qz + qw * qw < 1e-12:
        return None
    return Transform(
        translation=Vector3(x, y, z),
        rotation=Quaternion(qx, qy, qz, qw),
        frame_id=world_frame,
        child_frame_id=child_frame or obs.data.frame_id,
        ts=obs.ts,
    )


def _resolve_registration(
    first_obs: Observation[PointCloud2] | None,
    *,
    mode: RegistrationMode,
    requested_world: str | None,
    tf_buf: Any,
    tf_tolerance: float | None,
) -> tuple[str, str | None, Callable[[Observation[Any]], Transform | None] | None]:
    """Resolve the world frame and cloud registration strategy.

    The returned callback only transforms sensor-frame clouds. A ``None``
    callback means the input cloud is already expressed in the selected world
    frame.
    """
    cloud_frame = first_obs.data.frame_id if first_obs is not None else None

    if mode == "observation-pose":
        if requested_world is None:
            raise ValueError(
                "--frame is required with --registration=observation-pose because "
                "Observation.pose does not store its parent frame"
            )
        if cloud_frame is None or cloud_frame == requested_world:
            return requested_world, cloud_frame, None
        return (
            requested_world,
            cloud_frame,
            partial(
                _observation_pose_transform,
                world_frame=requested_world,
                child_frame=cloud_frame,
            ),
        )

    world = requested_world
    if world is None and first_obs is not None and cloud_frame is not None:
        world = _detect_world(tf_buf, cloud_frame, first_obs.ts)
        if world is None:
            frames = tf_buf.get_frames() if tf_buf is not None else set()
            known = ", ".join(sorted(frames)) or "dataset has no tf stream"
            raise ValueError(
                f"none of {', '.join(_WORLD_FRAMES)} resolves {cloud_frame!r} clouds; "
                f"pass --frame (tf frames: {known})"
            )
    if world is None:
        world = "world"
    if first_obs is None or cloud_frame is None or cloud_frame == world:
        return world, cloud_frame, None

    probe = tf_buf.get(world, cloud_frame, time_point=first_obs.ts) if tf_buf is not None else None
    if probe is None:
        frames = tf_buf.get_frames() if tf_buf is not None else set()
        known = ", ".join(sorted(frames)) or "dataset has no tf stream"
        raise ValueError(
            f"cannot register {cloud_frame!r} clouds into {world!r} (tf frames: {known})"
        )

    def register_from_tf(obs: Observation[Any]) -> Transform | None:
        return cast(
            "Transform | None",
            tf_buf.get(
                world,
                cloud_frame,
                time_point=obs.ts,
                time_tolerance=tf_tolerance,
            ),
        )

    return world, cloud_frame, register_from_tf


def _trajectory_position(
    obs: Observation[Any],
    *,
    mode: RegistrationMode,
    world_frame: str,
    child_frame: str,
    register: Callable[[Observation[Any]], Transform | None] | None,
) -> tuple[float, float, float] | None:
    """Return the world-frame sensor position used by paths and deduplication."""
    tf = (
        register(obs)
        if mode == "tf" and register is not None
        else _observation_pose_transform(
            obs,
            world_frame=world_frame,
            child_frame=child_frame,
        )
    )
    if tf is None:
        return None
    return (tf.translation.x, tf.translation.y, tf.translation.z)


def _observation_pose_registered(
    obs_iter: Iterator[Observation[PointCloud2]],
    *,
    world_frame: str,
    child_frame: str,
) -> Iterator[Observation[PointCloud2]]:
    """Yield valid observation-pose clouds in the world frame for PGO."""
    for obs in obs_iter:
        tf = _observation_pose_transform(
            obs,
            world_frame=world_frame,
            child_frame=child_frame,
        )
        if tf is None:
            continue
        yield obs if obs.data.frame_id == world_frame else obs.derive(data=obs.data.transform(tf))


def _prepare_raytrace_frame(
    obs: Observation[PointCloud2],
    *,
    world_frame: str,
    graph: PoseGraph | None,
    register: Callable[[Observation[Any]], Transform | None] | None,
) -> tuple[NDArray[np.float32], tuple[float, float, float]] | None:
    """Prepare corrected world endpoints and their matching sensor origin."""
    import numpy as np

    if len(obs.data) == 0:
        return None

    sensor_pose = (
        register(obs)
        if register is not None
        else _observation_pose_transform(obs, world_frame=world_frame)
    )
    if sensor_pose is None:
        return None

    correction = graph.correction_at(obs.ts) if graph is not None else None
    corrected_pose = sensor_pose if correction is None else correction + sensor_pose
    points = obs.data.points_f32()
    endpoint_tf = corrected_pose if register is not None else correction
    if endpoint_tf is not None:
        matrix = endpoint_tf.to_matrix()
        rotation = matrix[:3, :3].astype(np.float32)
        translation = matrix[:3, 3].astype(np.float32)
        points = points @ rotation.T + translation

    origin = corrected_pose.translation
    return points, (origin.x, origin.y, origin.z)


def _raytrace_accumulate(
    obs_iter: Iterable[Observation[PointCloud2]],
    *,
    voxel: float,
    world_frame: str,
    graph: PoseGraph | None,
    register: Callable[[Observation[Any]], Transform | None] | None,
    progress_cb: Callable[[Observation[Any]], None] | None = None,
) -> PointCloud2 | None:
    """Rebuild a map by replaying every frame through the ray tracer.

    ``register`` supplies the raw-world sensor pose for sensor-frame clouds.
    PGO drift correction is composed onto both hit endpoints and ray origins.
    Already-world-frame clouds still require ``obs.pose`` because ray tracing
    cannot clear free space without a sensor origin.
    """
    from dimos.mapping.ray_tracing.voxel_map import VoxelRayMapper
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

    mapper = VoxelRayMapper(voxel_size=voxel, max_range=30.0)
    last_ts: float | None = None
    frame_count = 0

    for obs in obs_iter:
        if progress_cb is not None:
            progress_cb(obs)
        prepared = _prepare_raytrace_frame(
            obs,
            world_frame=world_frame,
            graph=graph,
            register=register,
        )
        if prepared is None:
            continue
        points, origin = prepared
        mapper.add_frame(points, origin)
        last_ts = obs.ts
        frame_count += 1

    if last_ts is None:
        return None
    print(f"raytrace: replayed {frame_count} posed frames")
    return PointCloud2.from_numpy(
        mapper.global_map(),
        frame_id=world_frame,
        timestamp=last_ts,
    )


def _log_markers(
    prefix: str,
    centers: list[tuple[float, float, float]],
    quats: list[tuple[float, float, float, float]],
    *,
    fill_half: list[tuple[float, float, float]],
    outline_half: list[tuple[float, float, float]],
    colors: list[tuple[int, int, int]],
    labels: list[str],
) -> None:
    """Render per-marker fill + outline + pin-stem + label as four static entities."""
    n = len(centers)
    pin_strips = [[(cx, cy, cz), (cx, cy, cz + MARKER_STEM)] for (cx, cy, cz) in centers]
    label_positions = [(cx, cy, cz + MARKER_STEM + 0.01) for (cx, cy, cz) in centers]
    rr.log(
        f"{prefix}/fill",
        rr.Boxes3D(
            centers=centers,
            half_sizes=fill_half,
            quaternions=quats,
            colors=colors,
            fill_mode=rr.components.FillMode.Solid,
        ),
        static=True,
    )
    rr.log(
        f"{prefix}/outline",
        rr.Boxes3D(
            centers=centers,
            half_sizes=outline_half,
            quaternions=quats,
            colors=[(255, 255, 255)] * n,
            fill_mode=rr.components.FillMode.MajorWireframe,
            radii=0.002,
        ),
        static=True,
    )
    rr.log(
        f"{prefix}/pin",
        rr.LineStrips3D(strips=pin_strips, colors=colors, radii=[0.005]),
        static=True,
    )
    rr.log(
        f"{prefix}/label",
        rr.Points3D(positions=label_positions, labels=labels, colors=colors, radii=[0.001] * n),
        static=True,
    )


def _accumulate(
    obs_iter: Iterable[Observation[PointCloud2]],
    *,
    voxel: float,
    block_count: int,
    device: str,
    graph: PoseGraph | None = None,
    register: Callable[[Observation[Any]], Transform | None] | None = None,
    carve_columns: bool = False,
    progress_cb: Callable[[Observation[Any]], None] | None = None,
) -> PointCloud2 | None:
    """Accumulate a voxel map from `obs_iter`, optionally PGO-correcting each frame.

    ``register`` maps each observation to the transform lifting its cloud into
    the world frame; ``None`` means no transform is available and the frame is
    skipped. With ``register=None`` all clouds are assumed world-registered.

    Returns the final ``PointCloud2`` (or ``None`` if the input was empty).
    Disposal of the underlying ``VoxelGrid`` is handled by ``VoxelMapTransformer``.
    """
    from dimos.mapping.voxels import VoxelMapTransformer

    def prepared() -> Iterable[Observation[PointCloud2]]:
        for obs in obs_iter:
            if progress_cb is not None:
                progress_cb(obs)
            if len(obs.data) == 0:
                continue
            # sensor->world via `register`, unless the clouds are already
            # world-registered. graph adds the PGO correction on top
            # (correction ∘ tf), applied after the registration.
            tf: Transform | None = None
            if register is not None:
                tf = register(obs)
                if tf is None:
                    continue
            if graph is not None:
                if obs.pose_tuple is None:
                    continue
                correction = graph.correction_at(obs.ts)
                tf = correction if tf is None else correction + tf
            yield obs if tf is None else obs.derive(data=obs.data.transform(tf))

    vmt = VoxelMapTransformer(
        emit_every=0,  # batch mode: emit once on exhaustion
        voxel_size=voxel,
        block_count=block_count,
        device=device,
        carve_columns=carve_columns,
    )
    result = next(iter(vmt(iter(prepared()))), None)
    return result.data if result is not None else None


def _denoise(cloud: PointCloud2 | None) -> PointCloud2 | None:
    """Statistical outlier removal via o3d; drops sparse floaters, keeps colors."""
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

    if cloud is None or len(cloud.pointcloud.points) < 20:
        return cloud
    clean, _ = cloud.pointcloud_tensor.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
    return PointCloud2(pointcloud=clean, frame_id=cloud.frame_id, ts=cloud.ts)


def _log_reconstruction(
    *,
    voxel: float,
    global_map: PointCloud2 | None,
    path: list[tuple[float, float, float]],
    pgo_map: PointCloud2 | None,
    full_pgo_map: PointCloud2 | None,
    pgo_path: list[tuple[float, float, float]],
    graph: PoseGraph | None,
    marker_dets: list[Observation[Any]],
    marker_size: float,
    bottom_cutoff: float | None = None,
) -> None:
    """Log maps, paths, the PGO graph, and markers to the active rerun recording."""
    from dimos.memory2.vis.color import Color
    from dimos.msgs.geometry_msgs.Transform import Transform

    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="world")))
    if global_map is not None:
        rr.log(
            "world/raw_map/pointcloud",
            global_map.to_rerun(voxel_size=voxel / 2, bottom_cutoff=bottom_cutoff),
            static=True,
        )
    if path:
        rr.log(
            "world/raw_map/path",
            rr.LineStrips3D(strips=[path], colors=[[231, 76, 60]], radii=[PATH_THICKNESS]),
            static=True,
        )
    if pgo_map is not None:
        rr.log(
            "world/pgo_map/pointcloud",
            pgo_map.to_rerun(voxel_size=voxel / 2, bottom_cutoff=bottom_cutoff),
            static=True,
        )
    if full_pgo_map is not None:
        rr.log(
            "world/full_pgo_map/pointcloud",
            full_pgo_map.to_rerun(voxel_size=voxel / 2, bottom_cutoff=bottom_cutoff),
            static=True,
        )
    if pgo_path:
        rr.log(
            "world/pgo_map/path",
            rr.LineStrips3D(strips=[pgo_path], colors=[[255, 255, 255]], radii=[PATH_THICKNESS]),
            static=True,
        )
        rr.log(
            "world/pgo_map/pgo/keyframes",
            rr.Points3D(positions=pgo_path, colors=[[255, 0, 0]], radii=[0.025]),
            static=True,
        )
    if graph is not None and graph.loops:
        loop_strips = [
            [
                (lc.source.translation.x, lc.source.translation.y, lc.source.translation.z),
                (lc.target.translation.x, lc.target.translation.y, lc.target.translation.z),
            ]
            for lc in graph.loops
        ]
        rr.log(
            "world/pgo_map/pgo/loop_closures",
            rr.LineStrips3D(strips=loop_strips, colors=[[231, 76, 60]], radii=[0.025]),
            static=True,
        )
    if marker_dets:
        half = marker_size / 2.0
        n = len(marker_dets)
        fill_half = [(half, half, 0.005)] * n
        # Outline sits just outside the fill so both stay visible.
        outline_bump = marker_size * 0.05
        outline_half = [(half + outline_bump, half + outline_bump, 0.006)] * n
        raw_centers = [(d.data.center.x, d.data.center.y, d.data.center.z) for d in marker_dets]
        raw_quats = [
            (d.data.orientation.x, d.data.orientation.y, d.data.orientation.z, d.data.orientation.w)
            for d in marker_dets
        ]
        # One entry per tracked marker session — color stable per track_id.
        colors = [
            Color.from_cmap("tab10", (d.data.track_id % 10) / 10.0).rgb_u8() for d in marker_dets
        ]
        labels = [f"track={d.data.track_id} id={d.data.marker_id}" for d in marker_dets]

        _log_markers(
            "world/raw_map/markers",
            raw_centers,
            raw_quats,
            fill_half=fill_half,
            outline_half=outline_half,
            colors=colors,
            labels=labels,
        )

        if graph is not None:
            # PGO-correct each raw marker pose: lift it from world_raw into
            # world_corrected so it lines up with pgo_map.
            pgo_centers: list[tuple[float, float, float]] = []
            pgo_quats: list[tuple[float, float, float, float]] = []
            for d in marker_dets:
                raw_tf = Transform(
                    translation=d.data.center,
                    rotation=d.data.orientation,
                    frame_id="world",
                    child_frame_id=f"marker_{d.data.marker_id}",
                    ts=d.ts,
                )
                corrected = graph.correct(raw_tf)
                pgo_centers.append(
                    (corrected.translation.x, corrected.translation.y, corrected.translation.z)
                )
                pgo_quats.append(
                    (
                        corrected.rotation.x,
                        corrected.rotation.y,
                        corrected.rotation.z,
                        corrected.rotation.w,
                    )
                )
            _log_markers(
                "world/pgo_map/markers",
                pgo_centers,
                pgo_quats,
                fill_half=fill_half,
                outline_half=outline_half,
                colors=colors,
                labels=labels,
            )


def main(
    dataset: str = typer.Argument(..., help="Dataset .db: bare name (cwd or data/) or path"),
    lidar_stream: str = typer.Option(
        "lidar", "--lidar", help="Lidar point-cloud stream to reconstruct"
    ),
    seek: float = typer.Option(0.0, "--seek", help="Skip the first N seconds of the recording"),
    duration: float | None = typer.Option(
        None, "--duration", help="Use only N seconds from --seek (default: to the end)"
    ),
    voxel: float = typer.Option(0.05, "--voxel", help="Voxel size for the rebuild"),
    device: str = typer.Option(
        "CUDA:0", "--device", help="Open3D compute device (e.g. CUDA:0, CPU:0)"
    ),
    pgo: bool = typer.Option(
        False,
        "--pgo",
        help="Run pose graph optimization and rebuild from spatially-deduped frames",
    ),
    pgo_tol: float = typer.Option(
        0.3,
        "--pgo-tol",
        help="Spatial dedup tolerance (meters); applies to both raw and --pgo maps. 0 disables dedup (keep every posed frame)",
    ),
    block_count: int = typer.Option(
        2_000_000, "--block-count", help="VoxelBlockGrid capacity (raw and PGO rebuilds)"
    ),
    export: bool = typer.Option(
        False,
        "--export",
        help="Export PGO map to ./<dataset>.pc2.lcm in cwd (implies --pgo)",
    ),
    full_pgo: bool = typer.Option(
        False,
        "--full-pgo",
        help="Also build a full-replay PGO map (every frame) for comparison (implies --pgo)",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output .rrd path (default: ./<dataset>.rrd)"
    ),
    no_gui: bool = typer.Option(False, "--no-gui", help="Write the .rrd but don't launch rerun"),
    frame: str | None = typer.Option(
        None,
        "--frame",
        help="World frame to register clouds into. In --registration=tf mode, "
        "the default is auto-detected from tf. In observation-pose mode this "
        "is required because stored poses do not encode their parent frame.",
    ),
    registration: RegistrationMode = typer.Option(
        "tf",
        "--registration",
        help="Cloud registration source: 'tf' preserves the existing nearest-tf "
        "workflow; 'observation-pose' uses only the exact pose stored on each "
        "cloud observation and requires --frame.",
    ),
    tf_tolerance: float | None = typer.Option(
        None,
        "--tf-tolerance",
        help="Max |Δts| (s) for tf lookups; default unlimited (nearest message), "
        "which also serves static/rarely-published transforms",
    ),
    carve: bool = typer.Option(
        False,
        "--carve/--no-carve",
        help="Column carving: keep only the latest frame's points per (X,Y) column. "
        "Off by default (full 3D accumulation); on collapses vertical structure "
        "(stairs, revisited columns) to the most recent observation.",
    ),
    raytrace: bool = typer.Option(
        False,
        "--raytrace",
        help="With --registration=observation-pose, replay every frame through "
        "raycast clearing after PGO. Implies --pgo.",
    ),
    markers: bool = typer.Option(
        False,
        "--markers",
        help="Detect AprilTag markers in color_image and overlay them in rerun",
    ),
    camera_info: Path | None = typer.Option(
        None,
        "--camera-info",
        help="YAML calibration file for --markers; defaults to Go2 builtin",
    ),
    image_pose: str | None = typer.Option(
        None,
        "--image-pose",
        help="Re-pose color_image from this stream's pose (composed with the camera "
        "optical mount) before marker detection, instead of the image's stored pose",
    ),
    marker_size: float = typer.Option(
        0.1, "--marker-size", help="Physical marker edge length in meters (--markers only)"
    ),
    marker_max_speed: float = typer.Option(
        0.5,
        "--marker-max-speed",
        help="Skip frames where robot is moving faster than this (m/s); 0 disables",
    ),
    marker_max_rot_rate: float = typer.Option(
        50.0,
        "--marker-max-rot-rate",
        help="Skip frames where robot is rotating faster than this (deg/s); 0 disables",
    ),
    marker_quality_window: float = typer.Option(
        0.1,
        "--marker-quality-window",
        help="Sharpest-frame window for marker detection (s)",
    ),
    marker_smoothing: float = typer.Option(
        7.5,
        "--marker-smoothing",
        help="Sliding-window track buffer for marker pose averaging (s); 0 disables (one box per raw detection)",
    ),
    bottom_cutoff: float | None = typer.Option(
        None,
        "--bottom-cutoff",
        help="Drop global-map points below this Z (m) when rendering; e.g. 0 strips the floor",
    ),
    denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Statistical outlier removal on the finished maps (o3d, nb_neighbors=20, "
        "std_ratio=2.0): drops sparse floaters before rendering/export",
    ),
) -> None:
    """Rebuild a voxel map from a recorded SQLite dataset, write a .rrd, and open it in rerun."""
    from dimos.mapping.loop_closure.pgo import PGO
    from dimos.memory2.cli.dataset import open_store, resolve_dataset
    from dimos.memory2.transform import QualityWindow, SpeedLimit
    from dimos.memory2.utils.progress import progress
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.perception.fiducial.marker_transformer import DetectMarkers
    from dimos.robot.unitree.go2.connection import BASE_TO_OPTICAL, _camera_info_static
    from dimos.visualization.rerun.init import rerun_init

    db_path = resolve_dataset(dataset)
    store = open_store(db_path)
    if out is None:
        out = Path.cwd() / f"{db_path.stem}.rrd"
    if raytrace and registration != "observation-pose":
        raise typer.BadParameter(
            "--raytrace requires --registration=observation-pose so endpoints "
            "and sensor origins use the same exact snapshot pose",
            param_hint="--raytrace",
        )
    if export or full_pgo or raytrace:
        pgo = True

    lidar = store.stream(lidar_stream, PointCloud2).from_time(seek or None).to_time(duration)

    print(lidar.summary())

    total = lidar.count()

    from dimos.memory2.tf import StreamTF

    first_obs = next(iter(lidar), None)
    tf_buf = StreamTF.from_store(store) if registration == "tf" else None
    try:
        world, cloud_frame, register = _resolve_registration(
            first_obs,
            mode=registration,
            requested_world=frame,
            tf_buf=tf_buf,
            tf_tolerance=tf_tolerance,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--frame") from exc

    sensor_frame = cloud_frame or "sensor"
    if cloud_frame is not None:
        if registration == "observation-pose":
            if register is None:
                print(
                    f"clouds already in world frame {world!r}; using exact observation "
                    "poses as ray origins (no tf fallback)"
                )
            else:
                print(
                    f"registering clouds {world!r} ← {sensor_frame!r} via exact "
                    "observation pose (no tf fallback)"
                )
        elif register is not None:
            print(f"registering clouds {world!r} ← {cloud_frame!r} via tf")
        else:
            print(f"clouds already in world frame {world!r}; accumulating verbatim")
            print("warning: trajectory positions come from stored obs.pose (old dataset)")

    # Spatial dedup: bucket frames by 3D cell using the trajectory position,
    # keep the latest per cell. Shared by raw and PGO rebuilds. Doesn't touch
    # obs.data so it stays cheap (no pointcloud loading). With pgo_tol<=0 the
    # bucketing is disabled and every positioned frame is kept (keyed by index).
    seen: dict[Any, tuple[Observation[Any], tuple[float, float, float]]] = {}
    unregistered = 0
    for i, obs in enumerate(lidar):
        pos = _trajectory_position(
            obs,
            mode=registration,
            world_frame=world,
            child_frame=sensor_frame,
            register=register,
        )
        if pos is None:
            unregistered += 1
            continue
        if pgo_tol > 0:
            # math.floor so negative coords bucket consistently; int() truncates
            # toward zero and silently folds -0.5 and 0.5 into the same cell.
            key: Any = (
                math.floor(pos[0] / pgo_tol),
                math.floor(pos[1] / pgo_tol),
                math.floor(pos[2] / pgo_tol),
            )
        else:
            key = i
        seen[key] = (obs, pos)

    n_kept = len(seen)
    pct = 100 * n_kept / total if total else 0
    if pgo_tol > 0:
        print(f"dedup: kept [{n_kept}/{total}] frames ({pct:.1f}%) at tol={pgo_tol}m")
    else:
        print(f"dedup: disabled, kept all [{n_kept}/{total}] positioned frames")
    if unregistered and registration == "observation-pose":
        print(
            f"registration: dropped {unregistered}/{total} frames without valid observation poses"
        )
    if registration == "observation-pose" and total > 0 and n_kept == 0:
        raise typer.BadParameter(
            f"none of the {total} lidar observations has a valid stored pose; "
            "observation-pose registration never falls back to tf",
            param_hint="--registration",
        )

    # Dict insertion order = lidar iteration order = chronological.
    kept = [obs for obs, _ in seen.values()]
    path: list[tuple[float, float, float]] = [pos for _, pos in seen.values()]

    pgo_map = None
    pgo_path: list[tuple[float, float, float]] = []
    graph: PoseGraph | None = None
    if pgo:
        print("running PGO twopass map...")
        with progress(total, "pgo pass 1 (optimizing)") as bar:
            pgo_input = lidar.tap(bar)
            # PGO consumes world-frame endpoints and unregisters them with the
            # stored pose internally. Keep legacy tf mode unchanged; exact
            # observation-pose mode prepares both inputs from the same pose.
            if registration == "observation-pose":
                pgo_input = pgo_input.transform(
                    lambda upstream: _observation_pose_registered(
                        upstream,
                        world_frame=world,
                        child_frame=sensor_frame,
                    )
                )
            try:
                graph = pgo_input.transform(PGO()).last().data
            except LookupError as exc:
                if registration == "tf":
                    raise
                raise typer.BadParameter(
                    "PGO found no usable posed lidar frames",
                    param_hint="--pgo",
                ) from exc

        pgo_path = [
            (kf.optimized.translation.x, kf.optimized.translation.y, kf.optimized.translation.z)
            for kf in graph.keyframes
        ]

        if raytrace:
            with progress(total, "pgo pass 2 (raytracing)") as bar:
                # Ray clearing is temporal, so replay every frame in order;
                # spatially deduped frames are only appropriate for occupancy
                # accumulation, not free-space evidence.
                pgo_map = _raytrace_accumulate(
                    lidar,
                    voxel=voxel,
                    world_frame=world,
                    graph=graph,
                    register=register,
                    progress_cb=bar,
                )
        else:
            with progress(n_kept, "pgo pass 2 (rebuilding)") as bar:
                pgo_map = _accumulate(
                    kept,
                    voxel=voxel,
                    block_count=block_count,
                    device=device,
                    graph=graph,
                    register=register,
                    carve_columns=carve,
                    progress_cb=bar,
                )

    full_pgo_map = None
    if full_pgo:
        assert graph is not None
        if raytrace:
            print("full pgo: omitted because --raytrace already replays every frame")
        else:
            with progress(total, "full pgo (rebuilding)") as bar:
                full_pgo_map = _accumulate(
                    lidar,
                    voxel=voxel,
                    block_count=block_count,
                    device=device,
                    graph=graph,
                    register=register,
                    carve_columns=carve,
                    progress_cb=bar,
                )

    # Raw map: same dedup'd frames, no PGO correction.
    with progress(n_kept, "reconstructing global map") as bar:
        global_map = _accumulate(
            kept,
            voxel=voxel,
            block_count=block_count,
            device=device,
            register=register,
            carve_columns=carve,
            progress_cb=bar,
        )

    if denoise:
        print("denoising maps (statistical outlier removal)...")
        global_map = _denoise(global_map)
        pgo_map = _denoise(pgo_map)
        full_pgo_map = _denoise(full_pgo_map)

    marker_dets: list[Observation[Any]] = []
    if markers:
        # Image observations in dimos recordings are stamped with
        # frame_id="camera_optical", so obs.pose is already optical-in-world
        # (verified: matches lidar_base_pose + BASE_TO_OPTICAL to ~1mm). With
        # --image-pose, swap that stored pose for a different source (e.g.
        # fastlio_odometry), composing the base→optical mount onto it first.
        color_image = store.stream("color_image", Image).from_time(seek or None).to_time(duration)
        n_images = color_image.count()
        if image_pose is not None:
            from dimos.mapping.utils.cli.pose_fill import pose_fill

            src_pose: Stream[Any] = (
                store.stream(image_pose).from_time(seek or None).to_time(duration)
            )
            print(f"re-posing color_image from {image_pose!r} + camera optical mount")
            color_image = pose_fill(color_image, src_pose, tolerance=0.1, mount=BASE_TO_OPTICAL)
        cam_info = CameraInfo.from_yaml(str(camera_info)) if camera_info else _camera_info_static()
        xf = DetectMarkers(
            camera_info=cam_info,
            marker_length_m=marker_size,
            smoothing_window=marker_smoothing,
        )
        # Keep the sharpest frame per --marker-quality-window window, then
        # drop frames where the robot was moving (linear + rotational) faster
        # than the limits. Defaults match replay_marker.py so positions agree.
        with progress(n_images, "detecting markers") as bar:
            pipeline: Stream[Image] = color_image.tap(bar).transform(
                QualityWindow(lambda img: img.sharpness, window=marker_quality_window)
            )
            if marker_max_speed > 0:
                pipeline = pipeline.transform(
                    SpeedLimit(
                        max_mps=marker_max_speed,
                        max_dps=marker_max_rot_rate if marker_max_rot_rate > 0 else None,
                    )
                )
            all_dets = pipeline.transform(xf).to_list()
        if marker_smoothing > 0:
            # Keep only the latest emission per track_id — that's the most
            # averaged pose, drawn once per tracked marker session.
            by_track: dict[int, Observation[Any]] = {}
            for d in all_dets:
                by_track[d.data.track_id] = d
            marker_dets = list(by_track.values())
        else:
            marker_dets = all_dets
        unique_ids = sorted({obs.data.marker_id for obs in marker_dets})
        print(
            f"markers: {len(marker_dets)} entries from {len(all_dets)} raw detections "
            f"across {len(unique_ids)} unique ids {unique_ids}"
        )

    rerun_init("dimos map tool")
    rr.save(str(out))
    _log_reconstruction(
        voxel=voxel,
        global_map=global_map,
        path=path,
        pgo_map=pgo_map,
        full_pgo_map=full_pgo_map,
        pgo_path=pgo_path,
        graph=graph,
        marker_dets=marker_dets,
        marker_size=marker_size,
        bottom_cutoff=bottom_cutoff,
    )
    rr.rerun_shutdown()
    print(f"wrote {out}")

    if export and pgo_map is not None:
        out_path = Path.cwd() / f"{db_path.stem}.pc2.lcm"
        print(f"exporting PGO twopass map to {out_path}...")
        out_path.write_bytes(pgo_map.lcm_encode())
        print(f"wrote {out_path}")
        print()
        print("load back with:")
        print("    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2")
        print(f'    pcd = PointCloud2.lcm_decode(open("{out_path.name}", "rb").read())')

    if no_gui:
        print(f"open with: rerun {out}")
    elif rerun := shutil.which("rerun"):
        subprocess.Popen([rerun, str(out)])
    else:
        print(f"rerun viewer not found on PATH; open manually:\n    rerun {out}")


if __name__ == "__main__":
    typer.run(main)
