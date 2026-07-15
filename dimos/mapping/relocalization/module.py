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

from enum import Enum
from pathlib import Path
import threading
import time
from typing import Any, Literal

from dimos_lcm.std_msgs import Bool  # type: ignore[import-untyped]
import numpy as np
from pydantic import Field
import reactivex as rx
from reactivex import combine_latest, operators as ops
from reactivex.subject import ReplaySubject

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.relocalization.relocalize import relocalize as _relocalize
from dimos.mapping.voxels import VoxelGrid
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.data import resolve_named_path
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()

FRAME_MAP = "map"

PUBLISH_INTERVAL = 2.0  # for loaded map, readiness, and TF
RELOC_INTERVAL = 2.0
MAP_SUFFIX = ".pc2.lcm"


class Config(ModuleConfig):
    map_file: str | None = (
        None  # e.g. `-o relocalizationmodule.map_file=go2_hongkong_office_twopass_map`
    )
    publish_loaded_map: bool = False
    fitness_threshold: float = 0.45
    use_carving: bool = True
    live_frame: str | None = None
    merge_voxel_size: float = Field(default=0.05, gt=0.0)
    require_map_file: bool = False
    lock_after_first: bool = False
    gravity_aligned_4dof: bool = False
    min_local_points: int = Field(default=50_000, ge=1)
    matching_map: Literal["global_map", "local_map"] = "global_map"


class RelocalizationState(str, Enum):
    SEARCHING = "searching"
    LOCKED = "locked"


class RelocalizationModule(Module):
    config: Config
    local_map: In[PointCloud2]
    global_map: In[PointCloud2]
    loaded_map: Out[PointCloud2]
    merged_map: Out[PointCloud2]
    localized: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._premap: PointCloud2 | None = None
        self._last_skip_log = 0.0
        self._state = RelocalizationState.SEARCHING
        self._map_to_live: ReplaySubject[Transform] = ReplaySubject(buffer_size=1)
        self._locked_tf: Transform | None = None
        self._localized = False
        self._merge_grid: VoxelGrid | None = None
        self._state_lock = threading.RLock()
        self._stopping = False

    @rpc
    def start(self) -> None:
        if not self.config.map_file:
            if self.config.require_map_file:
                raise ValueError("Relocalization requires map_file")
            super().start()
            self.localized.publish(Bool(False))
            logger.info("Relocalization module disabled (no map_file configured)")
            return

        path = resolve_named_path(self.config.map_file, MAP_SUFFIX)
        self._premap = self._load_premap(path)

        super().start()
        self.localized.publish(Bool(False))

        self.register_disposable(
            backpressure(
                self._matching_map()
                .observable()
                .pipe(  # type: ignore[no-untyped-call]
                    ops.throttle_first(RELOC_INTERVAL),
                    ops.filter(lambda _: self._can_relocalize()),
                    ops.do_action(self._maybe_log_skip),
                    ops.filter(self._has_enough_points),
                )
            )
            .pipe(ops.map(self._try_relocalize))
            .subscribe(self._accept_relocalization)
        )

        self.register_disposable(
            backpressure(
                combine_latest(
                    self.global_map.observable(),  # type: ignore[no-untyped-call]
                    self._map_to_live,
                )
            ).subscribe(self._on_merge_input)
        )

        self.register_disposable(
            rx.interval(PUBLISH_INTERVAL).pipe(ops.start_with(0)).subscribe(self._publish_periodic)
        )

        logger.info(
            f"Relocalization module started: map_file={self.config.map_file!r}  "
            f"loaded_map.frame_id={self._premap.frame_id!r}"
        )

    def _matching_map(self) -> In[PointCloud2]:
        if self.config.matching_map == "local_map":
            return self.local_map
        return self.global_map

    @staticmethod
    def _load_premap(path: Path) -> PointCloud2:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Could not read relocalization map {path}") from exc

        try:
            premap = PointCloud2.lcm_decode(payload)
        except Exception as exc:
            raise ValueError(f"Could not decode relocalization map {path}") from exc

        points, _ = premap.as_numpy()
        if len(points) == 0:
            raise ValueError(f"Relocalization map is empty: {path}")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Relocalization map has invalid points: {path}")

        premap.frame_id = FRAME_MAP
        return premap

    def _maybe_log_skip(self, msg: PointCloud2) -> None:
        if self._has_enough_points(msg):
            return
        now = time.monotonic()
        with self._state_lock:
            if self._stopping or now - self._last_skip_log <= 5.0:
                return
            logger.warning(
                f"relocalize skipped: n_pts={len(msg)} < "
                f"min_local_points={self.config.min_local_points}"
            )
            self._last_skip_log = now

    def _has_enough_points(self, msg: PointCloud2) -> bool:
        return len(msg) >= self.config.min_local_points

    def _can_relocalize(self) -> bool:
        with self._state_lock:
            return self._can_relocalize_locked()

    def _can_relocalize_locked(self) -> bool:
        return not self._stopping and not (
            self.config.lock_after_first and self._state is RelocalizationState.LOCKED
        )

    def _accept_relocalization(self, tf: Transform | None) -> None:
        if tf is None:
            return
        with self._state_lock:
            if not self._can_relocalize_locked():
                return
            first_match = self._state is RelocalizationState.SEARCHING
            old_grid = self._reset_merge_grid_locked(tf)
            self._locked_tf = tf
            self._state = RelocalizationState.LOCKED
            was_localized = self._localized
            self._localized = False
            if was_localized:
                self.localized.publish(Bool(False))
            self._map_to_live.on_next(tf)
            if old_grid is not None:
                old_grid.dispose()
        action = "locked" if first_match else "updated"
        logger.info(
            f"Relocalization {action}: live_frame={tf.frame_id!r} map_frame={tf.child_frame_id!r}"
        )

    def _live_frame(self, msg: PointCloud2) -> str:
        message_frame = msg.frame_id.strip()
        configured_frame = self.config.live_frame
        if configured_frame is not None:
            configured_frame = configured_frame.strip()
            if not configured_frame:
                raise ValueError("live_frame must not be blank")
            if message_frame and message_frame != configured_frame:
                raise ValueError(
                    f"live map frame {message_frame!r} does not match configured "
                    f"live_frame {configured_frame!r}"
                )
            return configured_frame
        if not message_frame:
            raise ValueError("live map has no frame_id and live_frame is not configured")
        return message_frame

    def _try_relocalize(self, msg: PointCloud2) -> Transform | None:
        with self._state_lock:
            if not self._can_relocalize_locked():
                return None
            premap = self._premap
        assert premap is not None
        try:
            live_frame = self._live_frame(msg)
        except ValueError as exc:
            logger.error(f"relocalize rejected: {exc}")
            return None
        t0 = time.monotonic()
        try:
            T, fitness = _relocalize(
                premap.pointcloud,
                msg.pointcloud,
                gravity_aligned_4dof=self.config.gravity_aligned_4dof,
            )
        except Exception:
            logger.exception("relocalize() failed")
            return None
        dt = time.monotonic() - t0
        n_pts = len(msg)

        if fitness < self.config.fitness_threshold:
            logger.warning(
                f"relocalize rejected: fitness={fitness:.3f} < threshold={self.config.fitness_threshold} "
                f"time_cost={dt:.1f}s n_pts={n_pts}"
            )
            return None

        # relocalize(premap, live) returns T_map_live. The persistent map is
        # transformed into the current session's live frame for planning, so
        # publish its inverse T_live_map.
        T_inv = np.linalg.inv(T)
        new_tf = Transform(
            translation=Vector3(*T_inv[:3, 3]),
            rotation=Quaternion.from_rotation_matrix(T_inv[:3, :3]),
            frame_id=live_frame,
            child_frame_id=FRAME_MAP,
        )
        logger.info(
            f"relocalize: fitness={fitness:.3f} time_cost={dt:.1f}s n_pts={n_pts} "
            f"reloc_t={T[:3, 3].round(3).tolist()} "
            f"TF {live_frame!r} -> {FRAME_MAP!r} "
            f"published_t={T_inv[:3, 3].round(3).tolist()} "
        )
        return new_tf

    def _publish_periodic(self, _: int) -> None:
        with self._state_lock:
            if self._stopping or self._premap is None:
                return
            if self.config.publish_loaded_map:
                self.loaded_map.publish(self._premap)
            self.localized.publish(Bool(self._localized))
            if self._locked_tf is not None:
                self.tf.publish(self._locked_tf.now())

    def _reset_merge_grid_locked(self, tf: Transform) -> VoxelGrid | None:
        if not self.config.use_carving or self._premap is None:
            return None
        grid = VoxelGrid(
            voxel_size=self.config.merge_voxel_size,
            carve_columns=True,
            frame_id=tf.frame_id,
            show_startup_log=False,
        )
        grid.add_frame(self._premap.transform(tf))
        old_grid, self._merge_grid = self._merge_grid, grid
        return old_grid

    def _on_merge_input(self, pair: tuple[PointCloud2, Transform]) -> None:
        live_global, tf = pair
        try:
            live_frame = self._live_frame(live_global)
        except ValueError as exc:
            logger.error(f"map merge skipped: {exc}")
            return
        if live_frame != tf.frame_id:
            logger.error(
                f"map merge skipped: live map frame {live_frame!r} does not match locked "
                f"frame {tf.frame_id!r}"
            )
            return
        with self._state_lock:
            if self._stopping or self._premap is None or tf is not self._locked_tf:
                return
            if self.config.use_carving:
                if self._merge_grid is None:
                    self._reset_merge_grid_locked(tf)
                assert self._merge_grid is not None
                # The ray tracer's global map is cumulative. Re-adding it replaces
                # currently observed XY columns while the persistent grid retains
                # imported columns that have not been observed in this session.
                self._merge_grid.add_frame(live_global)
                merged = self._merge_grid.get_global_pointcloud2()
            else:
                merged = live_global + self._premap.transform(tf)

            became_localized = not self._localized
            self._localized = True
            self.merged_map.publish(merged)
            if became_localized:
                # Do not release navigation until its first authoritative map has
                # actually been emitted.
                self.localized.publish(Bool(True))

    @rpc
    def stop(self) -> None:
        with self._state_lock:
            if self._stopping:
                return
            self._stopping = True
        # Dispose the Rx producers before releasing their shared voxel grid. A
        # worker already queued by backpressure will observe _stopping and return
        # without touching the grid.
        super().stop()
        with self._state_lock:
            self._map_to_live.on_completed()
            if self._merge_grid is not None:
                self._merge_grid.dispose()
            self._merge_grid = None
