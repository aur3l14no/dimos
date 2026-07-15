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

from collections.abc import Callable, Iterator
from pathlib import Path
import threading
from typing import Any
from unittest.mock import Mock

import numpy as np
from pydantic import ValidationError
import pytest
from reactivex.subject import Subject
from scipy.spatial.transform import Rotation

from dimos.core.stream import Out, Stream, Transport
import dimos.mapping.relocalization.module as relocalization_module
from dimos.mapping.relocalization.module import RelocalizationModule, RelocalizationState
from dimos.mapping.relocalization.relocalize import project_to_4dof
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def _cloud(points: list[list[float]], frame_id: str) -> PointCloud2:
    return PointCloud2.from_numpy(
        np.asarray(points, dtype=np.float32),
        frame_id=frame_id,
        timestamp=1.0,
    )


class _PointCloudTransport(Transport[PointCloud2]):
    def __init__(self) -> None:
        self._subject: Subject[PointCloud2] = Subject()
        self.stopped = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            self._subject.on_completed()

    def broadcast(self, _: Out[PointCloud2] | None, value: PointCloud2) -> None:
        if not self.stopped:
            self._subject.on_next(value)

    def subscribe(
        self,
        callback: Callable[[PointCloud2], Any],
        selfstream: Stream[PointCloud2] | None = None,
    ) -> Callable[[], None]:
        del selfstream
        subscription = self._subject.subscribe(callback)
        return subscription.dispose


@pytest.fixture
def make_module() -> Iterator[Callable[..., RelocalizationModule]]:
    modules: list[RelocalizationModule] = []

    def make(**config: Any) -> RelocalizationModule:
        module = RelocalizationModule(**config)
        modules.append(module)
        return module

    yield make

    for module in reversed(modules):
        module.stop()


def _wire_inputs(
    module: RelocalizationModule,
) -> tuple[_PointCloudTransport, _PointCloudTransport]:
    local = _PointCloudTransport()
    global_ = _PointCloudTransport()
    module.local_map.transport = local
    module.global_map.transport = global_
    return local, global_


def _write_map(tmp_path: Path) -> Path:
    path = tmp_path / "premap.pc2.lcm"
    path.write_bytes(_cloud([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "recording").lcm_encode())
    return path


def test_project_to_4dof_keeps_translation_and_yaw() -> None:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    T[:3, 3] = [1.0, 2.0, 3.0]

    projected = project_to_4dof(T)
    euler = Rotation.from_matrix(projected[:3, :3]).as_euler("xyz")

    np.testing.assert_allclose(projected[:3, 3], T[:3, 3])
    np.testing.assert_allclose(euler[:2], 0.0, atol=1e-12)
    assert euler[2] == pytest.approx(1.1)


def test_load_premap_rejects_missing_and_corrupt_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not read"):
        RelocalizationModule._load_premap(tmp_path / "missing.pc2.lcm")

    corrupt = tmp_path / "corrupt.pc2.lcm"
    corrupt.write_bytes(b"not a point cloud")
    with pytest.raises(ValueError, match="Could not decode"):
        RelocalizationModule._load_premap(corrupt)


def test_min_local_points_is_configurable(
    make_module: Callable[..., RelocalizationModule],
) -> None:
    module = make_module(min_local_points=2)

    assert not module._has_enough_points(_cloud([[0.0, 0.0, 0.0]], "odom"))
    assert module._has_enough_points(_cloud([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "odom"))


@pytest.mark.parametrize(
    ("config", "value"),
    [
        ("merge_voxel_size", 0),
        ("min_local_points", 0),
    ],
)
def test_positive_map_thresholds_are_required(config: str, value: int) -> None:
    with pytest.raises(ValidationError):
        RelocalizationModule(**{config: value})


def test_relocalization_locks_once_and_uses_live_frame(
    monkeypatch: pytest.MonkeyPatch,
    make_module: Callable[..., RelocalizationModule],
) -> None:
    calls = 0
    T_map_live = np.eye(4)
    T_map_live[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    T_map_live[:3, 3] = [1.0, -2.0, 0.5]

    def fake_relocalize(*_: object, **__: object) -> tuple[np.ndarray, float]:
        nonlocal calls
        calls += 1
        return T_map_live, 0.9

    monkeypatch.setattr(relocalization_module, "_relocalize", fake_relocalize)
    module = make_module(
        map_file="unused",
        lock_after_first=True,
        gravity_aligned_4dof=True,
        use_carving=False,
    )
    module._premap = _cloud([[0.0, 0.0, 0.0]], "map")
    localized: list[bool] = []
    module.localized.subscribe(lambda msg: localized.append(msg.data))

    tf = module._try_relocalize(_cloud([[0.0, 0.0, 0.0]], "odom"))
    assert tf is not None
    module._accept_relocalization(tf)

    assert module._state is RelocalizationState.LOCKED
    assert tf.frame_id == "odom"
    assert tf.child_frame_id == "map"
    np.testing.assert_allclose(tf.to_matrix(), np.linalg.inv(T_map_live), atol=1e-7)
    assert localized == []

    assert module._try_relocalize(_cloud([[1.0, 0.0, 0.0]], "odom")) is None
    assert calls == 1


def test_live_global_map_replaces_observed_history_columns(
    make_module: Callable[..., RelocalizationModule],
) -> None:
    module = make_module(map_file="unused", merge_voxel_size=0.1)
    historical = _cloud([[0.01, 0.01, 0.0], [1.01, 1.01, 1.0]], "map")
    original_points = historical.points_f32().copy()
    module._premap = historical
    merged: list[PointCloud2] = []
    localized: list[bool] = []
    module.merged_map.subscribe(merged.append)
    module.localized.subscribe(lambda msg: localized.append(msg.data))
    identity = Transform(
        translation=Vector3(),
        rotation=Quaternion(),
        frame_id="odom",
        child_frame_id="map",
    )

    module._accept_relocalization(identity)
    module._on_merge_input((_cloud([[0.01, 0.01, 2.0]], "odom"), identity))

    assert len(merged) == 1
    points = merged[0].points_f32()
    assert merged[0].frame_id == "odom"
    assert len(points) == 2
    assert sorted(points[:, 2].tolist()) == pytest.approx([1.05, 2.05])
    np.testing.assert_array_equal(historical.points_f32(), original_points)
    assert localized == [True]

    updated = Transform(
        translation=Vector3(x=1.0),
        rotation=Quaternion(),
        frame_id="odom",
        child_frame_id="map",
    )
    module._accept_relocalization(updated)
    assert localized == [True, False]

    module._on_merge_input((_cloud([[1.01, 0.01, 2.0]], "odom"), updated))
    assert localized == [True, False, True]


def test_matching_map_defaults_to_global_and_can_use_local(
    make_module: Callable[..., RelocalizationModule],
) -> None:
    default = make_module()
    local = make_module(matching_map="local_map")

    assert default._matching_map() is default.global_map
    assert local._matching_map() is local.local_map


def test_publish_loaded_map_reaches_late_subscribers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_module: Callable[..., RelocalizationModule],
) -> None:
    monkeypatch.setattr(relocalization_module, "PUBLISH_INTERVAL", 60.0)
    module = make_module(
        map_file=str(_write_map(tmp_path)),
        publish_loaded_map=True,
        use_carving=False,
    )
    _wire_inputs(module)
    module.start()

    received: list[PointCloud2] = []
    module.loaded_map.subscribe(received.append)
    module._publish_periodic(1)

    assert len(received) == 1
    assert received[0].frame_id == "map"
    assert len(received[0]) == 2


def test_relock_merge_and_stop_are_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_module: Callable[..., RelocalizationModule],
) -> None:
    monkeypatch.setattr(relocalization_module, "PUBLISH_INTERVAL", 60.0)
    module = make_module(map_file=str(_write_map(tmp_path)), live_frame="odom")
    transports = _wire_inputs(module)
    block_add = threading.Event()
    add_started = threading.Event()
    release_add = threading.Event()

    class BlockingVoxelGrid:
        instances: list["BlockingVoxelGrid"] = []

        def __init__(self, **_: Any) -> None:
            self.disposed = False
            self.disposed_after_producers = False
            self.last_frame = _cloud([], "odom")
            self.instances.append(self)

        def add_frame(self, frame: PointCloud2) -> None:
            if self.disposed:
                raise RuntimeError("add_frame called after dispose")
            if block_add.is_set():
                add_started.set()
                assert release_add.wait(2.0)
            if self.disposed:
                raise RuntimeError("grid disposed during add_frame")
            self.last_frame = frame

        def get_global_pointcloud2(self) -> PointCloud2:
            if self.disposed:
                raise RuntimeError("get_global_pointcloud2 called after dispose")
            return self.last_frame

        def dispose(self) -> None:
            self.disposed_after_producers = all(transport.stopped for transport in transports)
            self.disposed = True

    monkeypatch.setattr(relocalization_module, "VoxelGrid", BlockingVoxelGrid)
    module.start()
    module._tf = Mock()
    tf1 = Transform(frame_id="odom", child_frame_id="map")
    tf2 = Transform(translation=Vector3(x=1.0), frame_id="odom", child_frame_id="map")
    module._accept_relocalization(tf1)

    errors: list[BaseException] = []

    def run(action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    block_add.set()
    merge = threading.Thread(
        target=run,
        args=(lambda: module._on_merge_input((_cloud([[2.0, 0.0, 0.0]], "odom"), tf1)),),
    )
    merge.start()
    assert add_started.wait(1.0)

    relock = threading.Thread(target=run, args=(lambda: module._accept_relocalization(tf2),))
    periodic = threading.Thread(target=run, args=(lambda: module._publish_periodic(1),))
    stopping = threading.Thread(target=run, args=(module.stop,))
    relock.start()
    periodic.start()
    stopping.start()
    release_add.set()

    for thread in (merge, relock, periodic, stopping):
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    assert errors == []
    assert BlockingVoxelGrid.instances
    assert all(grid.disposed for grid in BlockingVoxelGrid.instances)
    assert any(grid.disposed_after_producers for grid in BlockingVoxelGrid.instances)
    assert module._merge_grid is None

    instance_count = len(BlockingVoxelGrid.instances)
    module._accept_relocalization(tf2)
    assert len(BlockingVoxelGrid.instances) == instance_count
