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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any
from unittest.mock import MagicMock

from dimos_lcm.std_msgs import Bool
import pytest
from pytest_mock import MockerFixture

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.stream import Out, Stream, Transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner


class _DirectTransport(Transport[Any]):
    """Synchronous transport for exercising the module's subscribed handlers."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Any], Any]] = []

    def broadcast(self, _selfstream: Out[Any] | None, value: Any) -> None:
        for callback in self._subscribers:
            callback(value)

    def subscribe(
        self,
        callback: Callable[[Any], Any],
        _selfstream: Stream[Any] | None = None,
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            self._subscribers.remove(callback)

        return _unsubscribe

    def start(self) -> None: ...

    def stop(self) -> None:
        self._subscribers.clear()


@dataclass(frozen=True)
class _PlannerHarness:
    module: ReplanningAStarPlanner
    handle_goal: MagicMock
    cancel_goal: MagicMock

    def publish_navigation_enabled(self, enabled: bool) -> None:
        self.module.navigation_enabled.transport.publish(Bool(data=enabled))


@contextmanager
def _running_planner(
    mocker: MockerFixture, *, require_navigation_enabled: bool
) -> Iterator[_PlannerHarness]:
    mocker.patch.object(GlobalPlanner, "start", autospec=True)
    mocker.patch.object(GlobalPlanner, "stop", autospec=True)
    handle_goal = mocker.patch.object(GlobalPlanner, "handle_goal_request", autospec=True)
    cancel_goal = mocker.patch.object(GlobalPlanner, "cancel_goal", autospec=True)

    module = ReplanningAStarPlanner(require_navigation_enabled=require_navigation_enabled)
    for input_stream in module.inputs.values():
        input_stream.transport = _DirectTransport()

    with module:
        yield _PlannerHarness(module, handle_goal, cancel_goal)


@pytest.fixture
def gated_planner(mocker: MockerFixture) -> Iterator[_PlannerHarness]:
    with _running_planner(mocker, require_navigation_enabled=True) as harness:
        yield harness


@pytest.fixture
def default_planner(mocker: MockerFixture) -> Iterator[_PlannerHarness]:
    with _running_planner(mocker, require_navigation_enabled=False) as harness:
        yield harness


def test_required_navigation_enable_is_fail_closed(gated_planner: _PlannerHarness) -> None:
    goal = PoseStamped()

    assert gated_planner.module.set_goal(goal) is False
    gated_planner.handle_goal.assert_not_called()

    gated_planner.publish_navigation_enabled(True)
    assert gated_planner.module.set_goal(goal) is True
    assert gated_planner.handle_goal.call_count == 1

    gated_planner.publish_navigation_enabled(False)
    assert gated_planner.cancel_goal.call_count == 1
    assert gated_planner.module.set_goal(goal) is False
    assert gated_planner.handle_goal.call_count == 1


def test_navigation_enable_input_is_ignored_by_default(default_planner: _PlannerHarness) -> None:
    goal = PoseStamped()

    default_planner.publish_navigation_enabled(False)

    assert default_planner.module.set_goal(goal) is True
    assert default_planner.handle_goal.call_count == 1
    default_planner.cancel_goal.assert_not_called()


def test_disable_cancels_an_inflight_goal_before_reenabling(
    gated_planner: _PlannerHarness,
) -> None:
    goal_started = Event()
    release_goal = Event()
    disable_finished = Event()
    results: list[bool] = []
    inflight_goal = PoseStamped()
    next_goal = PoseStamped()

    def _blocking_goal(_planner: GlobalPlanner, goal: PoseStamped) -> None:
        if goal is inflight_goal:
            goal_started.set()
            assert release_goal.wait(DEFAULT_THREAD_JOIN_TIMEOUT)

    def _submit_goal() -> None:
        results.append(gated_planner.module.set_goal(inflight_goal))

    def _disable_navigation() -> None:
        gated_planner.publish_navigation_enabled(False)
        disable_finished.set()

    gated_planner.publish_navigation_enabled(True)
    gated_planner.handle_goal.side_effect = _blocking_goal
    goal_thread = Thread(target=_submit_goal)
    disable_thread = Thread(target=_disable_navigation)
    goal_thread.start()

    try:
        assert goal_started.wait(DEFAULT_THREAD_JOIN_TIMEOUT)
        disable_thread.start()
        assert disable_finished.wait(DEFAULT_THREAD_JOIN_TIMEOUT)
        assert gated_planner.cancel_goal.call_count == 1

        gated_planner.publish_navigation_enabled(True)
        assert gated_planner.module.set_goal(next_goal) is False
        assert gated_planner.handle_goal.call_count == 1
    finally:
        release_goal.set()
        goal_thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        if disable_thread.ident is not None:
            disable_thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)

    assert not goal_thread.is_alive()
    assert not disable_thread.is_alive()
    assert results == [False]
    assert gated_planner.handle_goal.call_count == 1
    assert gated_planner.cancel_goal.call_count == 2

    assert gated_planner.module.set_goal(next_goal) is True
    assert gated_planner.handle_goal.call_count == 2
