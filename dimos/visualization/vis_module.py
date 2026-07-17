#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Shared visualization module factory for all robot blueprints."""

from typing import Any, get_args

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.visualization.rerun.constants import ViewerBackend
from dimos.visualization.rerun.viewer_input_server import RerunViewerInputServer
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


def vis_module(
    viewer_backend: ViewerBackend,
    rerun_config: dict[str, Any] | None = None,
    *,
    viewer_controls: bool = True,
    web_dashboard: bool = True,
) -> Blueprint:
    """Create a visualization blueprint based on the selected viewer backend.

    By default, bundles the Rerun viewer module with the viewer-control server
    and legacy web dashboard. Products that do not use those optional services
    can omit them without disabling Rerun visualization.

    Example usage::

        from dimos.core.global_config import global_config
        viz = vis_module(
            global_config.viewer,
            rerun_config={
                "visual_override": {
                    "world/camera_info": lambda ci: ci.to_rerun(...),
                },
                "static": {
                    "world/tf/base_link": lambda rr: [rr.Boxes3D(...)],
                },
            },
        )
    """
    if rerun_config is None:
        rerun_config = {}

    match viewer_backend:
        case "rerun":
            from dimos.core.global_config import global_config
            from dimos.protocol.pubsub.impl.lcmpubsub import LCM
            from dimos.visualization.rerun.bridge import RerunBridgeModule

            rerun_config = {**rerun_config}  # copy (avoid mutation)
            rerun_config.setdefault("pubsubs", [LCM()])
            rerun_config.setdefault("rerun_open", global_config.rerun_open)
            rerun_config.setdefault("rerun_web", global_config.rerun_web)
            modules = [RerunBridgeModule.blueprint(**rerun_config)]
            if viewer_controls:
                modules.append(RerunViewerInputServer.blueprint())
            if web_dashboard:
                modules.append(WebsocketVisModule.blueprint())
            return autoconnect(*modules)
        case "none":
            if web_dashboard:
                return autoconnect(WebsocketVisModule.blueprint())
            return autoconnect()
        case _:
            valid = ", ".join(get_args(ViewerBackend))
            raise ValueError(f"Unknown viewer_backend {viewer_backend!r}. Expected one of: {valid}")
