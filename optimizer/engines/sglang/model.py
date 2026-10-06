"""How SGLang identifies a checkpoint's architecture: the transformer's
`_class_name` in its diffusers config, as SGLang itself reads it before
resolving the class (runtime/loader/component_loaders/transformer_loader.py:295).

Reads JSON only; SGLang is not imported. Fetching a hub checkpoint into a
local snapshot is the execution stage's.
"""

from __future__ import annotations

import json
import pathlib


def architecture(snapshot: pathlib.Path) -> str:
    config = pathlib.Path(snapshot) / "transformer" / "config.json"
    if not config.is_file():
        raise ValueError(f"{snapshot} is not a diffusers checkpoint with a transformer: no {config}")
    name = json.loads(config.read_text()).get("_class_name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{config} names no _class_name")
    return name
