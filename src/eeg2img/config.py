"""YAML experiment configs with inheritance and command-line overrides.

A config file may name one or more parents under ``_base_`` (paths relative to
the file). Parents are merged first, the file itself on top, and ``key.sub=value``
overrides last. No hyperparameter lives in Python: scripts read everything from
the resolved config, which is stored verbatim with each run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "_base_"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_file(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"circular _base_ chain: {chain}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")

    bases = raw.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    resolved: dict[str, Any] = {}
    for base in bases:
        resolved = deep_merge(resolved, _load_file(path.parent / base, (*seen, path)))
    return deep_merge(resolved, raw)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML."""
    result = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        dotted, raw_value = item.split("=", 1)
        node = result
        *parents, leaf = dotted.strip().split(".")
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[leaf] = yaml.safe_load(raw_value)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config, resolve its ``_base_`` chain, then apply overrides."""
    return apply_overrides(_load_file(Path(path)), overrides)


def config_hash(config: Mapping[str, Any]) -> str:
    """Short stable hash of a resolved config, independent of key order."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def require(config: Mapping[str, Any], dotted: str) -> Any:
    """Fetch a required ``dotted.key``, failing with the full key name."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"config is missing required key {dotted!r}")
        node = node[part]
    return node
