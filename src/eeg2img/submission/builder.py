"""Build a Codabench submission: one ``submission.py`` plus weights, zipped flat.

The platform accepts a single Python file and installs nothing, so the
inference code cannot import this package. :func:`render_submission_source`
therefore inlines the listed modules into one file, dropping their
intra-package imports. The modules are written to make that safe: they import
only the standard library and torch, and their top-level names do not collide
(checked here at build time).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eeg2img.submission.runtime import PACKAGE_CONFIG, PACKAGE_WEIGHTS

#: Modules inlined into ``submission.py``, in dependency order.
INLINE_MODULES: tuple[str, ...] = (
    "eeg2img.channels",
    "eeg2img.models.floor",
    "eeg2img.models",
    "eeg2img.submission.runtime",
)

PACKAGE_ROOT = "eeg2img"

_HEADER = '''"""Codabench submission for Track 01 (EEG-to-Image). GENERATED FILE.

Built by ``scripts/make_submission.py`` from the ``eeg2img`` package; edit the
package and rebuild rather than editing this file.

Source commit: {commit}
"""

from __future__ import annotations
'''

_SOLVER = '''

# ---------------------------------------------------------------------------
# Platform entry point
# ---------------------------------------------------------------------------
from benchmark_utils.base_solver import CompetSolver  # noqa: E402


class Solver(CompetSolver):
    """Loads the packaged model; the platform then calls ``model.predict(X)``."""

    name = "{solver_name}"

    def load_model(self, meta):
        return load_packaged_model(meta)
'''


def _module_path(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module)
    return Path(spec.origin)


def _is_dropped_import(node: ast.stmt) -> bool:
    """True for imports that must not survive inlining."""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return (
            node.level > 0
            or module == "__future__"
            or module == PACKAGE_ROOT
            or module.startswith(PACKAGE_ROOT + ".")
        )
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == PACKAGE_ROOT for alias in node.names)
    return False


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _inline(module: str) -> tuple[str, set[str]]:
    """Source of ``module`` without intra-package imports, and the names it defines."""
    source = _module_path(module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if _is_dropped_import(node):
            for lineno in range(node.lineno - 1, node.end_lineno):
                lines[lineno] = ""
    body = "\n".join(lines).strip("\n")
    banner = f"# {'-' * 75}\n# inlined from {module}\n# {'-' * 75}"
    return f"{banner}\n{body}\n", _top_level_names(tree)


def render_submission_source(
    solver_name: str = "eeg2img",
    commit: str = "unknown",
    modules: Sequence[str] = INLINE_MODULES,
) -> str:
    """Return the full text of a self-contained ``submission.py``."""
    parts = [_HEADER.format(commit=commit)]
    owner: dict[str, str] = {}
    for module in modules:
        text, names = _inline(module)
        for name in sorted(names):
            if name in owner:
                raise ValueError(
                    f"top-level name {name!r} is defined in both {owner[name]} and {module}; "
                    "inlined modules must not collide"
                )
            owner[name] = module
        parts.append(text)
    parts.append(_SOLVER.format(solver_name=solver_name))
    source = "\n\n".join(parts)

    for node in ast.walk(ast.parse(source)):
        if _is_dropped_import(node) and not (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ):
            raise ValueError("generated submission still imports the eeg2img package")
    return source


def write_package(
    model: nn.Module,
    out_dir: str | Path,
    *,
    model_config: Mapping[str, Any],
    ch_names: Sequence[str],
    n_times: int,
    n_outputs: int,
    sfreq: float,
    normalize_output: bool = True,
    solver_name: str = "eeg2img",
    commit: str = "unknown",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``submission.py``, weights and config into ``out_dir`` and zip them.

    Returns the path of ``submission.zip``, whose files sit at the archive root
    as the platform requires.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    package = {
        "model": {"name": model_config["name"], "params": dict(model_config.get("params") or {})},
        "ch_names": list(ch_names),
        "n_times": int(n_times),
        "n_outputs": int(n_outputs),
        "sfreq": float(sfreq),
        "normalize_output": bool(normalize_output),
        "source_commit": commit,
        **dict(extra or {}),
    }
    files = {
        "submission.py": out_dir / "submission.py",
        PACKAGE_WEIGHTS: out_dir / PACKAGE_WEIGHTS,
        PACKAGE_CONFIG: out_dir / PACKAGE_CONFIG,
    }
    files["submission.py"].write_text(
        render_submission_source(solver_name=solver_name, commit=commit), encoding="utf-8"
    )
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, files[PACKAGE_WEIGHTS])
    files[PACKAGE_CONFIG].write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in files.items():
            archive.write(path, arcname=arcname)
    return zip_path
