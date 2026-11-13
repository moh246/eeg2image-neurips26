"""Local replay of the platform's submission contract.

Sealed-phase logs are hidden and every accepted upload spends one of five
submissions, so a ZIP is checked here first. The check loads ``submission.py``
the way the platform does, in a fresh interpreter that cannot import this
package, with only a stand-in for ``benchmark_utils.base_solver``, and then
exercises ``predict`` on the documented input contract.

For a replay against the organisers' actual benchmark code, see
``make contract-replay``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any

from eeg2img import contract

_STUB = '''
class CompetSolver:
    """Stand-in for the platform's base class: only what load_model needs."""
    name = "stub"
    train_loader = None
'''

_PROBE = r"""
import importlib.util, json, sys
from pathlib import Path
import torch

sub_dir, n_outputs, ch_names, n_times, sfreq = (
    Path(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4]),
    float(sys.argv[5]),
)
spec = importlib.util.spec_from_file_location("submission", sub_dir / "submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "eeg2img" not in sys.modules, "submission imported the eeg2img package"

meta = dict(submission_dir=sub_dir, device="cpu", n_chans=len(ch_names), n_times=n_times,
            sfreq=sfreq, ch_names=ch_names, chs_info=[{"ch_name": c} for c in ch_names],
            n_outputs=n_outputs)
solver = module.Solver()
model = solver.load_model(meta)

report = {}
gen = torch.Generator().manual_seed(0)
x = torch.randn(7, len(ch_names), n_times, generator=gen)
out = model.predict(x)
report["shape"] = list(out.shape)
report["dtype"] = str(out.dtype)
report["finite"] = bool(torch.isfinite(out).all())
report["is_tensor"] = isinstance(out, torch.Tensor)
# Batch size is not part of the contract: 1 and 64 must both work.
report["batch_sizes_ok"] = all(
    tuple(model.predict(torch.randn(b, len(ch_names), n_times, generator=gen)).shape)
    == (b, n_outputs) for b in (1, 64)
)
# No state may be carried across predict calls.
report["stateless"] = bool(torch.equal(out, model.predict(x)))
# Channels are matched by name: permuting input channels and names together
# must not change the prediction.
perm = torch.randperm(len(ch_names), generator=gen)
meta_perm = dict(meta, ch_names=[ch_names[i] for i in perm.tolist()])
out_perm = module.Solver().load_model(meta_perm).predict(x[:, perm, :])
report["channel_order_invariant"] = bool(torch.allclose(out, out_perm, atol=1e-5))
report["max_abs"] = float(out.abs().max())
print("REPORT " + json.dumps(report))
"""


def check_submission(
    zip_path: str | Path,
    n_outputs: int = contract.EMBED_DIM,
    ch_names: tuple[str, ...] | list[str] = contract.EMOTIV_32,
    n_times: int = contract.N_TIMES,
    sfreq: float = contract.SFREQ,
) -> dict[str, Any]:
    """Validate a submission ZIP against the documented contract.

    Returns a report dict with an ``ok`` flag and one entry per check; raises
    ``RuntimeError`` only if the submission cannot be loaded at all.
    """
    zip_path = Path(zip_path)
    report: dict[str, Any] = {"zip": str(zip_path)}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        report["files"] = sorted(names)
        report["flat_layout"] = all("/" not in name for name in names)
        report["has_submission_py"] = "submission.py" in names
        report["single_python_file"] = sum(name.endswith(".py") for name in names) == 1

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sub_dir = tmp_dir / "submission"
            archive.extractall(sub_dir)
            stub_pkg = tmp_dir / "stubs" / "benchmark_utils"
            stub_pkg.mkdir(parents=True)
            (stub_pkg / "__init__.py").write_text("", encoding="utf-8")
            (stub_pkg / "base_solver.py").write_text(textwrap.dedent(_STUB), encoding="utf-8")
            probe = tmp_dir / "probe.py"
            probe.write_text(_PROBE, encoding="utf-8")

            # Fresh interpreter, cwd away from the repo, only the stub on the path.
            result = subprocess.run(
                [sys.executable, str(probe), str(sub_dir), str(n_outputs),
                 json.dumps(list(ch_names)), str(n_times), str(sfreq)],
                cwd=tmp_dir,
                env={"PYTHONPATH": str(tmp_dir / "stubs"), "PATH": ""},
                capture_output=True,
                text=True,
                timeout=600,
            )  # fmt: skip
    if result.returncode != 0:
        raise RuntimeError(f"submission failed to load or predict:\n{result.stderr[-4000:]}")
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("REPORT "))
    report.update(json.loads(line[len("REPORT ") :]))

    report["shape_ok"] = report["shape"] == [7, n_outputs]
    report["dtype_ok"] = report["dtype"] in ("torch.float32", "torch.float64")
    checks = (
        "flat_layout", "has_submission_py", "single_python_file", "is_tensor", "shape_ok",
        "dtype_ok", "finite", "batch_sizes_ok", "stateless", "channel_order_invariant",
    )  # fmt: skip
    report["ok"] = all(report[name] for name in checks)
    return report
