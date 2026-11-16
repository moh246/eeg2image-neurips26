# Experiment records

Each run writes `runs/<run_id>/` with:

- `config.yaml` — the fully resolved config;
- `meta.json` — git commit, dirty flag, config hash, seed, library versions;
- `metrics.jsonl` — one row per logged step;
- `summary.json` — final metrics.

`registry.jsonl` has one line per finished run. The run id is
`<UTC timestamp>_<experiment name>_<short commit>`.

`evaluations/<label>.json` holds evaluations of finished runs (seed ensembles, channel
subsets), each with the commit of the evaluating code and the run ids it scored.

Runs are made on a GPU machine and merged here with `scripts/import_runs.py`; the tables in
the README and report are generated from the registry with `make results`.

Records are small text files and are committed. Checkpoints and submission ZIPs go to
`outputs/<run_id>/`, which git ignores. A result quoted anywhere in this repository must cite
its run id, and the run must have `git_dirty: false`.
