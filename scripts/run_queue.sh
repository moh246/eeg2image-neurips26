#!/usr/bin/env bash
# Run a queue of experiments back to back, resumably.
#
#   bash scripts/run_queue.sh <out_dir> <experiment>:<seed> [<experiment>:<seed> ...] [-- overrides]
#
# Each job runs scripts/train.py on configs/experiment/<experiment>.yaml with
# its checkpoints, packages and run records under <out_dir>. A finished job
# leaves a marker and is skipped on the next call; an interrupted one resumes
# from its last checkpoint. Anything after "--" is passed to every job as
# config overrides (for example data.store=/content/stores/alljoined16m_v1).
set -uo pipefail

OUT_DIR=${1:?usage: run_queue.sh <out_dir> <experiment>:<seed> ... [-- overrides]}
shift
jobs=()
while (($#)) && [[ $1 != -- ]]; do
    jobs+=("$1")
    shift
done
(($#)) && shift
overrides=("$@")

mkdir -p "$OUT_DIR/done" "$OUT_DIR/logs"
failed=0
for job in "${jobs[@]}"; do
    experiment=${job%%:*}
    seed=${job##*:}
    tag="${experiment}_seed${seed}"
    if [[ -e "$OUT_DIR/done/$tag" ]]; then
        echo "== skip $tag (done)"
        continue
    fi
    echo "== start $tag $(date -u +%FT%TZ)"
    if python scripts/train.py --config "configs/experiment/$experiment.yaml" "seed=$seed" \
        "train.checkpoint_dir=$OUT_DIR/checkpoints/$tag" "output_root=$OUT_DIR/outputs" \
        "tracking.root=$OUT_DIR/experiments" ${overrides[@]+"${overrides[@]}"} \
        > "$OUT_DIR/logs/$tag.log" 2>&1; then
        touch "$OUT_DIR/done/$tag"
        echo "== done $tag $(date -u +%FT%TZ)"
    else
        failed=1
        echo "== FAILED $tag (see $OUT_DIR/logs/$tag.log)"
    fi
done
exit $failed
