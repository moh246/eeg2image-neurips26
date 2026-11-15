#!/usr/bin/env bash
# Evaluations and extra runs that follow the main comparison; run after the training queue.
#
#   bash scripts/colab_followups.sh <runs_dir> [-- overrides]
#
# <runs_dir> is the directory the queue wrote to (experiments/, outputs/). Each step is
# independent: a failure is reported and the next step still runs.
set -uo pipefail

RUNS_DIR=${1:?usage: colab_followups.sh <runs_dir> [-- overrides]}
shift
[[ ${1:-} == -- ]] && shift
overrides=("$@")
common=(--outputs "$RUNS_DIR/outputs" --experiments "$RUNS_DIR/experiments")

step() {
    echo "== $1 $(date -u +%FT%TZ)"
    shift
    "$@" ${overrides[@]+"${overrides[@]}"} || echo "== FAILED: $*"
}

for name in alljoined_conv alljoined_conv_transformer alljoined_conv_transformer_causal; do
    step "ensemble $name" python scripts/evaluate_ensemble.py \
        --config "configs/experiment/$name.yaml" --names "$name" \
        --label "ensemble_$name" "${common[@]}"
done
step "ensemble of both heads" python scripts/evaluate_ensemble.py \
    --config configs/experiment/alljoined_conv.yaml \
    --names alljoined_conv alljoined_conv_transformer \
    --label ensemble_conv_and_transformer "${common[@]}"

for name in alljoined_conv alljoined_channel_transformer; do
    step "channel subsets $name" python scripts/evaluate_channels.py \
        --config "configs/experiment/$name.yaml" --name "$name" --seed 0 \
        --keep 32 24 16 8 "${common[@]}"
done

echo "== long transformer run $(date -u +%FT%TZ)"
bash scripts/run_queue.sh "$RUNS_DIR" alljoined_conv_transformer_long:0 \
    -- ${overrides[@]+"${overrides[@]}"}
echo "== followups finished $(date -u +%FT%TZ)"
