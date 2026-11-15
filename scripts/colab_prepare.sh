#!/usr/bin/env bash
# Build the Alljoined-1.6M store and its DINOv2-giant targets on a GPU machine
# (written for Colab: two CPU cores, one GPU, a persistent directory on Drive).
#
#   bash scripts/colab_prepare.sh <work_dir> <persist_dir>
#
# <work_dir> is fast scratch space (raw download, shards, store); <persist_dir>
# receives the finished store. Every step resumes, so re-run after a disconnect.
# Running this script accepts the dataset's CC-BY-NC-ND-4.0 licence.
set -euo pipefail

WORK_DIR=${1:?usage: colab_prepare.sh <work_dir> <persist_dir>}
PERSIST_DIR=${2:?usage: colab_prepare.sh <work_dir> <persist_dir>}
CONFIG=configs/data/alljoined16m.yaml
STORE="$WORK_DIR/stores/alljoined16m_v1"
OVERRIDES=("data.root=$WORK_DIR/data" "data.store=$STORE" "data.embeddings=$STORE/embeddings.npy")
WORKERS=${WORKERS:-$(nproc)}
export ALLJOINED_ACCEPT_LICENCE=1

echo "== 1/4 download"
python scripts/prepare_alljoined.py --config "$CONFIG" --download-only "${OVERRIDES[@]}"

echo "== 2/4 preprocess with $WORKERS workers"
pids=()
for ((i = 0; i < WORKERS; i++)); do
    python scripts/prepare_alljoined.py --config "$CONFIG" --skip-download \
        --part "$i/$WORKERS" "${OVERRIDES[@]}" > "$WORK_DIR/prepare_part$i.log" 2>&1 &
    pids+=($!)
done
while kill -0 "${pids[@]}" 2> /dev/null; do
    echo "   shards: $(find "$STORE/shards" -name '*.npz' 2> /dev/null | wc -l)"
    sleep 60
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/prepare_alljoined.py --config "$CONFIG" --skip-download "${OVERRIDES[@]}"

echo "== 3/4 image targets"
python scripts/embed_images.py --config "$CONFIG" --backend truncated "${OVERRIDES[@]}"

echo "== 4/4 copy to $PERSIST_DIR"
DEST="$PERSIST_DIR/stores/alljoined16m_v1"
mkdir -p "$DEST"
for name in eeg.npy index.npz images.json meta.json embeddings.npy; do
    cp "$STORE/$name" "$DEST/$name"
done
(cd "$DEST" && sha256sum eeg.npy index.npz images.json meta.json embeddings.npy | tee SHA256SUMS)
ls -l "$DEST"
echo "== done"
