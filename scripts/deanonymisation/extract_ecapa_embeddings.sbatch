#!/bin/bash
#SBATCH --job-name=ecapa_emb_trainclean100_original
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x_%j.out

# Extract pretrained VoxCeleb ECAPA-TDNN embeddings for one LibriSpeech split
# into a single .npz file.
#
# The utterance list comes from the conversion records of CONDITION, so original
# and anonymised embeddings cover the same utterances in the same order
# (sorted by utterance ID).
#
#   AUDIO=original    the source LibriSpeech recordings (x_o). Default.
#   AUDIO=anonymised  the converted outputs (x_a).
#
# Submit from the repository root after `mkdir -p outputs/logs`:
#   sbatch scripts/deanonymisation/extract_ecapa_embeddings.sbatch
#   AUDIO=anonymised sbatch --job-name=ecapa_emb_trainclean100_knnvc_fixed3851_anonymised \
#       scripts/deanonymisation/extract_ecapa_embeddings.sbatch
#   LIMIT=8 sbatch --job-name=ecapa_emb_smoke scripts/deanonymisation/extract_ecapa_embeddings.sbatch
#
# Outputs, under outputs/embeddings/ecapa/:
#   original    librispeech_<SPLIT>_original.npz
#   anonymised  <CONDITION>_<SPLIT>_anonymised.npz
# LIMIT adds _limit<N> to the file name. Each .npz has a .json sidecar.

set -euo pipefail

AUDIO=${AUDIO:-original}
SPLIT=${SPLIT:-train-clean-100}
CONDITION=${CONDITION:-knnvc_fixed3851}
CONVERSION_GLOB=${CONVERSION_GLOB:-outputs/anonymized/knnvc/unversioned/librispeech/fixed-3851-*}
EXPECTED=${EXPECTED:-28539}
LIMIT=${LIMIT:-}
MODEL_DIR=${MODEL_DIR:-/scratch/elec/t412-speechsynth/palp/misc/pretrained_models/spkrec-ecapa-voxceleb}
QC_ENV=${QC_ENV:-/scratch/elec/t412-speechsynth/palp/envs/activate_quick_convert.sh}

source "$QC_ENV"
cd "$QC_REPO"
export HF_HUB_OFFLINE=1  # The model is already downloaded; compute nodes need no network.

case "$AUDIO" in
    original) name="librispeech_${SPLIT}_original" ;;
    anonymised) name="${CONDITION}_${SPLIT}_anonymised" ;;
    *) echo "AUDIO must be original or anonymised, got $AUDIO" >&2; exit 1 ;;
esac
extra=()
if [[ -n "$LIMIT" ]]; then
    name="${name}_limit${LIMIT}"
    extra=(--limit "$LIMIT")
fi
OUT=outputs/embeddings/ecapa/${name}.npz

# shellcheck disable=SC2206  # Intentional glob expansion over shard directories.
CONVERSION_DIRS=($CONVERSION_GLOB)
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "audio=$AUDIO split=$SPLIT condition=$CONDITION dirs=${#CONVERSION_DIRS[@]} expected=$EXPECTED out=$OUT"

python scripts/deanonymisation/extract_ecapa_embeddings.py \
    --conversion-dirs "${CONVERSION_DIRS[@]}" \
    --split "$SPLIT" \
    --audio "$AUDIO" \
    --expected "$EXPECTED" \
    --model-dir "$MODEL_DIR" \
    --out "$OUT" \
    --device cuda \
    --num-workers 3 \
    "${extra[@]}"
