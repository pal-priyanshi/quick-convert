#!/bin/bash
#SBATCH --job-name=knnvc_anon_fixed1069_tc360-dev-test
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=outputs/logs/%x_%A_%a.out

# Baseline: anonymise every source speaker into one fixed KNN-VC target voice.
#
# Each array task converts one shard. Shards split train-clean-360 by the first
# digit of the speaker ID; dev-clean and test-clean are one shard each:
#
#   task  0-8  train-clean-360, speakers 1* ... 9*
#   task  9    dev-clean
#   task  10   test-clean
#
# All tasks build the same matching set: up to 8 minutes of TARGET_SPEAKER from
# train-clean-100, picked by a seeded shuffle. Each shard writes to its own
# output directory, so tasks never share records.
#
# Submit from the repository root after `mkdir -p outputs/logs`:
#   sbatch scripts/deanonymisation/knnvc_fixed1069_array.sbatch
#   sbatch --array=10 scripts/deanonymisation/knnvc_fixed1069_array.sbatch   # test-clean only
#
# The log is outputs/logs/<job-name>_<array-job-id>_<task>.out. Its first line
# names the shard. Conversion records skip finished outputs, so resubmitting a
# task resumes it.

set -euo pipefail

LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:-/scratch/elec/t412-speechsynth/DATA/LibriSpeech/tarballs/LibriSpeech}
TARGET_SPEAKER=${TARGET_SPEAKER:-1069}
TARGET_SPLIT=${TARGET_SPLIT:-train-clean-100}
TARGET_MAX_SECONDS=${TARGET_MAX_SECONDS:-480}
QC_ENV=${QC_ENV:-/scratch/elec/t412-speechsynth/palp/envs/activate_quick_convert.sh}

SHARDS=(
    "train-clean-360 1*"
    "train-clean-360 2*"
    "train-clean-360 3*"
    "train-clean-360 4*"
    "train-clean-360 5*"
    "train-clean-360 6*"
    "train-clean-360 7*"
    "train-clean-360 8*"
    "train-clean-360 9*"
    "dev-clean *"
    "test-clean *"
)
read -r SOURCE_SPLIT SOURCE_SPEAKER <<< "${SHARDS[$SLURM_ARRAY_TASK_ID]}"

echo "shard=$SLURM_ARRAY_TASK_ID source=$SOURCE_SPLIT speakers=$SOURCE_SPEAKER target=$TARGET_SPLIT/$TARGET_SPEAKER max_seconds=$TARGET_MAX_SECONDS"

source "$QC_ENV"
cd "$QC_REPO"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

start=$SECONDS
quick-convert convert_knnvc_librispeech \
    root="$LIBRISPEECH_ROOT" \
    source_split="$SOURCE_SPLIT" \
    source_speaker="$SOURCE_SPEAKER" \
    target_speaker="$TARGET_SPEAKER" \
    "target_splits=[$TARGET_SPLIT]" \
    target_max_utterances=null \
    target_max_seconds="$TARGET_MAX_SECONDS"

echo "shard=$SLURM_ARRAY_TASK_ID finished in $((SECONDS - start))s"
