#!/bin/bash
#SBATCH --job-name=knnvc_anon_fixed3851_tc100
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=outputs/logs/%x_%A_%a.out

# KNN-VC anonymisation of train-clean-100 into one fixed target voice:
# speaker 3851 (female) from train-clean-360.
#
# Each array task converts one shard, split by the first digit of the speaker ID:
#
#   task 0-8  train-clean-100, speakers 1* ... 9*  (about 200-4,400 utterances each)
#
# All tasks build the same matching set: up to 8 minutes of TARGET_SPEAKER from
# TARGET_SPLIT, picked by a seeded shuffle. Each shard writes to its own output
# directory, so tasks never share records.
#
# Speaker 3851 is in train-clean-360. Exclude 3851 from attacker training
# whenever it is the target voice.
#
# Submit from the repository root after `mkdir -p outputs/logs`:
#   sbatch scripts/deanonymisation/knnvc_fixed3851_tc100_array.sbatch
#   sbatch --array=8 scripts/deanonymisation/knnvc_fixed3851_tc100_array.sbatch   # smallest shard only
#
# The log is outputs/logs/<job-name>_<array-job-id>_<task>.out. Its first line
# names the shard. Conversion records skip finished outputs, so resubmitting a
# task resumes it.

set -euo pipefail

LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:-/scratch/elec/t412-speechsynth/DATA/LibriSpeech/tarballs/LibriSpeech}
SOURCE_SPLIT=${SOURCE_SPLIT:-train-clean-100}
TARGET_SPEAKER=${TARGET_SPEAKER:-3851}
TARGET_SPLIT=${TARGET_SPLIT:-train-clean-360}
TARGET_MAX_SECONDS=${TARGET_MAX_SECONDS:-480}
QC_ENV=${QC_ENV:-/scratch/elec/t412-speechsynth/palp/envs/activate_quick_convert.sh}

SPEAKER_PREFIXES=("1*" "2*" "3*" "4*" "5*" "6*" "7*" "8*" "9*")
SOURCE_SPEAKER=${SPEAKER_PREFIXES[$SLURM_ARRAY_TASK_ID]}

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
