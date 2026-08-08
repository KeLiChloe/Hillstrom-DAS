#!/usr/bin/env bash
# One-shot permanent μ finetune for one (dataset, target).
# Tunes ALL μ model types; writes hparams/{DATASET}/{TARGET}.json
set -euo pipefail

DATASET="${DATASET:-hillstrom}"
TARGET="${TARGET:-conversion}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-42}"
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
fi

python finetune_mu.py \
    --dataset "${DATASET}" \
    --target "${TARGET}" \
    --sample_frac "${SAMPLE_FRAC}" \
    --seed "${SEED}" \
    "${FORCE_FLAG[@]}"
