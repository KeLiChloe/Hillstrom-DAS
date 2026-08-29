#!/usr/bin/env bash
# Permanent μ finetune over (dataset, target) pairs.
# Writes/overwrites models in hparams/{dataset}/{target}.json (dast preserved).
#
# Usage:
#   ./run_finetune_mu.sh
#       → all targets under all datasets
#   DATASET=hillstrom ./run_finetune_mu.sh
#       → all targets for that dataset
#   DATASET=hillstrom TARGET=conversion ./run_finetune_mu.sh
#       → single pair
#
# Optional: SAMPLE_FRAC SEED
set -euo pipefail

SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-42}"

ALL_DATASETS=(hillstrom criteo)

targets_for_dataset() {
    case "$1" in
        hillstrom) echo conversion visit spend ;;
        criteo) echo conversion visit ;;
        *)
            echo "[finetune_mu] unknown DATASET=$1" >&2
            return 1
            ;;
    esac
}

build_pairs() {
    local ds t
    PAIRS_LIST=()
    if [[ -n "${DATASET:-}" && -n "${TARGET:-}" ]]; then
        targets_for_dataset "${DATASET}" >/dev/null
        PAIRS_LIST=("${DATASET}:${TARGET}")
    elif [[ -n "${DATASET:-}" ]]; then
        for t in $(targets_for_dataset "${DATASET}"); do
            PAIRS_LIST+=("${DATASET}:${t}")
        done
    elif [[ -n "${TARGET:-}" ]]; then
        echo "[finetune_mu] TARGET alone is not supported; set DATASET too" >&2
        exit 1
    else
        for ds in "${ALL_DATASETS[@]}"; do
            for t in $(targets_for_dataset "${ds}"); do
                PAIRS_LIST+=("${ds}:${t}")
            done
        done
    fi
    if [[ ${#PAIRS_LIST[@]} -eq 0 ]]; then
        echo "[finetune_mu] empty pair list" >&2
        exit 1
    fi
}

build_pairs
echo "[finetune_mu] pairs=${PAIRS_LIST[*]} sample_frac=${SAMPLE_FRAC} seed=${SEED}"

fail_count=0
for pair in "${PAIRS_LIST[@]}"; do
    DATASET="${pair%%:*}"
    TARGET="${pair#*:}"
    echo ""
    echo "######################################################################"
    echo "[finetune_mu] START ${DATASET}/${TARGET}"
    echo "######################################################################"
    if python finetune_mu.py \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${SAMPLE_FRAC}" \
        --seed "${SEED}"
    then
        echo "[finetune_mu] DONE  ${DATASET}/${TARGET}"
    else
        echo "[finetune_mu] FAIL  ${DATASET}/${TARGET} (continue)" >&2
        fail_count=$((fail_count + 1))
    fi
done

echo ""
if [[ "${fail_count}" -gt 0 ]]; then
    echo "[finetune_mu] finished with ${fail_count} failure(s)." >&2
    exit 1
fi
echo "[finetune_mu] all pairs finished."
