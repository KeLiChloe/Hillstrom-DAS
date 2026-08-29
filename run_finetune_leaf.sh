#!/usr/bin/env bash
# DAST min_leaf_size finetune (after finetune_mu.py).
# For each (dataset, target) × available μ type, writes one keyed dast entry.
#
# Usage:
#   ./run_finetune_leaf.sh
#       → all targets under all datasets × μ types present in each hparams JSON
#   DATASET=hillstrom ./run_finetune_leaf.sh
#   DATASET=hillstrom TARGET=conversion ./run_finetune_leaf.sh
#   MU_MODEL_TYPE=lightgbm_reg ./run_finetune_leaf.sh
#   MU_MODEL_TYPES="lightgbm_reg mlp_reg" ./run_finetune_leaf.sh
#
# Optional: VALUE_TYPE_* ACTION_METHOD SAMPLE_FRAC SEED N_FOLDS_DAMS
set -euo pipefail

VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-diff_in_means}"
TREATMENT_COST="${TREATMENT_COST:-0}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-42}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"

ALL_DATASETS=(hillstrom criteo)

DEFAULT_MU_MODEL_TYPES=(
    lightgbm_reg
    mlp_reg
    linear
    logistic
    lightgbm_clf
    mlp_clf
)

targets_for_dataset() {
    case "$1" in
        hillstrom) echo conversion visit spend ;;
        criteo) echo conversion visit ;;
        *)
            echo "[finetune_leaf] unknown DATASET=$1" >&2
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
        echo "[finetune_leaf] TARGET alone is not supported; set DATASET too" >&2
        exit 1
    else
        for ds in "${ALL_DATASETS[@]}"; do
            for t in $(targets_for_dataset "${ds}"); do
                PAIRS_LIST+=("${ds}:${t}")
            done
        done
    fi
    if [[ ${#PAIRS_LIST[@]} -eq 0 ]]; then
        echo "[finetune_leaf] empty pair list" >&2
        exit 1
    fi
}

# μ types present under models[] in hparams/{ds}/{tgt}.json
models_in_hparams() {
    local ds="$1" tgt="$2"
    local path="hparams/${ds}/${tgt}.json"
    if [[ ! -f "${path}" ]]; then
        return 1
    fi
    python - "${path}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
models = payload.get("models") or {}
print(" ".join(sorted(models.keys())))
PY
}

build_pairs

if [[ -n "${MU_MODEL_TYPE:-}" ]]; then
    MU_REQUESTED=("${MU_MODEL_TYPE}")
elif [[ -n "${MU_MODEL_TYPES:-}" ]]; then
    # shellcheck disable=SC2206
    MU_REQUESTED=(${MU_MODEL_TYPES})
else
    MU_REQUESTED=("${DEFAULT_MU_MODEL_TYPES[@]}")
fi

echo "[finetune_leaf] pairs=${PAIRS_LIST[*]}"
echo "[finetune_leaf] requested_mu=${MU_REQUESTED[*]} action=${ACTION_METHOD}"

fail_count=0
for pair in "${PAIRS_LIST[@]}"; do
    DATASET="${pair%%:*}"
    TARGET="${pair#*:}"
    hparams_path="hparams/${DATASET}/${TARGET}.json"

    if [[ ! -f "${hparams_path}" ]]; then
        echo "[finetune_leaf] SKIP ${DATASET}/${TARGET}: missing ${hparams_path} (run finetune_mu first)" >&2
        fail_count=$((fail_count + 1))
        continue
    fi

    available_str="$(models_in_hparams "${DATASET}" "${TARGET}" || true)"
    if [[ -z "${available_str}" ]]; then
        echo "[finetune_leaf] SKIP ${DATASET}/${TARGET}: no models[] in ${hparams_path}" >&2
        fail_count=$((fail_count + 1))
        continue
    fi
    # shellcheck disable=SC2206
    available=(${available_str})

    # spend is continuous: classifiers are not tuned by finetune_mu and must not run.
    MU_CANDIDATES=("${MU_REQUESTED[@]}")
    if [[ "${TARGET}" == "spend" ]]; then
        MU_CANDIDATES=()
        for m in "${MU_REQUESTED[@]}"; do
            case "${m}" in
                logistic|mlp_clf|lightgbm_clf)
                    echo "[finetune_leaf] SKIP ${DATASET}/${TARGET} μ=${m} (classifier unused for spend)"
                    ;;
                *)
                    MU_CANDIDATES+=("${m}")
                    ;;
            esac
        done
    fi

    MU_LIST=()
    for m in "${MU_CANDIDATES[@]}"; do
        for a in "${available[@]}"; do
            if [[ "${m}" == "${a}" ]]; then
                MU_LIST+=("${m}")
                break
            fi
        done
    done

    if [[ ${#MU_LIST[@]} -eq 0 ]]; then
        echo "[finetune_leaf] SKIP ${DATASET}/${TARGET}: none of requested μ in models (${available_str})" >&2
        fail_count=$((fail_count + 1))
        continue
    fi

    echo ""
    echo "######################################################################"
    echo "[finetune_leaf] PAIR ${DATASET}/${TARGET} mu=${MU_LIST[*]} (available=${available_str})"
    echo "######################################################################"

    for MU_MODEL_TYPE in "${MU_LIST[@]}"; do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "[finetune_leaf] START ${DATASET}/${TARGET} μ=${MU_MODEL_TYPE}"
        echo "--------------------------------------------------------------------"
        if python finetune_leaf.py \
            --dataset "${DATASET}" \
            --target "${TARGET}" \
            --mu_model_type "${MU_MODEL_TYPE}" \
            --value_type_dast "${VALUE_TYPE_DAST}" \
            --value_type_dams "${VALUE_TYPE_DAMS}" \
            --action_method "${ACTION_METHOD}" \
            --treatment_cost "${TREATMENT_COST}" \
            --sample_frac "${SAMPLE_FRAC}" \
            --seed "${SEED}" \
            --n_folds_dams "${N_FOLDS_DAMS}"
        then
            echo "[finetune_leaf] DONE  ${DATASET}/${TARGET} μ=${MU_MODEL_TYPE}"
        else
            echo "[finetune_leaf] FAIL  ${DATASET}/${TARGET} μ=${MU_MODEL_TYPE} (continue)" >&2
            fail_count=$((fail_count + 1))
        fi
    done
done

echo ""
if [[ "${fail_count}" -gt 0 ]]; then
    echo "[finetune_leaf] finished with ${fail_count} failure(s)/skip(s)." >&2
    exit 1
fi
echo "[finetune_leaf] all pairs × μ types finished."
