#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-exp_july}"
DATASET="${DATASET:-hillstrom}"
TARGET="${TARGET:-conversion}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
MU_MODEL_TYPE="${MU_MODEL_TYPE:-mlp_clf}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-diff_in_means}"
SEED_SEQUENCE="${SEED_SEQUENCE:-1024}"

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/pilot_frac_with_fixed_${SAMPLE_FRAC}_sample_frac"
mkdir -p "${OUTDIR}"

for pilot_int in $(seq 5 5 50); do
    pilot_frac=$(printf "0.%02d" "${pilot_int}")
    pilot_tag=$(printf "%03d" "${pilot_int}")
    outpath="${OUTDIR}/pilot_frac_${pilot_tag}.pkl"

    if [[ -f "${outpath}" ]]; then
        echo "[SKIP] ${outpath} already exists"
        continue
    fi

    echo "[RUN ] pilot_frac=${pilot_frac} -> ${outpath}"
    python run_sims.py \
        --mu_model_type "${MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${SAMPLE_FRAC}" \
        --pilot_frac "${pilot_frac}" \
        --action_method "${ACTION_METHOD}" \
        --outpath "${outpath}"
done
