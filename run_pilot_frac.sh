#!/usr/bin/env bash
set -euo pipefail

MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-logistic}" # diff_in_means, logistic, gamma
SEED_SEQUENCE="${SEED_SEQUENCE:-202}"
DATASET="${DATASET:-criteo}"
TARGET="${TARGET:-visit}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.1}"
N_JOBS="${N_JOBS:-1}"

OUTDIR="exp_june/${DATASET}/${TARGET}/pilot_frac_with_fixed_${SAMPLE_FRAC}_sample_frac"
mkdir -p "${OUTDIR}"

for pilot_int in $(seq 5 5 50); do
    pilot_frac=$(printf "0.%02d" "${pilot_int}")
    pilot_tag=$(printf "%03d" "${pilot_int}")
    outpath="${OUTDIR}/pilot_frac_${pilot_tag}_imp.pkl"

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
        --n_jobs "${N_JOBS}" \
        --action_method "${ACTION_METHOD}" \
        --outpath "${outpath}"
done
