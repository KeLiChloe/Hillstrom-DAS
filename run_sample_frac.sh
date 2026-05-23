#!/usr/bin/env bash
set -euo pipefail

MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
SEED_SEQUENCE="${SEED_SEQUENCE:-202}"
PILOT_FRAC="${PILOT_FRAC:-0.20}"
N_JOBS="${N_JOBS:-1}"

DATASET="${DATASET:-criteo}"
TARGET="${TARGET:-visit}"


OUTDIR="exp_may/${DATASET}/${TARGET}/sample_frac_with_fixed_020_pilot"
mkdir -p "${OUTDIR}"

for sample_int in $(seq 5 5 50); do
    sample_frac=$(printf "0.%02d" "${sample_int}")
    sample_tag=$(printf "%03d" "${sample_int}")
    outpath="${OUTDIR}/sample_frac_${sample_tag}.pkl"

    if [[ -f "${outpath}" ]]; then
        echo "[SKIP] sample_frac=${sample_frac} exists: ${outpath}"
        continue
    fi

    echo "[RUN ] sample_frac=${sample_frac}, pilot_frac=${PILOT_FRAC} -> ${outpath}"
    python run_sims.py \
        --mu_model_type "${MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${sample_frac}" \
        --pilot_frac "${PILOT_FRAC}" \
        --n_jobs "${N_JOBS}" \
        --outpath "${outpath}"
done
