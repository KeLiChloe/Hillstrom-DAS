#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-exp_july}"
DATASET="${DATASET:-hillstrom}"
TARGET="${TARGET:-conversion}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
META_LEARNER_MU_MODEL_TYPE="${META_LEARNER_MU_MODEL_TYPE:-mlp_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-logistic}"
SEED_SEQUENCE="${SEED_SEQUENCE:-899}"
N_SIM="${N_SIM:-50}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"

BASE_OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/pilot_frac_with_fixed_${SAMPLE_FRAC}_sample_frac"
OUTDIR="${BASE_OUTDIR}"
if [[ -e "${OUTDIR}" ]]; then
    i=1
    while [[ -e "${BASE_OUTDIR}(${i})" ]]; do
        i=$((i + 1))
    done
    OUTDIR="${BASE_OUTDIR}(${i})"
    echo "[INFO] ${BASE_OUTDIR} exists → using ${OUTDIR}"
fi
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
        --meta_learner_mu_model_type "${META_LEARNER_MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${SAMPLE_FRAC}" \
        --pilot_frac "${pilot_frac}" \
        --action_method "${ACTION_METHOD}" \
        --N_sim "${N_SIM}" \
        --n_folds_dams "${N_FOLDS_DAMS}" \
        --outpath "${outpath}"
done
