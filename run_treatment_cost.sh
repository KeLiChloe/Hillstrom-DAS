#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-exp_aug}"
DATASET="${DATASET:-criteo}"
TARGET="${TARGET:-conversion}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.1}"
PILOT_FRAC="${PILOT_FRAC:-0.40}"
MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
META_LEARNER_MU_MODEL_TYPE="${META_LEARNER_MU_MODEL_TYPE:-mlp_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-diff_in_means}"
SEED_SEQUENCE="${SEED_SEQUENCE:-1088}"
N_SIM="${N_SIM:-50}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"

# Space-separated cost grid: 0, 0.0005, ..., 0.002 (step 0.0005)
COST_VALUES="${COST_VALUES:-0 0.0005 0.001 0.0015 0.002}"

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/treatment_cost_with_fixed_${SAMPLE_FRAC}_sample_frac_${PILOT_FRAC}_pilot_frac"
mkdir -p "${OUTDIR}"
echo "[INFO] outdir=${OUTDIR}"
echo "[INFO] cost grid: ${COST_VALUES}"

for cost in ${COST_VALUES}; do
    # Filesystem tag: 0 -> cost_0, 0.01 -> cost_0p01
    cost_tag=$(echo "${cost}" | sed 's/\./p/')
    outpath="${OUTDIR}/treatment_cost_${cost_tag}.pkl"

    if [[ -f "${outpath}" ]]; then
        echo "[SKIP] ${outpath} already exists"
        continue
    fi

    echo "[RUN ] treatment_cost=${cost} -> ${outpath}"
    python run_sims.py \
        --mu_model_type "${MU_MODEL_TYPE}" \
        --meta_learner_mu_model_type "${META_LEARNER_MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${SAMPLE_FRAC}" \
        --pilot_frac "${PILOT_FRAC}" \
        --action_method "${ACTION_METHOD}" \
        --treatment_cost "${cost}" \
        --N_sim "${N_SIM}" \
        --n_folds_dams "${N_FOLDS_DAMS}" \
        --outpath "${outpath}"
done
