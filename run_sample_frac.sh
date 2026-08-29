#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-exp_aug}"
DATASET="${DATASET:-hillstrom}"
TARGET="${TARGET:-conversion}"
PILOT_FRAC="${PILOT_FRAC:-0.40}"
MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
META_LEARNER_MU_MODEL_TYPE="${META_LEARNER_MU_MODEL_TYPE:-mlp_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
ACTION_METHOD="${ACTION_METHOD:-diff_in_means}"
TREATMENT_COST="${TREATMENT_COST:-0}"
SEED_SEQUENCE="${SEED_SEQUENCE:-1088}"
N_SIM="${N_SIM:-50}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/sample_frac_with_fixed_${PILOT_FRAC}_pilot_frac"
mkdir -p "${OUTDIR}"
echo "[INFO] outdir=${OUTDIR}"

for sample_int in $(seq 2.5 2.5 25); do
    # sample_int = percent of the loader base (Criteo: percent10 slice).
    # Half-percent steps need 3 decimals (2.5 → 0.025), not %.2f.
    sample_frac=$(awk -v i="${sample_int}" 'BEGIN { printf "%.3f", i / 100.0 }')
    # Tag = percent: 10→010, 2.5→002.5 (keeps a decimal for half steps).
    sample_tag=$(awk -v i="${sample_int}" 'BEGIN {
        if (i == int(i)) printf "%03d", i
        else printf "%05.1f", i
    }')
    outpath="${OUTDIR}/sample_frac_${sample_tag}.pkl"

    if [[ -f "${outpath}" ]]; then
        echo "[SKIP] ${outpath} already exists"
        continue
    fi

    echo "[RUN ] sample_frac=${sample_frac} -> ${outpath}"
    python run_sims.py \
        --mu_model_type "${MU_MODEL_TYPE}" \
        --meta_learner_mu_model_type "${META_LEARNER_MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        --target "${TARGET}" \
        --sample_frac "${sample_frac}" \
        --pilot_frac "${PILOT_FRAC}" \
        --action_method "${ACTION_METHOD}" \
        --treatment_cost "${TREATMENT_COST}" \
        --N_sim "${N_SIM}" \
        --n_folds_dams "${N_FOLDS_DAMS}" \
        --outpath "${outpath}"
done
