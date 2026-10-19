#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (this file lives in scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EXP_ROOT="${EXP_ROOT:-exp_aug_larger_M}"
DATASET="${DATASET:-criteo}"
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

SAMPLE_PCT_START="${SAMPLE_PCT_START:-2.5}"
SAMPLE_PCT_STEP="${SAMPLE_PCT_STEP:-2.5}"
SAMPLE_PCT_END="${SAMPLE_PCT_END:-15}"

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/sample_frac_with_fixed_${PILOT_FRAC}_pilot_frac"
mkdir -p "${OUTDIR}"
echo "[INFO] outdir=${OUTDIR}"
echo "[INFO] sample_pct=${SAMPLE_PCT_START}:${SAMPLE_PCT_STEP}:${SAMPLE_PCT_END}  pilot_frac=${PILOT_FRAC}  seed=${SEED_SEQUENCE}  N_sim=${N_SIM}"

for sample_pct in $(seq "${SAMPLE_PCT_START}" "${SAMPLE_PCT_STEP}" "${SAMPLE_PCT_END}"); do
    # percent → fraction: 2.5 → 0.025, 15 → 0.15
    sample_frac=$(awk -v p="${sample_pct}" 'BEGIN {
        x = p / 100.0
        if (p == int(p)) printf "%.2f", x
        else printf "%.3f", x
    }')
    # filename tag in percent units: 2.5 → 002.5, 5 → 005, 15 → 015
    sample_tag=$(awk -v p="${sample_pct}" 'BEGIN {
        if (p == int(p)) printf "%03d", p
        else printf "%05.1f", p
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
