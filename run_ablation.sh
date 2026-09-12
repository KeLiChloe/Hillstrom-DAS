#!/usr/bin/env bash
# DAMS ablation: segment baselines ± DAMS M-selection vs DAST.
# Default: Criteo conversion, sample_frac=0.025 (~N=35k on percent10 base).
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-exp_aug/ablation}"
DATASET="${DATASET:-criteo}"
TARGET="${TARGET:-conversion}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.025}"
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

ALGO_LIST=(
  dast
  kmeans
  gmm
  clr
  kmeans_dams
  gmm_dams
  clr_dams
)

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}"
mkdir -p "${OUTDIR}"

# Tag = percent of loader base: 0.05→005, 0.025→002.5 (same rule as run_sample_frac.sh).
sample_tag=$(awk -v f="${SAMPLE_FRAC}" 'BEGIN {
    pct = f * 100.0
    if (pct == int(pct)) printf "%03d", pct
    else printf "%05.1f", pct
}')
outpath="${OUTDIR}/sample_frac_${sample_tag}.pkl"

echo "[INFO] outdir=${OUTDIR}"
echo "[INFO] algos=${ALGO_LIST[*]}"

if [[ -f "${outpath}" ]]; then
    echo "[SKIP] ${outpath} already exists"
    exit 0
fi

echo "[RUN ] sample_frac=${SAMPLE_FRAC} -> ${outpath}"
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
    --treatment_cost "${TREATMENT_COST}" \
    --N_sim "${N_SIM}" \
    --n_folds_dams "${N_FOLDS_DAMS}" \
    --algo_list "${ALGO_LIST[@]}" \
    --outpath "${outpath}"
