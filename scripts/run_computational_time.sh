#!/usr/bin/env bash
# Rerun the Hillstrom conversion sample_frac=1 experiment for runtime analysis.
# DAST uses a single 70%/30% pilot holdout for DAMS, like the existing MST setup.
# Each algorithm records its total fit/selection/prediction/OPE time in "time".
# Usage (after activating the Python environment):
#   bash scripts/run_computational_time.sh
# Optional overrides: PYTHON_BIN, N_SIM, OUTDIR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
N_SIM="${N_SIM:-50}"
OUTDIR="${OUTDIR:-exp_aug_larger_M/computational_time}"
OUTPATH="${OUTDIR}/n_fold_dams_1.pkl"

# Match the algorithm list saved in the original sample_frac_100.pkl,
# including PolicyTree, which is absent from run_sims.py's default list.
ALGO_LIST=(
    dast causal_forest mst clr kmeans gmm
    t_learner s_learner x_learner dr_learner policytree
)

if [[ -f "${OUTPATH}" ]]; then
    echo "[SKIP] ${OUTPATH} already exists; use a different OUTDIR for a new run."
    exit 0
fi

mkdir -p "${OUTDIR}"
echo "[RUN] Hillstrom/conversion, sample_frac=1, pilot_frac=0.40, N_sim=${N_SIM}"
echo "[OUT] ${OUTPATH}"
echo "[DAST] Single 70%/30% pilot holdout for DAMS (n_folds_dams=1)."
echo "[TIME] Saving total runtime per algorithm in time (seconds)."

"${PYTHON_BIN}" -u run_sims.py \
    --dataset hillstrom \
    --target conversion \
    --sample_frac 1.0 \
    --pilot_frac 0.40 \
    --mu_model_type lightgbm_reg \
    --meta_learner_mu_model_type mlp_reg \
    --value_type_dast hybrid \
    --value_type_dams hybrid \
    --action_method diff_in_means \
    --treatment_cost 0 \
    --seed_sequence 1088 \
    --N_sim "${N_SIM}" \
    --n_folds_dams 1 \
    --algo_list "${ALGO_LIST[@]}" \
    --outpath "${OUTPATH}"
