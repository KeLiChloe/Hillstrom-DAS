#!/usr/bin/env bash
# Rerun the Hillstrom conversion sample_frac=1 experiment for runtime analysis.
# DAST uses N_FOLDS_DAMS; MST is configured independently in run_sims.py.
# Each algorithm records its total fit/selection/prediction/OPE time in "time".
# Usage (after activating the Python environment):
#   bash scripts/run_computational_time.sh
# Optional overrides: PYTHON_BIN, N_SIM, OUTDIR, N_FOLDS_DAMS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

N_SIM="${N_SIM:-50}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-1}"
OUTDIR="${OUTDIR:-exp_aug_larger_M/computational_time}"
OUTPATH="${OUTDIR}/n_fold_dams_${N_FOLDS_DAMS}.pkl"

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

# Activated Conda environments usually provide python; macOS may only provide
# python3. An explicit PYTHON_BIN always takes precedence.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "[ERROR] No Python found. Activate your environment or set PYTHON_BIN." >&2
        exit 127
    fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

mkdir -p "${OUTDIR}"
echo "[PYTHON] $(command -v "${PYTHON_BIN}")"
echo "[RUN] Hillstrom/conversion, sample_frac=1, pilot_frac=0.40, N_sim=${N_SIM}"
echo "[OUT] ${OUTPATH}"
if [[ "${N_FOLDS_DAMS}" == "1" ]]; then
    echo "[DAST] Single 70%/30% pilot holdout for DAMS (n_folds_dams=${N_FOLDS_DAMS})."
else
    echo "[DAST] ${N_FOLDS_DAMS}-fold CV for DAMS (n_folds_dams=${N_FOLDS_DAMS})."
fi
echo "[TIME] Saving total runtime per algorithm in time (seconds)."

"${PYTHON_BIN}" -u run_sims.py \
    --dataset hillstrom \
    --target conversion \
    --sample_frac 1.0 \
    --pilot_frac 0.40 \
    --mu_model_type lightgbm_reg \
    --meta_learner_mu_model_type mlp_reg \
    --treatment_cost 0 \
    --seed_sequence 1088 \
    --N_sim "${N_SIM}" \
    --n_folds_dams "${N_FOLDS_DAMS}" \
    --algo_list "${ALGO_LIST[@]}" \
    --outpath "${OUTPATH}"
