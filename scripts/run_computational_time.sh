#!/usr/bin/env bash
# Rerun the Hillstrom conversion sample_frac=1 experiment for runtime analysis.
# Resume the historical fold5 timing experiment:
# DAST 5 folds, MST independently fixed at 5 folds, Causal Forest 1000 trees.
# Each algorithm records its total fit/selection/prediction/OPE time in "time".
# Usage (after activating the Python environment):
#   bash scripts/run_computational_time.sh
# Optional overrides: PYTHON_BIN, N_SIM, OUTDIR, N_FOLDS_DAMS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

N_SIM="${N_SIM:-50}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"
OUTDIR="${OUTDIR:-exp_aug_larger_M/computational_time}"
OUTPATH="${OUTDIR}/n_fold_dams_${N_FOLDS_DAMS}.pkl"

# Match the algorithm list saved in the original sample_frac_100.pkl,
# including PolicyTree, which is absent from run_sims.py's default list.
ALGO_LIST=(
    dast causal_forest mst clr kmeans gmm
    t_learner s_learner x_learner dr_learner policytree
)

if [[ "${N_FOLDS_DAMS}" != "5" ]]; then
    echo "[ERROR] This script is configured to resume the historical fold5 experiment." >&2
    exit 1
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
echo "[DAST] ${N_FOLDS_DAMS}-fold CV for DAMS (n_folds_dams=${N_FOLDS_DAMS})."
echo "[PROFILE] MST 5 folds; Causal Forest 1000 trees."
if [[ -f "${OUTPATH}" ]]; then
    echo "[RESUME] Keeping saved results; continuing to ${N_SIM} successful simulations."
fi
echo "[TIME] Saving total runtime per algorithm in time (seconds)."

"${PYTHON_BIN}" -u - \
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
    --outpath "${OUTPATH}" <<'PY'
# These overrides exist only in this Python process. Normal experiment scripts
# continue using the defaults in run_sims.py and segmentation.py.
import runpy
import sys
import causal_forest
import segmentation

fit_cf = causal_forest.fit_multiarm_causal_forest
fit_mst = segmentation.run_mst_dams

def historical_cf(*args, **kwargs):
    kwargs["num_trees"] = 1000
    return fit_cf(*args, **kwargs)

def historical_mst(*args, **kwargs):
    kwargs["n_folds"] = 5
    return fit_mst(*args, **kwargs)

causal_forest.fit_multiarm_causal_forest = historical_cf
segmentation.run_mst_dams = historical_mst
sys.argv[0] = "run_sims.py"
runpy.run_path("run_sims.py", run_name="__main__")
PY
