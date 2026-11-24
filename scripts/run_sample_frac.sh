#!/usr/bin/env bash
# Reproduce the Criteo conversion sweep (sample_frac=0.025:0.025:0.15).
# Use a separate root so existing experiment files are not skipped or replaced.
set -euo pipefail

# Always run from repo root (this file lives in scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EXP_ROOT="${EXP_ROOT:-exp_nov_MST_fold1}"
DATASET="${DATASET:-criteo}"
CRITEO_SOURCE="${CRITEO_SOURCE:-local}"  # local or sklift
CRITEO_CSV_PATH="${CRITEO_CSV_PATH:-${ROOT}/data/criteo10.csv.gz}"
TARGET="${TARGET:-conversion}"
PILOT_FRAC="${PILOT_FRAC:-0.40}"
MU_MODEL_TYPE="${MU_MODEL_TYPE:-lightgbm_reg}"
META_LEARNER_MU_MODEL_TYPE="${META_LEARNER_MU_MODEL_TYPE:-mlp_reg}"
VALUE_TYPE_DAST="${VALUE_TYPE_DAST:-hybrid}"
VALUE_TYPE_DAMS="${VALUE_TYPE_DAMS:-hybrid}"
TREATMENT_COST="${TREATMENT_COST:-0}"
SEED_SEQUENCE="${SEED_SEQUENCE:-1088}"
N_SIM="${N_SIM:-10}"
N_FOLDS_DAMS="${N_FOLDS_DAMS:-5}"

SAMPLE_PCT_START="${SAMPLE_PCT_START:-2.5}"
SAMPLE_PCT_STEP="${SAMPLE_PCT_STEP:-2.5}"
SAMPLE_PCT_END="${SAMPLE_PCT_END:-15}"

# Match the original Criteo experiment's complete algorithm list.
ALGO_LIST=(
    dast
    causal_forest
    mst
    clr
    kmeans
    gmm
    t_learner
    s_learner
    x_learner
    dr_learner
    policytree
)

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

OUTDIR="${EXP_ROOT}/${DATASET}/${TARGET}/${MU_MODEL_TYPE}/sample_frac_with_fixed_${PILOT_FRAC}_pilot_frac"
mkdir -p "${OUTDIR}"
echo "[INFO] python=$(command -v "${PYTHON_BIN}")"
echo "[INFO] outdir=${OUTDIR}"
echo "[INFO] sample_pct=${SAMPLE_PCT_START}:${SAMPLE_PCT_STEP}:${SAMPLE_PCT_END}  pilot_frac=${PILOT_FRAC}  seed=${SEED_SEQUENCE}  N_sim=${N_SIM}"
CRITEO_ARGS=()
if [[ "${DATASET}" == "criteo" ]]; then
    if [[ "${CRITEO_SOURCE}" != "local" && "${CRITEO_SOURCE}" != "sklift" ]]; then
        echo "[ERROR] CRITEO_SOURCE must be local or sklift"
        exit 1
    fi
    CRITEO_ARGS=(--criteo_source "${CRITEO_SOURCE}")
    if [[ "${CRITEO_SOURCE}" == "local" ]]; then
        if [[ ! -f "${CRITEO_CSV_PATH}" ]]; then
            echo "[ERROR] Local Criteo file not found: ${CRITEO_CSV_PATH}"
            exit 1
        fi
        CRITEO_ARGS+=(--criteo_csv_path "${CRITEO_CSV_PATH}")
    fi
    echo "[INFO] criteo_source=${CRITEO_SOURCE}"
fi

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
        if [[ "${DATASET}" == "criteo" ]]; then
            "${PYTHON_BIN}" - "${outpath}" "${CRITEO_SOURCE}" "${CRITEO_CSV_PATH}" <<'PY'
import gzip
import pickle
import sys
from pathlib import Path

path, source, csv_path = sys.argv[1:]
try:
    with gzip.open(path, "rb") as stream:
        saved = pickle.load(stream)["params"]
except gzip.BadGzipFile:
    with open(path, "rb") as stream:
        saved = pickle.load(stream)["params"]
matches = saved.get("criteo_source", "sklift") == source
if source == "local":
    matches = matches and saved.get("criteo_csv_path") == str(Path(csv_path).expanduser().resolve())
    matches = matches and "criteo_subset_seed" not in saved
if not matches:
    raise SystemExit(f"[ERROR] {path} uses a different Criteo source/subset. Set a new EXP_ROOT.")
PY
        fi
        echo "[SKIP] ${outpath} already exists"
        continue
    fi

    echo "[RUN ] sample_frac=${sample_frac} -> ${outpath}"
    "${PYTHON_BIN}" -u run_sims.py \
        --mu_model_type "${MU_MODEL_TYPE}" \
        --meta_learner_mu_model_type "${META_LEARNER_MU_MODEL_TYPE}" \
        --value_type_dast "${VALUE_TYPE_DAST}" \
        --value_type_dams "${VALUE_TYPE_DAMS}" \
        --seed_sequence "${SEED_SEQUENCE}" \
        --dataset "${DATASET}" \
        ${CRITEO_ARGS[@]+"${CRITEO_ARGS[@]}"} \
        --target "${TARGET}" \
        --sample_frac "${sample_frac}" \
        --pilot_frac "${PILOT_FRAC}" \
        --treatment_cost "${TREATMENT_COST}" \
        --N_sim "${N_SIM}" \
        --n_folds_dams "${N_FOLDS_DAMS}" \
        --algo_list "${ALGO_LIST[@]}" \
        --outpath "${outpath}"
done
