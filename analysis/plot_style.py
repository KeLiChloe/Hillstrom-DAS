"""Shared plot colors and display labels for Hillstrom-DAS figures."""

from __future__ import annotations

import matplotlib.colors as mcolors

# ------------------------------------------------------------------
# Colors (8-digit hex includes alpha)
# ------------------------------------------------------------------
DEFAULT_COLORS = {
    "kmeans-standard": "#FFD22FAF",
    "kmeans-dams": "#C9A000AF",
    "gmm-standard": "#006135AF",
    "gmm-dams": "#3D9B6AAF",
    "clr-standard": "#F59134AF",
    "clr-dams": "#C45C1AAF",
    "mst": "#937860AF",
    "dast_old": "#C2C2C2AE",
    "t_learner": "#6BC735AF",
    "s_learner": "#1F5BFFAF",
    "x_learner": "#FF5832AD",
    "dr_learner": "#7A5CFFAF",
    "policy_tree": "#333333AF",
    "causal_forest": "#0097A7AF",
}

# ------------------------------------------------------------------
# Display labels
# ------------------------------------------------------------------
LABEL_MAP = {
    "kmeans-standard": "K-Means",
    "kmeans-dams": "K-Means + DAMS",
    "gmm-standard": "GMM",
    "gmm-dams": "GMM + DAMS",
    "clr-standard": "CLR",
    "clr-dams": "CLR + DAMS",
    "mst": "MST",
    "dast_old": "DAST (old)",
    "t_learner": "T-Learner",
    "s_learner": "S-Learner",
    "x_learner": "X-Learner",
    "dr_learner": "DR-Learner",
    "policy_tree": "Policy Tree",
    "causal_forest": "Causal Forest",
}

# Pickle / run_sims.py keys -> DEFAULT_COLORS keys
BASELINE_KEY_ALIASES = {
    "kmeans": "kmeans-standard",
    "kmeans_dams": "kmeans-dams",
    "gmm": "gmm-standard",
    "gmm_dams": "gmm-dams",
    "clr": "clr-standard",
    "clr_dams": "clr-dams",
}

EXTRA_LABEL_MAP = {
    "random": "Random",
    "kmeans": "K-Means",
    "kmeans_dams": "K-Means + DAMS",
    "gmm": "GMM",
    "gmm_dams": "GMM + DAMS",
    "clr": "CLR",
    "clr_dams": "CLR + DAMS",
    "all_0": "All Action=0",
    "all_1": "All Action=1",
    "all_2": "All Action=2",
}

PREFERRED_ORDER = [
    "Random",
    "K-Means",
    "K-Means + DAMS",
    "GMM",
    "GMM + DAMS",
    "CLR",
    "CLR + DAMS",
    "MST",
    "Causal Forest",
    "T-Learner",
    "S-Learner",
    "X-Learner",
    "DR-Learner",
    "Policy Tree",
    "DAST (old)",
]

_FALLBACK_COLOR = "#333333AF"


def resolve_color_key(baseline_key: str) -> str:
    return BASELINE_KEY_ALIASES.get(baseline_key, baseline_key)


def baseline_label(baseline_key: str) -> str:
    if baseline_key in LABEL_MAP:
        return LABEL_MAP[baseline_key]
    resolved = resolve_color_key(baseline_key)
    if resolved in LABEL_MAP:
        return LABEL_MAP[resolved]
    if baseline_key in EXTRA_LABEL_MAP:
        return EXTRA_LABEL_MAP[baseline_key]
    return baseline_key.replace("_", " ").title()


def baseline_color(baseline_key: str, *, default: str = _FALLBACK_COLOR) -> str:
    resolved = resolve_color_key(baseline_key)
    if resolved in DEFAULT_COLORS:
        return DEFAULT_COLORS[resolved]
    if baseline_key in DEFAULT_COLORS:
        return DEFAULT_COLORS[baseline_key]
    if baseline_key == "random":
        return DEFAULT_COLORS["policy_tree"]
    return default


def vs_label(baseline_key: str) -> str:
    return "vs. " + baseline_label(baseline_key)


def comparator_colors_by_label() -> dict[str, str]:
    keys = set(LABEL_MAP) | set(BASELINE_KEY_ALIASES) | set(EXTRA_LABEL_MAP)
    return {baseline_label(k): baseline_color(k) for k in keys}


def to_rgba(hex_color: str, alpha: float | None = None) -> tuple:
    """Convert hex to RGBA; preserve alpha embedded in #RRGGBBAA."""
    if len(hex_color) in (9, 10) and hex_color.startswith("#"):
        return mcolors.to_rgba(hex_color)
    if alpha is None:
        return mcolors.to_rgba(hex_color)
    return mcolors.to_rgba(hex_color, alpha=alpha)
