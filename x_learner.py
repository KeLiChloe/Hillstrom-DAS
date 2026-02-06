# x_learner.py
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor


# =========================================================
# Utils
# =========================================================

def _inv_link(y_hat, log_y: bool):
    """If outcome model was trained on log1p(y), invert back to original scale."""
    if not log_y:
        return y_hat
    return np.expm1(y_hat)


def _predict_mu_action(mu_models: dict, X: np.ndarray, a: int, *, log_y: bool) -> np.ndarray:
    """Predict mu_a(x) on ORIGINAL scale (invert log if needed)."""
    if a not in mu_models:
        raise ValueError(f"mu_models does not contain action {a}. keys={sorted(mu_models.keys())}")
    yhat = mu_models[a].predict(X)
    return _inv_link(np.asarray(yhat, dtype=float), log_y=log_y)


def _get_gate_vector(gate_obj, n: int, X: np.ndarray) -> np.ndarray:
    """
    gate_obj can be:
      - float/int: constant gating probability
      - sklearn classifier with predict_proba
    """
    if isinstance(gate_obj, (float, int, np.floating)):
        g = np.full(n, float(gate_obj), dtype=float)
    else:
        g = gate_obj.predict_proba(X)[:, 1].astype(float)
    # numerical safety
    return np.clip(g, 1e-6, 1.0 - 1e-6)


# =========================================================
# Fit
# =========================================================

def fit_x_learner(
    X_pilot: np.ndarray,
    D_pilot: np.ndarray,
    y_pilot: np.ndarray,
    mu_pilot_models: dict,
    *,
    log_y: bool,
    control_action: int = 0,
    # effect models
    effect_model=None,
    # gating options
    random_state: int = 0,
):
    """
    Multi-action X-learner (one-vs-control).

    For each action a != control_action, restrict to pair {control_action, a}:
      - tau_a^(1): fit on treated (D=a), target d1 = y - mu_0(x)
      - tau_a^(0): fit on control (D=0), target d0 = mu_a(x) - y
      - gate g_a(x):
          * if use_rct_gate=True: constant g_a = pi_a / (pi_0 + pi_a)
            where pi_a is known RCT assignment prob (from propensities if provided, else empirical)
          * else: fit a classifier P(D=a | pair, X)

    Returns
    -------
    dict with keys:
      - control_action
      - actions (sorted)
      - tau_1[a], tau_0[a]  (effect regressors)
      - gate[a]             (float constant or classifier)
      - pi (optional)       (per-action probabilities used)
    """
    X_pilot = np.asarray(X_pilot)
    D_pilot = np.asarray(D_pilot).astype(int).ravel()
    y_pilot = np.asarray(y_pilot, dtype=float).ravel()

    actions = np.array(sorted(mu_pilot_models.keys()), dtype=int)
    if actions.size == 0:
        raise ValueError("mu_pilot_models is empty.")
    if control_action not in actions:
        raise ValueError(f"control_action={control_action} not in mu_pilot_models keys={actions.tolist()}")

    K = int(actions.max()) + 1

    # default effect model (can be swapped to LGBMRegressor outside)
    if effect_model is None:
        effect_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        )


    # ---- per-action assignment probabilities pi_a ----
    # priority: user-provided propensities > empirical from D_pilot
    pi = np.zeros(K, dtype=float)

    for a in actions:
        pi[a] = float((D_pilot == a).mean())
  
    pi = np.clip(pi, 1e-8, 1.0)

    # ---- precompute mu_0(x) on ALL pilot, original scale ----
    mu0_all = _predict_mu_action(mu_pilot_models, X_pilot, int(control_action), log_y=log_y)

    models = {
        "control_action": int(control_action),
        "actions": actions,
        "tau_1": {},   # tau_a^(1)
        "tau_0": {},   # tau_a^(0)
        "gate": {},    # float or classifier
        "pi": pi,      # store for reproducibility/debug
    }

    for a in actions:
        a = int(a)
        if a == int(control_action):
            continue

        # restrict to pair {control, a}
        mask_pair = (D_pilot == control_action) | (D_pilot == a)
        X_pair = X_pilot[mask_pair]
        D_pair = D_pilot[mask_pair]
        y_pair = y_pilot[mask_pair]

        # predictions on pair subset (original scale)
        mu0_pair = mu0_all[mask_pair]
        mua_pair = _predict_mu_action(mu_pilot_models, X_pair, a, log_y=log_y)

        # treated pseudo-outcome d1 = y - mu0(x)
        mask_t = (D_pair == a)
        X_t = X_pair[mask_t]
        y_t = y_pair[mask_t]
        mu0_t = mu0_pair[mask_t]
        d1 = (y_t - mu0_t)

        # control pseudo-outcome d0 = mu_a(x) - y
        mask_c = (D_pair == control_action)
        X_c = X_pair[mask_c]
        y_c = y_pair[mask_c]
        mua_c = mua_pair[mask_c]
        d0 = (mua_c - y_c)


        tau1 = clone(effect_model).fit(X_t, d1)
        tau0 = clone(effect_model).fit(X_c, d0)

        # gating
        denom = float(pi[control_action] + pi[a])
        g_const = float(pi[a] / denom) if denom > 0 else 0.5
        g_const = float(np.clip(g_const, 1e-6, 1.0 - 1e-6))
        gate = g_const
        
        models["tau_1"][a] = tau1
        models["tau_0"][a] = tau0
        models["gate"][a] = gate

    return models


# =========================================================
# Predict
# =========================================================

def predict_best_action_x_learner(
    x_learner_models: dict,
    X: np.ndarray,
    mu_pilot_models: dict,
    *,
    log_y: bool,
):
    """
    Predict best action for each x by maximizing:
      mu_hat_0(x) for control
      mu_hat_a(x) = mu_hat_0(x) + tau_hat_a(x) for a != control
    where tau_hat_a(x) = g(x)*tau0(x) + (1-g(x))*tau1(x)

    Returns
    -------
    a_hat : (n,) int
    mu_hat : (n, K) float  (filled with -inf for unavailable actions)
    """
    X = np.asarray(X)
    n = X.shape[0]

    control = int(x_learner_models["control_action"])
    actions = np.array(x_learner_models["actions"], dtype=int)
    K = int(actions.max()) + 1

    # base mu0
    mu0 = _predict_mu_action(mu_pilot_models, X, control, log_y=log_y)

    # mu_hat matrix
    mu_hat = np.full((n, K), -np.inf, dtype=float)
    mu_hat[:, control] = mu0

    for a in actions:
        a = int(a)
        if a == control:
            continue

        tau1 = x_learner_models["tau_1"][a]
        tau0 = x_learner_models["tau_0"][a]
        gate_obj = x_learner_models["gate"][a]

        g = _get_gate_vector(gate_obj, n=n, X=X)
        tau = g * np.asarray(tau0.predict(X), dtype=float) + (1.0 - g) * np.asarray(tau1.predict(X), dtype=float)

        mu_hat[:, a] = mu0 + tau

    a_hat = np.argmax(mu_hat, axis=1).astype(int)
    return a_hat, mu_hat

