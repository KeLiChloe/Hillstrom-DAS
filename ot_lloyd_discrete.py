# ot_lloyd_discrete.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from outcome_model import predict_mu
import heapq

def _inv_link(y_hat: np.ndarray, log_y: bool) -> np.ndarray:
    if not log_y:
        return y_hat
    return np.expm1(y_hat)


def build_mu_matrix(mu_models: Dict[int, object], X: np.ndarray, K: int, *, log_y: bool) -> np.ndarray:
    """
    mu[i,a] = E[Y | X_i, D=a] on ORIGINAL SCALE (invert log1p if needed).
    """
    X = np.asarray(X)
    mu = np.zeros((X.shape[0], K), dtype=float)
    for a in range(K):
        if a not in mu_models:
            raise KeyError(f"mu_models missing action={a}. keys={sorted(mu_models.keys())}")
        mu[:, a] = _inv_link(np.asarray(predict_mu(mu_models[a], X), dtype=float), log_y=log_y)
    return mu


def _make_capacities(n: int, L: int, q: Optional[np.ndarray] = None) -> np.ndarray:
    """
    capacities per segment (sum to n). default: equal.
    """
    if q is None:
        base = n // L
        cap = np.full(L, base, dtype=int)
        cap[: (n - base * L)] += 1
        return cap

    q = np.asarray(q, dtype=float).ravel()
    q = np.clip(q, 1e-12, 1.0)
    q = q / q.sum()
    cap = np.floor(q * n).astype(int)
    # distribute remainder
    rem = n - cap.sum()
    if rem > 0:
        frac = (q * n) - np.floor(q * n)
        add_idx = np.argsort(-frac)[:rem]
        cap[add_idx] += 1
    elif rem < 0:
        frac = (q * n) - np.floor(q * n)
        sub_idx = np.argsort(frac)[: (-rem)]
        cap[sub_idx] -= 1
    assert cap.sum() == n and np.all(cap >= 0)
    return cap




# ---- Min-Cost Max-Flow (SSAP with potentials) ----
class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")
    def __init__(self, to: int, rev: int, cap: int, cost: int):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost

def _add_edge(g: List[List[_Edge]], fr: int, to: int, cap: int, cost: int) -> None:
    g[fr].append(_Edge(to, len(g[to]), cap, cost))
    g[to].append(_Edge(fr, len(g[fr]) - 1, 0, -cost))

def _min_cost_flow(g: List[List[_Edge]], s: int, t: int, maxf: int) -> Tuple[int, int]:
    """
    Returns (flow, cost) with integer costs.
    """
    n = len(g)
    INF = 10**18
    flow = 0
    cost = 0
    pot = [0] * n  # potentials

    # If there are negative costs, pot=0 still works for this bipartite network
    # since we only push forward edges with non-negative reduced costs after first dijkstra.
    # For safety, you could run Bellman-Ford once; usually unnecessary here.

    while flow < maxf:
        dist = [INF] * n
        prev_v = [-1] * n
        prev_e = [-1] * n
        dist[s] = 0
        pq = [(0, s)]

        while pq:
            d, v = heapq.heappop(pq)
            if d != dist[v]:
                continue
            for ei, e in enumerate(g[v]):
                if e.cap <= 0:
                    continue
                nd = d + e.cost + pot[v] - pot[e.to]
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    prev_v[e.to] = v
                    prev_e[e.to] = ei
                    heapq.heappush(pq, (nd, e.to))

        if dist[t] == INF:
            break  # cannot send more flow

        # update potentials
        for v in range(n):
            if dist[v] < INF:
                pot[v] += dist[v]

        # add as much as possible on found path (here will be 1 each time, but keep generic)
        addf = maxf - flow
        v = t
        while v != s:
            pv = prev_v[v]
            pe = prev_e[v]
            if pv < 0:
                addf = 0
                break
            addf = min(addf, g[pv][pe].cap)
            v = pv
        if addf == 0:
            break

        v = t
        while v != s:
            pv = prev_v[v]
            pe = prev_e[v]
            e = g[pv][pe]
            e.cap -= addf
            g[v][e.rev].cap += addf
            v = pv

        flow += addf
        cost += addf * pot[t]  # shortest path cost w.r.t original costs

    return flow, cost


def balanced_assign_mincost(cost: np.ndarray, *, q: Optional[np.ndarray] = None, scale: int = 10**6) -> np.ndarray:
    """
    Exact capacitated assignment via min-cost flow:
      minimize sum_i cost[i, label_i] s.t. each segment l has cap[l] points.

    cost: (n, L) float
    q: segment weights, optional
    scale: float->int scaling factor. Increase if you need more precision.
    returns labels: (n,)
    """
    cost = np.asarray(cost, dtype=float)
    n, L = cost.shape

    # capacities (sum to n)
    cap = _make_capacities(n, L, q=q)

    # ---- Build bipartite min-cost flow graph ----
    # nodes: source (0), items (1..n), segments (1+n .. n+L), sink (n+L+1)
    S = 0
    item0 = 1
    seg0 = 1 + n
    T = 1 + n + L
    N = T + 1
    g: List[List[_Edge]] = [[] for _ in range(N)]

    # source -> each item (cap=1, cost=0)
    for i in range(n):
        _add_edge(g, S, item0 + i, 1, 0)

    # each segment -> sink (cap=cap[l], cost=0)
    for l in range(L):
        _add_edge(g, seg0 + l, T, int(cap[l]), 0)

    # item -> segment edges with costs
    # Make costs non-negative + convert to int to avoid float issues
    cmin = float(np.min(cost))
    shifted = cost - cmin  # >= 0
    int_cost = np.rint(shifted * scale).astype(np.int64)

    for i in range(n):
        ui = item0 + i
        row = int_cost[i]
        for l in range(L):
            _add_edge(g, ui, seg0 + l, 1, int(row[l]))

    flow, _ = _min_cost_flow(g, S, T, maxf=n)
    if flow != n:
        raise RuntimeError(f"Min-cost flow failed to assign all items: flow={flow}, n={n}")

    # ---- Read assignment: for each item node, find which item->segment edge was used (cap==0 on that forward edge) ----
    labels = np.empty(n, dtype=int)
    for i in range(n):
        ui = item0 + i
        assigned = -1
        for e in g[ui]:
            if seg0 <= e.to < seg0 + L:
                # forward edge had cap=1 initially; if used, cap becomes 0
                if e.cap == 0:
                    assigned = e.to - seg0
                    break
        if assigned < 0:
            raise RuntimeError("Could not decode assignment from flow.")
        labels[i] = assigned

    return labels


@dataclass
class DiscreteOTLloydSegmenter:
    """
    Segmenter defined by:
      - L segments
      - each segment l has an assigned action a_l
      - assignment uses regret-based cost + optional capacity balancing
    """
    mu_models: Dict[int, object]
    actions_per_segment: np.ndarray          # (L,) each in {0..K-1}
    K: int
    log_y: bool = False
    q: Optional[np.ndarray] = None           # segment weights (sum=1). default equal.
    use_balanced_ot: bool = True

    def assign(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        mu = build_mu_matrix(self.mu_models, X, self.K, log_y=self.log_y)   # (n,K)
        r_bar = np.max(mu, axis=1)                                         # (n,)

        L = int(self.actions_per_segment.shape[0])
        cost = np.zeros((X.shape[0], L), dtype=float)
        for l, a_l in enumerate(self.actions_per_segment):
            cost[:, l] = (r_bar - mu[:, int(a_l)]) ** 2

        if self.use_balanced_ot:
            return balanced_assign_mincost(cost, q=self.q)  # <- 精确 OT / min-cost flow
        else:
            return np.argmin(cost, axis=1).astype(int)



def fit_discrete_ot_lloyd(
    X: np.ndarray,
    mu_models: Dict[int, object],
    *,
    K: int,
    L: int,
    log_y: bool,
    seed: int = 0,
    max_iter: int = 50,
    use_balanced_ot: bool = True,
    q: Optional[np.ndarray] = None,
) -> Tuple[DiscreteOTLloydSegmenter, np.ndarray, np.ndarray]:
    """
    Discrete OT-Lloyd:
      repeat:
        1) assign by regret-cost (optionally balanced)
        2) update each segment's action = argmax_a sum_{i in seg} mu_i[a]
    """
    rng = np.random.default_rng(seed)

    # init segment actions (ensure coverage, then shuffle)
    init = np.tile(np.arange(K, dtype=int), int(np.ceil(L / K)))[:L].copy()
    rng.shuffle(init)
    actions = init

    seg = DiscreteOTLloydSegmenter(
        mu_models=mu_models,
        actions_per_segment=actions.copy(),
        K=K,
        log_y=log_y,
        q=q,
        use_balanced_ot=use_balanced_ot,
    )

    prev_actions = None
    prev_labels = None

    for _ in range(max_iter):
        # assignment
        labels = seg.assign(X)

        # update actions by profit-maximization (your “projection to discrete set”)
        mu = build_mu_matrix(mu_models, X, K, log_y=log_y)    # (n,K)
        r_bar = np.max(mu, axis=1)                            # (n,)

        # 先用当前 actions 做 assignment（你也可以把 seg.assign 改成接收 mu/r_bar 来避免重复计算）
        labels = seg.assign(X)

        new_actions = actions.copy()
        for l in range(L):
            idx = np.where(labels == l)[0]
            if idx.size == 0:
                new_actions[l] = int(rng.integers(0, K))
                continue
            # (|idx|, K) -> (K,)
            regret_sums = ((r_bar[idx, None] - mu[idx, :]) ** 2).sum(axis=0)
            new_actions[l] = int(np.argmin(regret_sums))


        # check convergence
        if prev_actions is not None:
            if np.array_equal(new_actions, prev_actions) and np.array_equal(labels, prev_labels):
                actions = new_actions
                break

        actions = new_actions
        seg.actions_per_segment = actions.copy()
        prev_actions = actions.copy()
        prev_labels = labels.copy()

    return seg, labels, actions


def run_discrete_ot_lloyd_model_select(
    X_train: np.ndarray,
    X_val: np.ndarray,
    D_val: np.ndarray,
    y_val: np.ndarray,
    mu_models: Dict[int, object],
    *,
    K: int,
    M_candidates: List[int],
    eval_fn,
    log_y: bool,
    seed: int = 0,
    max_iter: int = 50,
    use_balanced_ot: bool = True,
    q: Optional[np.ndarray] = None,
) -> Tuple[DiscreteOTLloydSegmenter, np.ndarray, int, np.ndarray]:
    """
    Choose best L in M_candidates by maximizing eval_fn on validation set.
    Then refit on X_train ∪ X_val (caller can pass full pilot as X_train if wanted).
    """
    best_score = -np.inf
    best_L = None
    best_actions = None
    best_seg = None

    for L in M_candidates:
        seg, _, actions = fit_discrete_ot_lloyd(
            X_train, mu_models, K=K, L=int(L), log_y=log_y,
            seed=seed, max_iter=max_iter, use_balanced_ot=use_balanced_ot, q=q
        )
        lab_val = seg.assign(X_val)

        score = eval_fn(
            X_val, D_val, y_val,
            lab_val,
            mu_models,
            actions,
            log_y=log_y,
        )["value_mean"]

        if float(score) > best_score:
            best_score = float(score)
            best_L = int(L)
            best_actions = actions.copy()
            best_seg = seg

    if best_L is None:
        raise RuntimeError("No valid L found in M_candidates for discrete OT-Lloyd.")

    return best_seg, best_seg.assign(X_train), best_L, best_actions
