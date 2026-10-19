"""Treatment / action integer-code alignment helpers.

Hillstrom historically had two labelings:
  - alphabetical: Mens=0, No E-Mail=1, Womens=2
  - canonical (current): No E-Mail=0, Mens=1, Womens=2

OPE scalars are encoding-invariant when (D, a) are labeled consistently at
eval time. STZ sidecars store factual D alongside per-algo actions; merging a
fresh algo trained under a *different* labeling into an older sidecar without
remapping actions makes Matched-Action look spuriously bad.

These helpers:
  1. learn a label permutation from two D arrays on the same rows
  2. remap actions through that permutation
  3. verify remapped actions reproduce a stored IPW OPE on the sidecar (D, y)
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable

import numpy as np


def empirical_propensities(D: np.ndarray, n_actions: int | None = None) -> np.ndarray:
    D = np.asarray(D, dtype=int)
    k = int(D.max()) + 1 if n_actions is None else int(n_actions)
    e = np.zeros(k, dtype=float)
    for a in np.unique(D):
        e[int(a)] = float((D == a).mean())
    return e


def ipw_value(D: np.ndarray, y: np.ndarray, actions: np.ndarray) -> float:
    """Simple IPW policy value with empirical propensities (matches evaluate_policy_ipw)."""
    D = np.asarray(D, dtype=int)
    y = np.asarray(y, dtype=float)
    a = np.asarray(actions, dtype=int)
    if len(D) != len(y) or len(D) != len(a):
        raise ValueError(
            f"length mismatch: len(D)={len(D)}, len(y)={len(y)}, len(a)={len(a)}"
        )
    e = empirical_propensities(D, n_actions=int(max(D.max(), a.max())) + 1)
    if np.any(e[a] <= 0):
        raise ValueError("propensity for some recommended action is zero")
    w = (D == a).astype(float) / e[a]
    return float((w * y).mean())


def learn_label_permutation(D_src: np.ndarray, D_dst: np.ndarray) -> np.ndarray:
    """Return perm where ``perm[src_label] == dst_label`` (row-aligned arrays).

    Requires a near-deterministic map: each src label maps to one dst label on
    ≥99% of rows. Raises if the map is not a bijection on the observed labels.
    """
    D_src = np.asarray(D_src, dtype=int)
    D_dst = np.asarray(D_dst, dtype=int)
    if len(D_src) != len(D_dst):
        raise ValueError(
            f"D_src/D_dst length mismatch: {len(D_src)} vs {len(D_dst)}"
        )
    k = int(max(D_src.max(), D_dst.max())) + 1
    perm = np.arange(k, dtype=int)
    used_dst: set[int] = set()
    for s in np.unique(D_src):
        mask = D_src == s
        vals, counts = np.unique(D_dst[mask], return_counts=True)
        frac = float(counts.max()) / float(mask.sum())
        if frac < 0.99:
            raise ValueError(
                f"non-deterministic label map for src={int(s)}: "
                f"{dict(zip(vals.tolist(), counts.tolist()))} (best frac={frac:.3f})"
            )
        d = int(vals[int(np.argmax(counts))])
        if d in used_dst:
            raise ValueError(
                f"label map not injective: src={int(s)} and another both → dst={d}"
            )
        used_dst.add(d)
        perm[int(s)] = d
    return perm


def apply_label_permutation(actions: np.ndarray, perm: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=int)
    perm = np.asarray(perm, dtype=int)
    if a.min() < 0 or a.max() >= len(perm):
        raise ValueError(
            f"action labels out of perm range [0, {len(perm)-1}]; "
            f"got [{a.min()}, {a.max()}]"
        )
    return perm[a].astype(a.dtype, copy=False)


def actions_match_ipw(
    D: np.ndarray,
    y: np.ndarray,
    actions: np.ndarray,
    target_ipw: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> bool:
    got = ipw_value(D, y, actions)
    return bool(np.isclose(got, float(target_ipw), atol=atol, rtol=rtol))


def align_actions_via_D_permutation(
    *,
    actions_src: np.ndarray,
    D_src: np.ndarray,
    D_dst: np.ndarray,
    y_dst: np.ndarray,
    target_ipw: float,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap ``actions_src`` (coded like ``D_src``) into ``D_dst``'s labeling.

    Returns ``(actions_aligned, perm)`` with ``perm[src] = dst``.
    Verifies IPW on ``(D_dst, y_dst, actions_aligned)`` matches ``target_ipw``.
    """
    perm = learn_label_permutation(D_src, D_dst)
    aligned = apply_label_permutation(actions_src, perm)
    got = ipw_value(D_dst, y_dst, aligned)
    if not np.isclose(got, float(target_ipw), atol=atol, rtol=rtol):
        raise RuntimeError(
            f"after D-permutation align, IPW={got:.10g} != target {target_ipw:.10g} "
            f"(perm={perm.tolist()})"
        )
    return aligned, perm


def repair_actions_by_ipw_search(
    D: np.ndarray,
    y: np.ndarray,
    actions: np.ndarray,
    target_ipw: float,
    *,
    labels: Iterable[int] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Try identity, then all permutations of ``labels``, to match ``target_ipw``.

    Returns ``(actions_fixed, perm_or_None)``. ``perm is None`` means identity
    already matched. Raises if no permutation works.
    """
    D = np.asarray(D, dtype=int)
    y = np.asarray(y, dtype=float)
    a = np.asarray(actions, dtype=int)
    if actions_match_ipw(D, y, a, target_ipw, atol=atol, rtol=rtol):
        return a, None

    if labels is None:
        labels = sorted(set(np.unique(D).tolist()) | set(np.unique(a).tolist()))
    labels = [int(x) for x in labels]
    k = max(max(labels), int(D.max()), int(a.max())) + 1

    for dest in permutations(labels):
        perm = np.arange(k, dtype=int)
        for s, d in zip(labels, dest):
            perm[s] = d
        # skip identity (already tried)
        if np.array_equal(perm[labels], labels):
            continue
        a2 = apply_label_permutation(a, perm)
        if actions_match_ipw(D, y, a2, target_ipw, atol=atol, rtol=rtol):
            return a2, perm

    raise RuntimeError(
        f"no label permutation reproduces target IPW={target_ipw:.10g}; "
        f"identity IPW={ipw_value(D, y, a):.10g}"
    )


def align_fresh_algo_actions_to_sidecar(
    stored_impl: dict,
    fresh_impl: dict,
    algo: str,
    target_ipw: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``fresh_impl['actions'][algo]`` onto ``stored_impl``'s D labeling.

    Requires identical ``customer_id`` order (as produced by
    ``_attach_sim_implementation``). Returns ``(aligned_actions, perm)``.
    """
    cid_s = np.asarray(stored_impl["customer_id"])
    cid_f = np.asarray(fresh_impl["customer_id"])
    if not np.array_equal(cid_s, cid_f):
        raise RuntimeError(
            f"customer_id mismatch when aligning {algo}: cannot safely remap actions"
        )
    D_s = np.asarray(stored_impl["D"])
    D_f = np.asarray(fresh_impl["D"])
    y_s = np.asarray(stored_impl["y"])
    a_f = np.asarray(fresh_impl["actions"][algo])

    if np.array_equal(D_s, D_f):
        if not actions_match_ipw(D_s, y_s, a_f, target_ipw, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"{algo}: D labels match but IPW "
                f"{ipw_value(D_s, y_s, a_f):.10g} != OPE {target_ipw:.10g}"
            )
        return a_f.astype(np.int8), np.arange(int(max(D_s.max(), a_f.max())) + 1)

    return align_actions_via_D_permutation(
        actions_src=a_f,
        D_src=D_f,
        D_dst=D_s,
        y_dst=y_s,
        target_ipw=target_ipw,
        atol=atol,
        rtol=rtol,
    )
