"""Statistics helpers (Member B). Per-scene paired comparisons with bootstrap CIs."""
import numpy as np


def scene_means(rows, key, condition=None):
    """Average seeds (and raters) within each scene first. rows: list of dicts with scene_id and key."""
    d = {}
    for r in rows:
        if condition is not None and r.get("condition") != condition:
            continue
        if r.get(key) is None:
            continue
        d.setdefault(r["scene_id"], []).append(r[key])
    return {s: float(np.mean(v)) for s, v in d.items()}


def boot_ci(x, n=5000, seed=0):
    """95% bootstrap CI of the mean over scenes."""
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return tuple(float(v) for v in np.percentile(m, [2.5, 97.5]))


def paired(a, b, n=5000, seed=0, tie=0.0):
    """a, b: dict scene_id -> mean. Difference b - a on shared scenes, its CI, and win/tie/loss of b over a."""
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids], dtype=float)
    if len(d) == 0:
        return {"diff": float("nan"), "ci95": (float("nan"), float("nan")), "n": 0, "win": 0, "tie": 0, "loss": 0}
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return {"diff": float(d.mean()), "ci95": tuple(float(v) for v in np.percentile(m, [2.5, 97.5])), "n": len(ids),
            "win": int((d > tie).sum()), "tie": int((np.abs(d) <= tie).sum()), "loss": int((d < -tie).sum())}
