from itertools import combinations
from math import factorial
import json
import os

ORDER = ["char", "action", "setting", "mood", "tail"]

def shapley(v, players):
    """v: dict frozenset(coalition) -> value (mean over seeds). Exact Shapley values."""
    n, phi = len(players), {}
    for i in players:
        rest, s = [p for p in players if p != i], 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                S = frozenset(S)
                if S | {i} in v and S in v:
                    s += w * (v[S | {i}] - v[S])
        phi[i] = s
    return phi

def interaction(v, players, i, j):
    """Shapley interaction index for the pair (i, j): >0 complements, <0 substitutes."""
    n, rest, s = len(players), [p for p in players if p not in (i, j)], 0.0
    for k in range(n - 1):
        w = factorial(k) * factorial(n - k - 2) / factorial(n - 1)
        for S in combinations(rest, k):
            S = frozenset(S)
            if (S | {i, j} in v) and (S | {i} in v) and (S | {j} in v) and (S in v):
                s += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return s

def run_shap_analysis():
    metrics_path = "outputs/eval/shap_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}. Please evaluate first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    print("SHAP analysis script initialized successfully.")
    print(f"Loaded metrics from {metrics_path}")

if __name__ == "__main__":
    run_shap_analysis()