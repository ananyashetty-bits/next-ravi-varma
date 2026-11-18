"""Chart of mean Shapley values with 95% CIs, one panel per metric (faithfulness and style).

python analysis/shap/plot_shap.py [--values analysis/shap/shap_values_dev.json] [--out docs/figures/shap_dev.png]
"""
import argparse
import json

ORDER = ["char", "action", "setting", "mood", "tail"]
INK, MUTED, BAR, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default="analysis/shap/shap_values_dev.json")
    ap.add_argument("--out", default="docs/figures/shap_dev.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open(a.values, encoding="utf-8"))
    panels = [(res["faithfulness_metric"], f"Faithfulness ({res['faithfulness_metric']})"), ("style_margin", "Style (style_margin)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, (m, title) in zip(axes, panels):
        phi = res["metrics"][m]["phi"]
        y = list(range(len(ORDER)))[::-1]
        means = [phi[p]["mean"] for p in ORDER]
        err = [[phi[p]["mean"] - phi[p]["lo"] for p in ORDER], [phi[p]["hi"] - phi[p]["mean"] for p in ORDER]]
        ax.set_facecolor(SURFACE)
        ax.barh(y, means, height=0.55, color=BAR)
        ax.errorbar(means, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, ORDER, color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_xlabel("mean Shapley value (95% CI over scenes)", color=MUTED, fontsize=9)
    fig.suptitle(f"Prompt players' contribution, {res['n_scenes']} dev scenes × 32 coalitions × 2 seeds",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
