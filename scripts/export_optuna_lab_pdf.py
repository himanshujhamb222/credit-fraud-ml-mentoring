"""Write a shareable PDF of the Optuna teaching lab (no Streamlit required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "optuna_lab.pdf"

NAVY = "#1F4E6A"
ORANGE = "#C45C26"
GREEN = "#3D6B54"
GRAY = "#555555"
TEAL = "#2F6F8F"


def _new_page(title: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.07, 0.84, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.08, 0.035, "Optuna lab · UCI credit default · XGBoost + TPE", fontsize=8, color="#888888")
    fig.text(0.92, 0.035, "for sharing", fontsize=8, color="#888888", ha="right")
    if title:
        ax.text(0, 0.97, title, fontsize=16, fontweight="bold", color=NAVY, va="top")
        ax.plot([0, 1], [0.925, 0.925], color=ORANGE, lw=1.4)
    return fig, ax


def _cover(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.add_patch(FancyBboxPatch((0, 0.62), 1, 0.28, boxstyle="round,pad=0.02", fc=NAVY, ec="none"))
    ax.text(0.05, 0.82, "Optuna lab", fontsize=28, color="white", fontweight="bold", va="center")
    ax.text(
        0.05,
        0.70,
        "Bayesian tuning of XGBoost on UCI Default of Credit Card Clients",
        fontsize=11,
        color="white",
        va="center",
    )
    ax.text(
        0,
        0.52,
        "What this note is\n"
        "A walk-through of the Streamlit Optuna lab: why we tune, grid vs random vs TPE,\n"
        "the TPE math, the three XGBoost knobs, why PR-AUC, and how the code is wired.",
        fontsize=11,
        color=GRAY,
        va="top",
    )
    ax.text(
        0,
        0.32,
        "Task in one line\n"
        "Features stay fixed. Optuna searches n_estimators, max_depth, and learning_rate\n"
        "to maximise stratified CV PR-AUC. Default rate ≈ 22%.",
        fontsize=11,
        color=GRAY,
        va="top",
    )
    ax.text(
        0,
        0.14,
        "Code: src/credit_fraud/optimize.py\n"
        "App:  streamlit run app/streamlit_app.py  →  Optuna lab\n"
        "Notebook: notebooks/02_optuna_bayesian.ipynb",
        fontsize=9,
        color=NAVY,
        va="top",
        family="monospace",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _map(pdf: PdfPages) -> None:
    fig, ax = _new_page("Session map")
    rows = [
        ("1", "Why we tune", "Features fixed; search XGB settings; score = CV PR-AUC."),
        ("2", "Grid vs random vs Bayesian", "Grid explodes. Random has no memory. TPE learns."),
        ("3", "The loop", "pick → train CV → store → pick next. First 3–5 TPE trials are random."),
        ("4", "TPE math", "Two densities ℓ (good) and g (bad). Next x maximises ℓ(x)/g(x)."),
        ("5", "XGB settings", "n_estimators 100–400, max_depth 2–8, learning_rate 0.01–0.2 (log)."),
        ("6", "Why PR-AUC", "Always-no-default accuracy ≈ 78% is useless. Random PR-AUC ≈ 0.22."),
        ("7", "Code", "xgb_search_space → make_xgboost → CV → study.optimize → refit winner."),
        ("8", "TPE vs random", "Same budget on a UCI sample. TPE should cluster near the best."),
        ("9", "Tune XGBoost", "Live search in the app. History, scatter, Optuna plots, hold-out."),
    ]
    y = 0.88
    for n, title, body in rows:
        ax.add_patch(FancyBboxPatch((0, y - 0.07), 0.08, 0.065, boxstyle="round,pad=0.008", fc=TEAL, ec="none"))
        ax.text(0.04, y - 0.037, n, color="white", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(0.12, y - 0.018, title, fontsize=11, fontweight="bold", color=NAVY, va="top")
        ax.text(0.12, y - 0.042, body, fontsize=9, color=GRAY, va="top")
        y -= 0.085
    pdf.savefig(fig)
    plt.close(fig)


def _why(pdf: PdfPages) -> None:
    fig, ax = _new_page("1. Why we tune")
    ax.text(
        0,
        0.88,
        "We are not changing the feature set. We change a few XGBoost settings (how deep\n"
        "each tree is, how many trees, how large each boosting step). Each try trains a model\n"
        "and scores mean CV PR-AUC. You cannot try every combo, so Optuna picks the next try\n"
        "from past scores.",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    labels = [
        ("Task", "~22% default\nnext month", ORANGE),
        ("Fixed", "engineered\nfeatures", "#777777"),
        ("Tuned", "depth · lr\n· trees", TEAL),
        ("Score", "CV\nPR-AUC", GREEN),
    ]
    for i, (title, body, color) in enumerate(labels):
        x = i * 0.25
        ax.add_patch(FancyBboxPatch((x, 0.48), 0.22, 0.22, boxstyle="round,pad=0.02", fc=color, ec="none"))
        ax.text(x + 0.11, 0.64, title, ha="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 0.11, 0.55, body, ha="center", va="center", color="white", fontsize=9)
        if i < 3:
            ax.annotate("", xy=(x + 0.24, 0.59), xytext=(x + 0.22, 0.59), arrowprops={"arrowstyle": "->", "color": "#aaaaaa"})
    ax.text(
        0,
        0.38,
        "scale_pos_weight is n_good / n_default (class imbalance). It is set from the labels,\n"
        "not searched. Other XGB knobs stay at make_xgboost defaults (200 trees, depth 4,\n"
        "lr 0.05, subsample/colsample 0.8, hist, eval_metric='aucpr').",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _search(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, "2. Grid vs random vs Bayesian", fontsize=16, fontweight="bold", color=NAVY)
    fig.lines.extend([])
    ax_line = fig.add_axes([0.08, 0.91, 0.84, 0.002])
    ax_line.axis("off")
    ax_line.plot([0, 1], [0.5, 0.5], color=ORANGE, lw=1.4, transform=ax_line.transAxes)

    ax = fig.add_axes([0.08, 0.55, 0.84, 0.32])
    methods = ["Grid", "Random", "Bayesian (TPE)"]
    tries = [100, 40, 25]
    quality = [55, 45, 85]
    colors = [ORANGE, "#888888", GREEN]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, tries, w, color=colors, label="relative trial budget")
    ax.bar(x + w / 2, quality, w, color=colors, alpha=0.45, label="relative success")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("relative (sketch)")
    ax.set_title("Grid explodes · random wanders · TPE learns from past trials", color=NAVY, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    notes = fig.add_axes([0.08, 0.08, 0.84, 0.42])
    notes.axis("off")
    notes.text(
        0,
        0.95,
        "Grid\n  Tries every combo. With 5–6 settings the budget explodes.\n\n"
        "Random\n  Draws uniformly. Simple, but every trial ignores what already worked.\n\n"
        "Bayesian (TPE)\n  Next try looks like high-scoring settings, with a little exploration.\n"
        "  First 3–5 trials are still random (startup). Then it concentrates.\n\n"
        "TPE does not guarantee the global best. It spends a small budget better than a grid.",
        fontsize=10,
        color=GRAY,
        va="top",
        family="sans-serif",
    )
    fig.text(0.08, 0.035, "Optuna lab · UCI credit default · XGBoost + TPE", fontsize=8, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def _loop(pdf: PdfPages) -> None:
    fig, ax = _new_page("3. The loop")
    ax.text(0, 0.88, "pick settings  →  train XGB (CV)  →  PR-AUC  →  store trial  →  pick next", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(
        0,
        0.78,
        "First 3–5 TPE trials are random. Then TPE splits past trials into high vs low PR-AUC\n"
        "and samples more from the high pile (with occasional draws from the low pile).",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    ax.text(0, 0.62, "4. TPE math (if asked)", fontsize=16, fontweight="bold", color=NAVY)
    ax.plot([0, 1], [0.575, 0.575], color=ORANGE, lw=1.4)
    ax.text(
        0,
        0.54,
        "Hyperparameters x = (learning rate, max depth, …). Score y = f(x) = CV PR-AUC.\n"
        "Example: x = (0.05, 5) might produce y = 0.84.\n\n"
        "Choose a cutoff y* that splits good vs bad trials (we maximise, so good means y > y*).\n"
        "Fit two densities over settings:\n"
        "    ℓ(x) = P(x | y > y*)     good trials\n"
        "    g(x) = P(x | y ≤ y*)     bad trials\n\n"
        "TPE wants an x that is likely under the good density and unlikely under the bad one:\n"
        "    x_next = arg max  ℓ(x) / g(x)\n\n"
        "Ordinary Bayesian optimisation models P(y | x): if I pick these settings, how good?\n"
        "TPE flips it and models P(x | y): given a good result, which settings were likely?",
        fontsize=10,
        color=GRAY,
        va="top",
        family="sans-serif",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _parzen(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, "TPE — two Parzen densities (sketch)", fontsize=16, fontweight="bold", color=NAVY)

    ax = fig.add_axes([0.12, 0.52, 0.76, 0.36])
    x = np.linspace(0.01, 0.35, 400)
    good = np.exp(-0.5 * ((x - 0.06) / 0.018) ** 2)
    bad = np.exp(-0.5 * ((x - 0.22) / 0.035) ** 2)
    good = good / good.max()
    bad = bad / bad.max() * 0.85
    ax.fill_between(x, good, color=TEAL, alpha=0.35, label="ℓ(x)  good  P(x | y > y*)")
    ax.plot(x, good, color=TEAL)
    ax.fill_between(x, bad, color=ORANGE, alpha=0.25, label="g(x)  bad  P(x | y ≤ y*)")
    ax.plot(x, bad, color=ORANGE)
    ax.axvline(0.05, color=TEAL, ls=":", lw=1.5)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("relative density")
    ax.set_title("Prefer ~0.05: high ℓ, low g", color=NAVY, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.38])
    ax2.axis("off")
    ax2.table(
        cellText=[
            ["0.05", "0.40", "0.05", "8.00"],
            ["0.10", "0.20", "0.15", "1.33"],
            ["0.20", "0.03", "0.30", "0.10"],
        ],
        colLabels=["learning rate", "ℓ(x) good", "g(x) bad", "ℓ(x) / g(x)"],
        loc="upper center",
        cellLoc="center",
    )
    ax2.text(
        0,
        0.35,
        "TPE therefore prefers something around x = 0.05 because ℓ/g = 8.\n"
        "That is the whole idea: sample more where past winners lived.",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    fig.text(0.08, 0.035, "Optuna lab · UCI credit default · XGBoost + TPE", fontsize=8, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def _knobs_prauc(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, "5–6. Settings we search · Why PR-AUC", fontsize=16, fontweight="bold", color=NAVY)

    ax_t = fig.add_axes([0.08, 0.68, 0.84, 0.22])
    ax_t.axis("off")
    ax_t.table(
        cellText=[
            ["n_estimators", "How many boosting rounds.", "100–400"],
            ["max_depth", "Depth of each small tree.", "2–8"],
            ["learning_rate", "Step size per new tree (log scale).", "0.01–0.2"],
        ],
        colLabels=["setting", "meaning", "range"],
        loc="center",
        cellLoc="left",
        colWidths=[0.22, 0.48, 0.22],
    )

    ax = fig.add_axes([0.12, 0.28, 0.76, 0.36])
    recall = np.linspace(0.0, 1.0, 80)
    pr = 0.22
    random_pr = np.full_like(recall, pr)
    good_pr = pr + (1.0 - pr) * (1.0 - (1.0 - recall) ** 1.4)
    good_pr = np.clip(good_pr, pr, 1.0)
    ax.fill_between(recall, good_pr, color=TEAL, alpha=0.18)
    ax.plot(recall, good_pr, color=TEAL, lw=2.5, label="good model (defaults near top)")
    ax.plot(recall, random_pr, color="#888888", ls="--", lw=2, label="random ranking ≈ 22%")
    ax.set_xlabel("recall (fraction of defaults caught)")
    ax.set_ylabel("precision at that cutoff")
    ax.set_ylim(0, 1.05)
    ax.set_title("Why PR-AUC on ~22% default rate", color=NAVY, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.08,
        0.12,
        "Never-predict-default accuracy ≈ 78%. Useless for this problem.\n"
        "Random ranking PR-AUC ≈ default rate ≈ 0.22. A model at 0.50–0.55 is more than 2× that.\n"
        "That is not a coin flip (coin-flip ROC-AUC would be 0.50). Optuna maximises PR-AUC.",
        fontsize=10,
        color=GRAY,
    )
    fig.text(0.08, 0.035, "Optuna lab · UCI credit default · XGBoost + TPE", fontsize=8, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def _code(pdf: PdfPages) -> None:
    fig, ax = _new_page("7. Code")
    ax.text(
        0,
        0.88,
        "xgb_search_space  →  make_xgboost (unfitted)  →  CV PR-AUC  →  study.optimize\n"
        "After the study: make_xgboost(**best_params).fit(...) on all train rows.",
        fontsize=10,
        color=NAVY,
        va="top",
    )
    pieces = [
        ("xgb_search_space", "Ask Optuna for trees, depth, learning rate."),
        ("make_xgboost", "Factory: defaults + trial params + scale_pos_weight. Does not fit."),
        ("build_objective", "One trial: those params → mean CV PR-AUC."),
        ("create_hpo_study", "Study that maximises PR-AUC (TPE or random)."),
        ("run_n_trials", "Repeat the objective N times."),
        ("optimize_xgboost", "All of the above, then refit the winner on all train rows."),
    ]
    y = 0.72
    for name, does in pieces:
        ax.text(0, y, name, fontsize=10, fontweight="bold", color=TEAL, va="top", family="monospace")
        ax.text(0.42, y, does, fontsize=10, color=GRAY, va="top")
        y -= 0.06
    ax.text(
        0,
        0.32,
        'suggest_int / suggest_float: “give me a value in this range and remember it.”\n'
        "log=True on learning_rate: 0.01 vs 0.1 matters more than 0.11 vs 0.12.\n\n"
        "direction='maximize'. TPE uses n_startup_trials random draws, then concentrates.\n"
        "Random sampler: every trial is still a uniform draw — no memory.",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _look(pdf: PdfPages) -> None:
    fig, ax = _new_page("8–9. What to look at in the app")
    ax.text(
        0,
        0.88,
        "Step 8 runs TPE and random with the same budget on a UCI sample (~2k rows,\n"
        "2-fold, 20 trials each in the demo). Step 9 is the live tuner.",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    bullets = [
        ("History chart", "Dashed line ≈ 0.22 (random ranking). Best-so-far should sit near 0.50+."),
        ("Scatter", "Star = best. Later TPE (orange) near the star. Random (×) stays spread out."),
        ("Table", "phase = early vs later. near_best = within 0.01 of the best score."),
        ("Hold-out", "Refit best settings on train; score PR-AUC and ROC-AUC on the held-out slice."),
        ("Optuna plots", "Need ≥3 finished trials: history, parallel coordinates, contour, slice, importances."),
    ]
    y = 0.72
    for title, body in bullets:
        ax.add_patch(FancyBboxPatch((0, y - 0.09), 1.0, 0.085, boxstyle="round,pad=0.012", fc="#F4F7F8", ec="none"))
        ax.text(0.03, y - 0.025, title, fontsize=11, fontweight="bold", color=NAVY, va="top")
        ax.text(0.03, y - 0.055, body, fontsize=9, color=GRAY, va="top")
        y -= 0.11
    ax.text(
        0,
        0.14,
        "TPE does not guarantee the global best. It spends a small budget better than a grid.",
        fontsize=10,
        color=ORANGE,
        va="top",
        fontweight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        _cover(pdf)
        _map(pdf)
        _why(pdf)
        _search(pdf)
        _loop(pdf)
        _parzen(pdf)
        _knobs_prauc(pdf)
        _code(pdf)
        _look(pdf)
        d = pdf.infodict()
        d["Title"] = "Optuna lab — XGBoost on UCI credit default"
        d["Author"] = "Credit-fraud mentoring repo"
        d["Subject"] = "Bayesian hyperparameter optimisation with Optuna (TPE)"
    return OUT


if __name__ == "__main__":
    path = main()
    print(path)
