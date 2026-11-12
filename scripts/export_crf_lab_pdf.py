"""Write a shareable PDF of the CRF teaching lab (no Streamlit required)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_fraud.crf import (  # noqa: E402
    CRF_TRAINING_EXAMPLES,
    START,
    TAG_GLOSSARY,
    demo_dog_barks_model,
    triplet_key,
)
from credit_fraud.hmm import PosTaggingHMM, ToyHMM  # noqa: E402

OUT = ROOT / "docs" / "crf_lab.pdf"

NAVY = "#1F4E6A"
ORANGE = "#C45C26"
GREEN = "#3D6B54"
GRAY = "#555555"
TEAL = "#2F6F8F"
FOOTER = "CRF lab · HMM → Viterbi → linear-chain CRF"


def _new_page(title: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.07, 0.84, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.08, 0.035, FOOTER, fontsize=8, color="#888888")
    fig.text(0.92, 0.035, "for sharing", fontsize=8, color="#888888", ha="right")
    if title:
        ax.text(0, 0.97, title, fontsize=16, fontweight="bold", color=NAVY, va="top")
        ax.plot([0, 1], [0.925, 0.925], color=ORANGE, lw=1.4)
    return fig, ax


def _cover(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.add_patch(FancyBboxPatch((0, 0.62), 1, 0.28, boxstyle="round,pad=0.02", fc=NAVY, ec="none"))
    ax.text(0.05, 0.82, "CRF lab", fontsize=28, color="white", fontweight="bold", va="center")
    ax.text(
        0.05,
        0.70,
        "From moods & shirts to linear-chain Conditional Random Fields",
        fontsize=11,
        color="white",
        va="center",
    )
    ax.text(
        0,
        0.52,
        "What this note is\n"
        "A walk-through of the Streamlit CRF lab: Hidden Markov Models, the Viterbi\n"
        "algorithm, how CRFs differ from HMMs, and the teaching TripletCRF code\n"
        "(path score, forward Z(x), Viterbi decode, gradient step).",
        fontsize=11,
        color=GRAY,
        va="top",
    )
    ax.text(
        0,
        0.30,
        "Task in one line\n"
        "Infer the best tag (or mood) sequence given observations — first with a\n"
        "generative HMM, then with a discriminative linear-chain CRF on “the dog barks”.",
        fontsize=11,
        color=GRAY,
        va="top",
    )
    ax.text(
        0,
        0.12,
        "Code: src/credit_fraud/crf.py · hmm.py · crf_plots.py\n"
        "App:  streamlit run app/streamlit_app.py  →  CRF lab\n"
        "Notebook: src/credit_fraud/crf.ipynb",
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
        ("1", "Hidden Markov model", "Happy/Sad moods + shirt colors. Joint P(x,y). Enumerate all paths."),
        ("2", "Viterbi algorithm", "O(N·K²) DP finds the best path. Mood demo + POS “fans watch race”."),
        ("3", "Conditional random fields", "Discriminative P(y|x). Features, Z(x) forward, Viterbi, training."),
        ("4", "CRF code walkthrough", "TripletCRF: path_score, forward, probability, viterbi, gradient step."),
    ]
    y = 0.88
    for n, title, body in rows:
        ax.add_patch(FancyBboxPatch((0, y - 0.07), 0.08, 0.065, boxstyle="round,pad=0.008", fc=TEAL, ec="none"))
        ax.text(0.04, y - 0.037, n, color="white", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(0.12, y - 0.018, title, fontsize=11, fontweight="bold", color=NAVY, va="top")
        ax.text(0.12, y - 0.042, body, fontsize=9, color=GRAY, va="top")
        y -= 0.10

    ax.text(0, 0.42, "Further watching (from the app)", fontsize=12, fontweight="bold", color=NAVY)
    ax.plot([0, 1], [0.395, 0.395], color=ORANGE, lw=1.2)
    videos = [
        "1. Hidden Markov Model — youtube.com/watch?v=fX5bYmnHqqE",
        "2. The Viterbi Algorithm — youtube.com/watch?v=IqXdjdOgXPM",
        "3. Conditional Random Fields — youtube.com/watch?v=rI3DQS0P2fk",
        "4. Implementing a basic CRF in Python — youtube.com/watch?v=XesdKt4J6Zc",
    ]
    y = 0.36
    for line in videos:
        ax.text(0, y, line, fontsize=9, color=GRAY, va="top", family="monospace")
        y -= 0.05
    pdf.savefig(fig)
    plt.close(fig)


def _hmm(pdf: PdfPages) -> None:
    hmm = ToyHMM.happy_sad()
    obs = ToyHMM.sample_demo_sequence()
    path, _ = hmm.viterbi(obs)
    all_paths = hmm.enumerate_all_paths(obs)
    best_joint = float(all_paths.iloc[0]["joint probability"])

    fig, ax = _new_page("1. Hidden Markov model")
    ax.text(
        0,
        0.88,
        "Hidden moods (Happy / Sad) generate visible shirt colors (red / green / blue).\n"
        f"Demo shirts: {' → '.join(obs)}. Best mood path: {' → '.join(path)} (joint ≈ {best_joint:.4f}).",
        fontsize=10,
        color=GRAY,
        va="top",
    )

    # Compact transition + emission tables
    ax.text(0, 0.72, "Start / transition / emission", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.67,
        "START: P(Happy)=0.4, P(Sad)=0.6\n\n"
        "Transition P(mood_t | mood_{t-1}):\n"
        "              Happy   Sad\n"
        "  from Happy   0.7    0.3\n"
        "  from Sad     0.5    0.5\n\n"
        "Emission P(shirt | mood):\n"
        "           red   green  blue\n"
        "  Happy    0.8    0.1    0.1\n"
        "  Sad      0.2    0.3    0.5",
        fontsize=9,
        color=GRAY,
        va="top",
        family="monospace",
    )

    ax.text(0.52, 0.72, "Joint probability", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        0.52,
        0.67,
        "P(x, y) = P(y₁)·P(x₁|y₁)\n"
        "  × ∏_{t=2}^N P(y_t|y_{t-1})·P(x_t|y_t)\n\n"
        "Worked path Sad→Sad→Happy\n"
        "with green→blue→red:\n\n"
        "0.6×0.3 × 0.5×0.5 × 0.5×0.8\n"
        f"= {0.6 * 0.3 * 0.5 * 0.5 * 0.5 * 0.8:.4f}\n\n"
        f"2 moods × 3 days = {len(all_paths)} paths.\n"
        "Brute force works here; longer\n"
        "sequences need Viterbi.",
        fontsize=9,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _viterbi(pdf: PdfPages) -> None:
    hmm = ToyHMM.happy_sad()
    obs = ToyHMM.sample_demo_sequence()
    mood_path, _ = hmm.viterbi(obs)
    pos = PosTaggingHMM.fans_watch_race()
    pos_path, _ = pos.viterbi()

    fig, ax = _new_page("2. Viterbi algorithm")
    ax.text(
        0,
        0.88,
        "Brute force is O(K^N). Viterbi keeps only the best partial path into each state\n"
        "at each step → O(N·K²). Same idea for moods and for POS tags.",
        fontsize=10,
        color=GRAY,
        va="top",
    )

    boxes = [
        ("At each step", "K destinations × K\npredecessors = K² ops"),
        ("Whole sequence", "N tokens × K²\n= O(N·K²)"),
        ("POS example", "N=20, K=45 → 40,500\nvs 45²⁰ ≈ 10³³"),
    ]
    for i, (title, body) in enumerate(boxes):
        x = i * 0.34
        ax.add_patch(FancyBboxPatch((x, 0.62), 0.31, 0.16, boxstyle="round,pad=0.015", fc="#F4F7F8", ec=TEAL, lw=1))
        ax.text(x + 0.015, 0.74, title, fontsize=10, fontweight="bold", color=NAVY, va="top")
        ax.text(x + 0.015, 0.69, body, fontsize=8, color=GRAY, va="top")

    ax.text(0, 0.55, "Mood demo (shirts green → blue → red)", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.50,
        f"Winning mood path: {' → '.join(mood_path)}\n"
        "Day 1: prior × emission. Later days: for each target mood, keep the best previous\n"
        "mood and drop the rest (pruning). Backtrace the thick path through the trellis.",
        fontsize=9,
        color=GRAY,
        va="top",
    )

    ax.text(0, 0.36, "POS demo — fans watch the race", fontsize=12, fontweight="bold", color=NAVY)
    tags = [
        f"{tag}: {info['full_name']} — {info['means'].replace('**', '')}"
        for tag, info in TAG_GLOSSARY.items()
    ]
    ax.text(0, 0.31, "\n".join(tags), fontsize=9, color=GRAY, va="top")
    ax.text(
        0,
        0.16,
        f"Sentence: {' '.join(pos.sentence)}\n"
        f"Winning tags: {' → '.join(pos_path)}\n"
        "Zero emission (e.g. P(\"the\"|NOUN)=0) → log = −∞ → pruned immediately.",
        fontsize=9,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _crf_contrast(pdf: PdfPages) -> None:
    fig, ax = _new_page("3. HMM vs CRF")
    ax.text(
        0,
        0.88,
        "HMM tells a generative story (pick mood, then shirt). CRF sees the whole sentence\n"
        "and scores tag sequences with learned weights — it classifies, it does not generate.",
        fontsize=10,
        color=GRAY,
        va="top",
    )

    # Two columns
    ax.add_patch(FancyBboxPatch((0, 0.42), 0.47, 0.38, boxstyle="round,pad=0.02", fc="#EEF4F7", ec="none"))
    ax.add_patch(FancyBboxPatch((0.53, 0.42), 0.47, 0.38, boxstyle="round,pad=0.02", fc="#F7F0EB", ec="none"))
    ax.text(0.235, 0.76, "HMM (generative)", ha="center", fontsize=12, fontweight="bold", color=TEAL)
    ax.text(0.765, 0.76, "CRF (discriminative)", ha="center", fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(
        0.03,
        0.70,
        "Models joint P(x, y)\n"
        "Fixed CPTs (counts / MLE)\n"
        "Local: y_t depends on x_t\n"
        "Emission + transition tables\n"
        "Good for “story” intuition",
        fontsize=9,
        color=GRAY,
        va="top",
    )
    ax.text(
        0.56,
        0.70,
        "Models conditional P(y | x)\n"
        "Learned weights w on features\n"
        "Global: features can see x\n"
        "Maximize log P(y*|x)\n"
        "Standard for sequence tagging",
        fontsize=9,
        color=GRAY,
        va="top",
    )

    ax.text(0, 0.35, "Linear-chain CRF scoring", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.30,
        "P(y | x) = (1 / Z(x)) · exp( Σ_t Σ_j w_j f_j(y_{t−1}, y_t, x, t) )\n\n"
        "• Numerator: unnormalized score of one tag path.\n"
        "• Z(x): sum of scores over all K^N tag paths (partition function).\n"
        "• Feature f_j: e.g. I(prev=DET, curr=NOUN, word=\"dog\").\n"
        "• Weight w_j: positive rewards the pattern; negative suppresses it.",
        fontsize=9,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _crf_math(pdf: PdfPages) -> None:
    model = demo_dog_barks_model()
    sentence = ["the", "dog", "barks"]
    gold = ["DET", "NOUN", "VERB"]
    path_score = model.path_score(sentence, gold)
    z = model.partition(sentence)
    prob = model.probability(sentence, gold)
    path_pred, _ = model.viterbi(sentence)

    fig, ax = _new_page("3. Forward Z(x) · Viterbi · training")
    ax.text(
        0,
        0.88,
        f"Running example: “{' '.join(sentence)}” · gold {' → '.join(gold)}\n"
        f"path score = {path_score:.0f} · Z(x) = {z:.0f} · P(gold|x) ≈ {prob:.1%} · "
        f"Viterbi → {' → '.join(path_pred)}",
        fontsize=9,
        color=GRAY,
        va="top",
    )

    ax.text(0, 0.76, "Forward DP for Z(x) — O(N·K²)", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.71,
        "α(1,k) = ψ(START, k, x₁)\n"
        "α(t,k) = Σ_j α(t−1,j) · ψ(j, k, x_t)     (log-space: logsumexp)\n"
        "Z(x) = Σ_k α(N,k)\n"
        "Push sums inside products so you never enumerate K^N paths.",
        fontsize=9,
        color=GRAY,
        va="top",
        family="monospace",
    )

    ax.text(0, 0.52, "Viterbi decode — same DP, max instead of sum", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.47,
        "V(t,k) = max_j [ V(t−1,j) + w_(j,k,x_t) ]\n"
        "Store backpointers; terminate with argmax_k V(N,k); trace back.",
        fontsize=9,
        color=GRAY,
        va="top",
        family="monospace",
    )

    ax.text(0, 0.36, "Training — NLL and feature expectations", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(
        0,
        0.31,
        "L(w) = log Z(x;w) − score(x, y*; w)\n"
        "∂L/∂w_j = E_{y~P(y|x)}[f_j] − f_j(x, y*)\n"
        "If the model under-uses a gold feature → increase w_j; if over-uses → decrease.",
        fontsize=9,
        color=GRAY,
        va="top",
        family="monospace",
    )

    ax.text(0, 0.16, "Sample supervised sentences (lab corpus)", fontsize=11, fontweight="bold", color=NAVY)
    y = 0.12
    for row in CRF_TRAINING_EXAMPLES:
        ax.text(0, y, f"• {row['sentence']:16}  {row['gold tags']}", fontsize=8, color=GRAY, va="top", family="monospace")
        y -= 0.035
    pdf.savefig(fig)
    plt.close(fig)


def _features(pdf: PdfPages) -> None:
    model = demo_dog_barks_model()
    fig, ax = _new_page("3. Feature templates (the dog barks)")
    ax.text(
        0,
        0.88,
        "Triplet features are (previous tag, current tag, word). Potential ψ = exp(w).\n"
        "Gold path multiplies the three strong triplets: 2 × 3 × 2 = 12.",
        fontsize=10,
        color=GRAY,
        va="top",
    )

    rows = [
        ("(START, DET, \"the\")", "Sentence starts with determiner the", "high +"),
        ("(DET, NOUN, \"dog\")", "NOUN after DET on dog", "high +"),
        ("(NOUN, VERB, \"barks\")", "VERB after NOUN on barks", "high +"),
        ("(DET, DET, \"dog\")", "Two determiners in a row", "negative"),
    ]
    y = 0.72
    for feat, meaning, role in rows:
        ax.add_patch(FancyBboxPatch((0, y - 0.08), 1.0, 0.075, boxstyle="round,pad=0.01", fc="#F4F7F8", ec="none"))
        ax.text(0.02, y - 0.02, feat, fontsize=9, fontweight="bold", color=TEAL, va="top", family="monospace")
        ax.text(0.02, y - 0.045, meaning, fontsize=8, color=GRAY, va="top")
        ax.text(0.85, y - 0.035, role, fontsize=9, color=ORANGE, va="center", ha="left")
        y -= 0.10

    pot_rows = [
        (START, "DET", "the"),
        ("DET", "NOUN", "dog"),
        ("NOUN", "VERB", "barks"),
    ]
    ax.text(0, 0.28, "Gold-path potentials in the demo model", fontsize=11, fontweight="bold", color=NAVY)
    y = 0.23
    for prev, tag, word in pot_rows:
        w = model.potentials.get(triplet_key(prev, tag, word), 0.0)
        psi = model.potential(prev, tag, word)
        ax.text(
            0,
            y,
            f"({prev}, {tag}, '{word}')   w = log({psi:.0f}) ≈ {w:.3f}   ψ = {psi:.1f}",
            fontsize=9,
            color=GRAY,
            va="top",
            family="monospace",
        )
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


def _code(pdf: PdfPages) -> None:
    model = demo_dog_barks_model()
    sentence = ["the", "dog", "barks"]
    gold = ["DET", "NOUN", "VERB"]
    path_score = model.path_score(sentence, gold)
    z = model.partition(sentence)
    prob = model.probability(sentence, gold)
    grad = model.training_gradient_step(sentence, gold)
    path_pred, _ = model.viterbi(sentence)

    fig, ax = _new_page("4. CRF code walkthrough")
    ax.text(
        0,
        0.88,
        "Class TripletCRF in src/credit_fraud/crf.py — five methods you care about:",
        fontsize=10,
        color=GRAY,
        va="top",
    )

    pieces = [
        ("path_score", "Multiply ψ = exp(w) along one path → numerator."),
        ("_forward_log / partition", "Log-space DP → Z(x) in O(N·K²)."),
        ("probability", "exp(log score − log Z) = score / Z(x)."),
        ("viterbi", "Max + backpointers → best tag sequence."),
        ("training_gradient_step", "∇L = E[f] − f* ; optional weight update."),
    ]
    y = 0.78
    for name, does in pieces:
        ax.text(0, y, name, fontsize=9, fontweight="bold", color=TEAL, va="top", family="monospace")
        ax.text(0.42, y, does, fontsize=9, color=GRAY, va="top")
        y -= 0.055

    ax.text(0, 0.48, "Numbers for “the dog barks”", fontsize=12, fontweight="bold", color=NAVY)
    ax.plot([0, 1], [0.455, 0.455], color=ORANGE, lw=1.2)
    ax.text(
        0,
        0.42,
        f"Numerator (DET→NOUN→VERB) = 2 × 3 × 2 = {path_score:.0f}\n"
        f"Z(x) = {z:.0f}   →   P(gold|x) = {path_score:.0f}/{z:.0f} ≈ {prob:.1%}\n"
        f"Viterbi path = {' → '.join(path_pred)}\n"
        f"NLL loss = log Z − log score* ≈ {grad['loss']:.3f}   (p_gold ≈ {grad['p_gold']:.1%})",
        fontsize=9,
        color=GRAY,
        va="top",
        family="monospace",
    )

    ax.text(
        0,
        0.22,
        "Pipeline in the app\n"
        "1. Score one path   2. Forward → Z(x)   3. Probability\n"
        "4. Viterbi decode   5. One supervised gradient step\n\n"
        "Real systems use thousands of sentences and sklearn-crfsuite / pytorch-crf;\n"
        "this lab keeps a tiny hand-set potential table so every number is inspectable.",
        fontsize=9,
        color=GRAY,
        va="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _look(pdf: PdfPages) -> None:
    fig, ax = _new_page("What to click in the Streamlit app")
    bullets = [
        ("Step 1 — HMM", "Overview → matrices → joint formula → two paths → all 8 mood paths."),
        ("Step 2 — Viterbi", "Complexity visual, day-by-day mood pruning, then POS trellis for fans watch race."),
        ("Step 3 — CRF", "HMM vs CRF table, scoring formula, forward trellis, Viterbi vs forward, training flow."),
        ("Step 4 — Code", "Pipeline diagram, path-score table (27 paths), forward/Viterbi tables, gradient table, source."),
    ]
    y = 0.88
    for title, body in bullets:
        ax.add_patch(FancyBboxPatch((0, y - 0.12), 1.0, 0.11, boxstyle="round,pad=0.012", fc="#F4F7F8", ec="none"))
        ax.text(0.03, y - 0.03, title, fontsize=11, fontweight="bold", color=NAVY, va="top")
        ax.text(0.03, y - 0.07, body, fontsize=9, color=GRAY, va="top")
        y -= 0.14

    ax.text(
        0,
        0.28,
        "Takeaway\n"
        "HMM explains generation. Viterbi makes the best path cheap. CRF flips to\n"
        "P(y|x) with features and Z(x). The teaching code is the same DP ideas in Python.",
        fontsize=10,
        color=GRAY,
        va="top",
    )
    ax.text(
        0,
        0.12,
        "streamlit run app/streamlit_app.py  →  sidebar: CRF lab",
        fontsize=9,
        color=NAVY,
        va="top",
        family="monospace",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        _cover(pdf)
        _map(pdf)
        _hmm(pdf)
        _viterbi(pdf)
        _crf_contrast(pdf)
        _features(pdf)
        _crf_math(pdf)
        _code(pdf)
        _look(pdf)
        d = pdf.infodict()
        d["Title"] = "CRF lab — HMM, Viterbi, linear-chain CRF"
        d["Author"] = "Credit-fraud mentoring repo"
        d["Subject"] = "Teaching notes for the Streamlit CRF lab"
    return OUT


if __name__ == "__main__":
    path = main()
    print(path)
