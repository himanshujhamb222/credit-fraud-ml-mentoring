"""Fixed four-review walkthrough numbers for the Naive Bayes lab."""

from __future__ import annotations

import math
from dataclasses import dataclass

from naive_bayes_lab.model import Model, fit, predict, word_prob

TEXTS = [
    "good movie",
    "great movie",
    "bad movie",
    "not good",
]
LABELS = [1, 1, 0, 0]

WORD_TABLE_ROWS = [
    ("good", 1, 1),
    ("movie", 2, 1),
    ("great", 1, 0),
    ("bad", 0, 1),
    ("not", 0, 1),
]


@dataclass(frozen=True)
class TrainingWalk:
    model: Model
    p_pos: float
    p_neg: float
    word_counts_pos: dict[str, int]
    word_counts_neg: dict[str, int]
    total_pos: int
    total_neg: int
    vocab_size: int
    vocab_sorted: list[str]


def training_walk() -> TrainingWalk:
    model = fit(TEXTS, LABELS)
    word_counts, total_words, doc_counts, vocab, n_docs = model
    return TrainingWalk(
        model=model,
        p_pos=doc_counts[1] / n_docs,
        p_neg=doc_counts[0] / n_docs,
        word_counts_pos=dict(word_counts[1]),
        word_counts_neg=dict(word_counts[0]),
        total_pos=int(total_words[1]),
        total_neg=int(total_words[0]),
        vocab_size=len(vocab),
        vocab_sorted=sorted(vocab),
    )


@dataclass(frozen=True)
class WordStep:
    word: str
    count_pos: int
    count_neg: int
    p_pos: float
    p_neg: float
    log_ratio: float
    score_after: float
    lines: list[str]


@dataclass(frozen=True)
class PredictTrace:
    text: str
    words: list[str]
    prior_score: float
    prior_lines: list[str]
    steps: list[WordStep]
    label: str
    score: float
    summary_lines: list[str]


def predict_trace(text: str = "good movie", alpha: float = 1.0) -> PredictTrace:
    walk = training_walk()
    model = walk.model
    word_counts, total_words, doc_counts, vocab, n_docs = model
    words = text.lower().split()
    v = len(vocab)

    prior_score = math.log(doc_counts[1] / n_docs) - math.log(doc_counts[0] / n_docs)
    prior_lines = [
        f"P(+) = {doc_counts[1]}/{n_docs} = {walk.p_pos:.1f}",
        f"P(-) = {doc_counts[0]}/{n_docs} = {walk.p_neg:.1f}",
        f"prior log-odds = log(P+/P-) = log({walk.p_pos:.1f}/{walk.p_neg:.1f}) = {prior_score:.3f}",
    ]

    score = prior_score
    steps: list[WordStep] = []
    for word in words:
        if word not in vocab:
            continue
        c_pos = word_counts[1][word]
        c_neg = word_counts[0][word]
        p_pos = word_prob(word, 1, model, alpha)
        p_neg = word_prob(word, 0, model, alpha)
        ratio = p_pos / p_neg
        log_ratio = math.log(ratio)
        score += log_ratio
        steps.append(
            WordStep(
                word=word,
                count_pos=c_pos,
                count_neg=c_neg,
                p_pos=p_pos,
                p_neg=p_neg,
                log_ratio=log_ratio,
                score_after=score,
                lines=[
                    f"P({word}|+) = ({c_pos}+{alpha:g}) / ({total_words[1]}+{alpha:g}·{v}) = {p_pos:.4f}",
                    f"P({word}|-) = ({c_neg}+{alpha:g}) / ({total_words[0]}+{alpha:g}·{v}) = {p_neg:.4f}",
                    f"log(P+/P-) = log({p_pos:.4f}/{p_neg:.4f}) = {log_ratio:.4f}",
                    f"running score → {score:.4f}",
                ],
            )
        )

    label, final_score = predict(text, model, alpha)
    summary_lines = [
        f"final score = {final_score:.4f}",
        f"prediction = {label}",
        "score > 0 → positive is more likely",
        "score < 0 → negative is more likely",
        "score = 0 → tie",
    ]
    return PredictTrace(
        text=text,
        words=words,
        prior_score=prior_score,
        prior_lines=prior_lines,
        steps=steps,
        label=label,
        score=final_score,
        summary_lines=summary_lines,
    )


def smoothed_example(word: str = "good", label: int = 1, alpha: float = 1.0) -> list[str]:
    walk = training_walk()
    model = walk.model
    word_counts, total_words, _, vocab, _ = model
    v = len(vocab)
    count = word_counts[label][word]
    total = total_words[label]
    p = word_prob(word, label, model, alpha)
    side = "+" if label == 1 else "-"
    return [
        f"P({word}|{side}) = (count + α) / (totalWords + αV)",
        f"         = ({count} + {alpha:g}) / ({total} + {alpha:g}·{v})",
        f"         = {count + alpha:g}/{total + alpha * v:g} = {p:.4f}",
    ]
