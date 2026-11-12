"""Temperature, top-k, and top-p on the lesson's toy logits."""

from __future__ import annotations

import math

import pandas as pd

from transformer_lab.common import f4

VOCAB = ["the", "cat", "dog", "sat", "ran", "jumped", "on", "mat", "floor", "."]
LOGITS = [3.2, 1.1, 2.8, 0.4, 0.9, 1.5, -0.2, 2.1, 0.7, 0.1]


def _softmax(logits: list[float], temperature: float) -> tuple[list[float], list[float], float, list[float]]:
    adjusted = [value / temperature for value in logits]
    peak = max(adjusted)
    exps = [math.exp(value - peak) for value in adjusted]
    total = sum(exps)
    probs = [value / total for value in exps]
    return adjusted, exps, total, probs


def temperature_table(temperature: float) -> pd.DataFrame:
    adjusted, _, _, probs = _softmax(LOGITS, temperature)
    return pd.DataFrame(
        {
            "token": VOCAB,
            "logit": [f4(value) for value in LOGITS],
            "logit / T": [f4(value) for value in adjusted],
            "probability": [f4(value) for value in probs],
        }
    )


def _shown_exp(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.4e}"
    return f4(value)


def temperature_lines(temperature: float) -> list[str]:
    winner = max(range(len(LOGITS)), key=lambda index: LOGITS[index])
    adjusted = [value / temperature for value in LOGITS]
    raw_exps = [math.exp(value) for value in adjusted]
    total = sum(raw_exps)
    prob = raw_exps[winner] / total
    return [
        "P(word) = exp(logit / T) / sum exp(logit / T)",
        f"T = {f4(temperature)}",
        f"{VOCAB[winner]} has logit {f4(LOGITS[winner])}",
        f"{f4(LOGITS[winner])} / {f4(temperature)} = {f4(adjusted[winner])}",
        f"exp({f4(adjusted[winner])}) = {_shown_exp(raw_exps[winner])}",
        f"sum of exp = {_shown_exp(total)}",
        f"P({VOCAB[winner]}) = {_shown_exp(raw_exps[winner])} / {_shown_exp(total)} = {f4(prob)}",
        "T below 1 makes the winner heavier.",
        "T above 1 flattens the list.",
        "T near 0 picks the largest logit.",
    ]


def top_k_table(k: int, temperature: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
    ranked = sorted(range(len(LOGITS)), key=lambda index: LOGITS[index], reverse=True)
    kept = set(ranked[:k])
    kept_logits = [LOGITS[index] for index in ranked[:k]]
    _, _, _, kept_probs = _softmax(kept_logits, temperature)
    prob_by_index = {index: prob for index, prob in zip(ranked[:k], kept_probs)}
    rows = []
    for index, token in enumerate(VOCAB):
        rows.append(
            {
                "token": token,
                "logit": f4(LOGITS[index]),
                "kept": "yes" if index in kept else "no",
                "probability": f4(prob_by_index.get(index, 0.0)),
            }
        )
    kept_exps = [math.exp(LOGITS[index] / temperature) for index in ranked[:k]]
    exp_total = sum(kept_exps)
    lines = [
        "S = the k largest logits",
        "P(word) = exp(logit) / sum_S exp(logit)    if the word is in S",
        "P(word) = 0                                otherwise",
        f"k = {k}",
    ]
    for index, exp_value in zip(ranked[:k], kept_exps):
        lines.append(f"keep {VOCAB[index]}  logit {f4(LOGITS[index])}  exp { _shown_exp(exp_value)}")
    lines.append(f"sum of exp over S = {_shown_exp(exp_total)}")
    lines.append(f"The new probabilities sum to {f4(sum(kept_probs))}.")
    return pd.DataFrame(rows), lines


def top_p_table(p: float, temperature: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
    _, _, _, probs = _softmax(LOGITS, temperature)
    ranked = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)
    running = 0.0
    cutoff = len(ranked)
    trace: list[tuple[int, float, float]] = []
    for place, index in enumerate(ranked):
        running += probs[index]
        trace.append((index, probs[index], running))
        if running >= p:
            cutoff = place + 1
            break
    kept_trace = trace[:cutoff]
    kept_total = sum(item[1] for item in kept_trace)
    renorm = {index: prob / kept_total for index, prob, _ in kept_trace}
    rows = []
    for index, token in enumerate(VOCAB):
        rows.append(
            {
                "token": token,
                "probability": f4(probs[index]),
                "kept": "yes" if index in renorm else "no",
                "after renorm": f4(renorm.get(index, 0.0)),
            }
        )
    lines = [
        "Sort words from largest P to smallest.",
        "S = the smallest set whose probabilities add up to at least p",
        "P'(word) = P(word) / sum_S P(word)    if the word is in S",
        "P'(word) = 0                          otherwise",
        f"p = {f4(p)}",
    ]
    for index, prob, cum in kept_trace:
        lines.append(f"{VOCAB[index]}  {f4(prob)}  running {f4(cum)}")
    lines.append(f"Kept sum = {f4(kept_total)}")
    lines.append("Divide each kept probability by that sum.")
    lines.append(f"New sum = {f4(sum(renorm.values()))}")
    return pd.DataFrame(rows), lines
