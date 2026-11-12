"""Scaled dot-product attention on a 3-word sentence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from transformer_lab.common import f4, labeled_frame, softmax_lines, softmax_rows

TOKENS = ["the", "cat", "sat"]
Q = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)
K = np.array(
    [
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
    ]
)
V = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)
D_K = 2
FOCUS_I = 1
FOCUS_J = 1
LARGE_SCORES = np.array([16.0, 8.0, 0.0])
LARGE_D_K = 64


@dataclass
class AttentionWalk:
    tokens: list[str]
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    scores: np.ndarray
    scale: float
    scaled: np.ndarray
    weights: np.ndarray
    output: np.ndarray
    dot_lines: list[str]
    scale_lines: list[str]
    softmax_lines: list[str]
    output_lines: list[str]


def vector_frame(matrix: np.ndarray, tokens: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(np.round(matrix, 4), index=tokens, columns=["dim 1", "dim 2"])
    frame.index.name = "token"
    return frame


def _dot_lines(q_row: np.ndarray, k_row: np.ndarray, query: str, key: str) -> list[str]:
    products = [float(q_row[i]) * float(k_row[i]) for i in range(len(q_row))]
    score = float(sum(products))
    product_text = " + ".join(
        f"{f4(q_row[i])}*{f4(k_row[i])}" for i in range(len(q_row))
    )
    return [
        f"{query} looks up {key}.",
        f"q = [{f4(q_row[0])}, {f4(q_row[1])}]",
        f"k = [{f4(k_row[0])}, {f4(k_row[1])}]",
        f"score = {product_text} = {f4(score)}",
    ]


def _scale_lines(row: np.ndarray, tokens: list[str], scale: float) -> list[str]:
    lines = [
        f"d_k = {D_K}",
        f"sqrt(d_k) = sqrt({D_K}) = {f4(scale)}",
    ]
    for name, score in zip(tokens, row):
        lines.append(f"{name}: {f4(score)} / {f4(scale)} = {f4(float(score) / scale)}")
    return lines


def _output_lines(weights: np.ndarray, values: np.ndarray, tokens: list[str]) -> list[str]:
    output = weights @ values
    lines = ["output = weight_1*v_1 + weight_2*v_2 + weight_3*v_3"]
    for dim in range(values.shape[1]):
        parts = " + ".join(
            f"{f4(weights[j])}*{f4(values[j, dim])}" for j in range(len(tokens))
        )
        pieces = [float(weights[j]) * float(values[j, dim]) for j in range(len(tokens))]
        lines.append(f"dim {dim + 1}: {parts} = {f4(sum(pieces))}")
    lines.append(f"output vector = [{f4(output[0])}, {f4(output[1])}]")
    return lines


def attention_walk() -> AttentionWalk:
    scores = Q @ K.T
    scale = math.sqrt(D_K)
    scaled = scores / scale
    weights = softmax_rows(scaled)
    output = weights @ V
    focus_scores = scaled[FOCUS_I]
    soft_lines, _ = softmax_lines(focus_scores, TOKENS)
    return AttentionWalk(
        tokens=TOKENS,
        q=Q,
        k=K,
        v=V,
        scores=scores,
        scale=scale,
        scaled=scaled,
        weights=weights,
        output=output,
        dot_lines=_dot_lines(Q[FOCUS_I], K[FOCUS_J], TOKENS[FOCUS_I], TOKENS[FOCUS_J]),
        scale_lines=_scale_lines(scores[FOCUS_I], TOKENS, scale),
        softmax_lines=[f"Softmax for the {TOKENS[FOCUS_I]} row."] + soft_lines,
        output_lines=[f"Mix the values for {TOKENS[FOCUS_I]}."] + _output_lines(weights[FOCUS_I], V, TOKENS),
    )


def score_frame(matrix: np.ndarray) -> pd.DataFrame:
    return labeled_frame(matrix, TOKENS, index_name="query \\ key")


@dataclass
class ScalingWalk:
    raw: np.ndarray
    scaled: np.ndarray
    raw_weights: np.ndarray
    scaled_weights: np.ndarray
    why_lines: list[str]
    divide_lines: list[str]
    raw_softmax_lines: list[str]
    scaled_softmax_lines: list[str]


def scaling_walk(seed: int = 0, draws: int = 400) -> ScalingWalk:
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 1.0, size=(draws, LARGE_D_K))
    k = rng.normal(0.0, 1.0, size=(draws, LARGE_D_K))
    dots = np.sum(q * k, axis=1)
    scale = math.sqrt(LARGE_D_K)
    scaled = LARGE_SCORES / scale
    raw_lines, raw_weights = softmax_lines(LARGE_SCORES, TOKENS)
    scaled_lines, scaled_weights = softmax_lines(scaled, TOKENS)
    why = [
        f"d_k = {LARGE_D_K}",
        f"sqrt(d_k) = {f4(scale)}",
        "Each part of q and k has variance 1.",
        "The dot product then has variance d_k.",
        f"One random dot product = {f4(float(dots[0]))}",
        f"Variance of {draws} dot products = {f4(float(dots.var()))}",
        "Divide by sqrt(d_k) so a typical score stays near size 1.",
        "A huge score makes softmax put almost all weight on one word.",
    ]
    divide = [f"Divide the large scores by sqrt({LARGE_D_K}) = {f4(scale)}."]
    for name, score in zip(TOKENS, LARGE_SCORES):
        divide.append(f"{name}: {f4(score)} / {f4(scale)} = {f4(float(score) / scale)}")
    return ScalingWalk(
        raw=LARGE_SCORES,
        scaled=scaled,
        raw_weights=raw_weights,
        scaled_weights=scaled_weights,
        why_lines=why,
        divide_lines=divide,
        raw_softmax_lines=["Softmax before scaling."] + raw_lines,
        scaled_softmax_lines=["Softmax after scaling."] + scaled_lines,
    )


@dataclass
class CausalWalk:
    scaled: np.ndarray
    masked: np.ndarray
    full_weights: np.ndarray
    causal_weights: np.ndarray
    lines: list[str]


def causal_walk(walk: AttentionWalk | None = None) -> CausalWalk:
    base = walk or attention_walk()
    masked = base.scaled.copy()
    future = np.triu_indices(len(base.tokens), k=1)
    masked[future] = -np.inf
    query_i, key_j = 0, 2
    raw_score = float(base.scaled[query_i, key_j])
    lines = [
        f"{base.tokens[query_i]} is word 1. {base.tokens[key_j]} is word 3.",
        f"{base.tokens[key_j]} comes later, so a decoder blocks it.",
        f"scaled score = {f4(raw_score)}",
        "add -inf",
        "masked score = -inf",
        "exp(-inf) = 0",
        "that weight becomes 0",
    ]
    return CausalWalk(
        scaled=base.scaled,
        masked=masked,
        full_weights=base.weights,
        causal_weights=softmax_rows(masked),
        lines=lines,
    )
