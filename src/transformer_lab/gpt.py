"""Next-word loss, and why generation is a different loop."""

from __future__ import annotations

import math

import pandas as pd

from transformer_lab.common import f4

PAIRS = (
    ("the", "cat", 0.70),
    ("cat", "sat", 0.60),
    ("sat", "on", 0.50),
)


def shift_frame() -> pd.DataFrame:
    rows = []
    for seen, target, prob in PAIRS:
        rows.append(
            {
                "sees": seen,
                "predicts": target,
                "P(next)": f4(prob),
                "loss": f4(-math.log(prob)),
            }
        )
    return pd.DataFrame(rows)


def dependency_lines() -> list[str]:
    return [
        "The next word depends only on the words already seen.",
        "P(cat | the)",
        "P(sat | the, cat)",
        "P(on  | the, cat, sat)",
        "The mask is that rule. Position t cannot see the words after it.",
        "Training already has the true sentence.",
        "Every prefix above is known before the pass starts.",
        "So those three probabilities are three rows of one matrix.",
        "L = - [ log P(cat|the) + log P(sat|the, cat) + log P(on|the, cat, sat) ]",
        "Nothing in L waits for a word the model just wrote.",
    ]


def generation_lines() -> list[str]:
    return [
        "Generation does not have the later words yet.",
        "The word you just sampled becomes the next input.",
        "x2 ~ P(. | x1)",
        "x3 ~ P(. | x1, x2)",
        "x4 ~ P(. | x1, x2, x3)",
        "x3 cannot be scored until x2 exists.",
        "That wait is the sequential loop.",
        "Training skips it, because x1, x2, and x3 are already on the page.",
    ]


def loss_lines() -> list[str]:
    seen, target, prob = PAIRS[0]
    loss = -math.log(prob)
    losses = [-math.log(item[2]) for item in PAIRS]
    mean = sum(losses) / len(losses)
    return [
        "Sentence: the cat sat on",
        "Input: the cat sat",
        "Targets: cat sat on",
        f"At '{seen}', the true next word is {target}.",
        f"P({target} | {seen}) = {f4(prob)}",
        f"loss = -log({f4(prob)}) = {f4(loss)}",
        f"Mean loss = ({' + '.join(f4(item) for item in losses)}) / {len(losses)} = {f4(mean)}",
    ]


