"""BERT's 15% mask and the loss on the selected words."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from transformer_lab.common import f4

SENTENCE = (
    "the quick brown fox jumps over the lazy dog and then the small cat "
    "sat on the red mat near the old barn after lunch"
).split()
EXTRA_WORDS = ["bird", "tree", "moon", "ship"]
SEED = 6


def _true_prob(index: int) -> float:
    return round(0.30 + (index % 7) * 0.08, 2)


def mask_draw(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vocab = list(dict.fromkeys([*SENTENCE, *EXTRA_WORDS]))
    rows: list[dict[str, object]] = []
    for index, word in enumerate(SENTENCE):
        selected = bool(rng.random() < 0.15)
        if not selected:
            rows.append(
                {
                    "position": index + 1,
                    "word": word,
                    "shown": word,
                    "role": "not selected",
                    "P(true word)": "",
                    "loss": f4(0.0),
                }
            )
            continue
        roll = float(rng.random())
        if roll < 0.8:
            shown = "[MASK]"
            role = "80% mask"
        elif roll < 0.9:
            choices = [item for item in vocab if item != word]
            shown = str(rng.choice(choices))
            role = "10% random"
        else:
            shown = word
            role = "10% keep"
        prob = _true_prob(index)
        rows.append(
            {
                "position": index + 1,
                "word": word,
                "shown": shown,
                "role": role,
                "P(true word)": f4(prob),
                "loss": f4(-math.log(prob)),
            }
        )
    return pd.DataFrame(rows)


def loss_lines(frame: pd.DataFrame) -> list[str]:
    selected = frame[frame["role"] != "not selected"]
    first = selected.iloc[0]
    prob = float(first["P(true word)"])
    loss = -math.log(prob)
    losses = [float(value) for value in selected["loss"]]
    return [
        f"Seed = {SEED}. About 15% of the words are selected.",
        "Of those, 80% become [MASK], 10% become a random word, 10% stay.",
        f"Work the first selected word: {first['word']}.",
        f"The model is shown: {first['shown']}.",
        f"P({first['word']}) = {f4(prob)}",
        f"loss = -log({f4(prob)}) = {f4(loss)}",
        "Words that were not selected add 0.",
        f"Mean loss on the selected words = {f4(float(np.mean(losses)))}",
    ]


def head_lines() -> list[str]:
    hidden = (0.4, 0.8)
    weight = (1.0, 0.5)
    bias = 0.0
    score = hidden[0] * weight[0] + hidden[1] * weight[1] + bias
    return [
        "Fine-tune keeps the encoder.",
        "A small new layer reads one vector from it.",
        f"h = [{f4(hidden[0])}, {f4(hidden[1])}]",
        f"w = [{f4(weight[0])}, {f4(weight[1])}], b = {f4(bias)}",
        f"score = {f4(hidden[0])}*{f4(weight[0])} + {f4(hidden[1])}*{f4(weight[1])} + {f4(bias)}",
        f"score = {f4(score)}",
        "That score can mean positive, or a name, or a match.",
    ]
