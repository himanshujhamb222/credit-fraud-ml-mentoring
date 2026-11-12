"""Tiny arithmetic for the residual add and the head split."""

from __future__ import annotations

from transformer_lab.common import f4


def residual_lines() -> list[str]:
    x = (1.0, 0.5)
    attention = (0.2, -0.1)
    total = (x[0] + attention[0], x[1] + attention[1])
    return [
        f"x = [{f4(x[0])}, {f4(x[1])}]",
        f"attention output = [{f4(attention[0])}, {f4(attention[1])}]",
        f"sum = [{f4(x[0])} + {f4(attention[0])}, {f4(x[1])} + ({f4(attention[1])})]",
        f"sum = [{f4(total[0])}, {f4(total[1])}]",
        "The original numbers are still in the vector.",
    ]


def head_lines() -> list[str]:
    d_model = 8
    n_heads = 4
    d_head = d_model // n_heads
    return [
        f"d_model = {d_model}",
        f"n_heads = {n_heads}",
        f"d_head = {d_model} / {n_heads} = {d_head}",
        f"Join the heads: {n_heads} * {d_head} = {n_heads * d_head}",
        "The width is d_model again.",
        "One last matrix, W_o, mixes the heads.",
    ]
