"""Tiny numeric walks that mirror the DIET paper formulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EXAMPLE_UTTERANCE = "play ping pong"
EXAMPLE_INTENT = "play_game"
EXAMPLE_ENTITY = ("game_name", "ping pong")
TOKENS = ["play", "ping", "pong", "__CLS__"]


@dataclass(frozen=True)
class FeaturizeWalk:
    tokens: list[str]
    sparse_dim: int
    dense_dim: int
    projected: pd.DataFrame
    lines: list[str]


@dataclass(frozen=True)
class IntentLossWalk:
    intents: list[str]
    similarities: dict[str, float]
    loss: float
    lines: list[str]
    frame: pd.DataFrame


@dataclass(frozen=True)
class MaskLossWalk:
    tokens: list[str]
    masked_index: int
    masked_as: str
    similarities: dict[str, float]
    loss: float
    lines: list[str]


def pipeline_steps() -> list[tuple[str, str]]:
    return [
        ("1. Tokens", "Split the sentence. Append a special __CLS__ token at the end."),
        ("2. Features", "Build sparse (word/char) and optional dense (GloVe/BERT/ConveRT) vectors."),
        ("3. Project", "Shared feed-forward layers align dimensions, then concatenate."),
        ("4. Transformer", "A 2-layer transformer with relative position attention mixes context."),
        ("5. Entities", "A CRF reads every token vector and tags spans (BILOU / BIO)."),
        ("6. Intent", "The __CLS__ vector is compared to intent label embeddings via dot product."),
        ("7. Mask (optional)", "Predict randomly masked tokens — same style of loss as intent."),
        ("8. Total loss", "L_total = L_I + L_E + L_M. Any term can be turned off."),
    ]


def featurize_walk() -> FeaturizeWalk:
    """Toy numbers: sparse → dense projection, then concat."""
    rng = np.random.default_rng(7)
    sparse_dim, dense_dim, hidden = 8, 4, 4
    # One-hot-ish sparse + multi-hot char n-gram toy vectors
    sparse = rng.integers(0, 2, size=(len(TOKENS), sparse_dim)).astype(float)
    dense = rng.normal(0, 0.4, size=(len(TOKENS), dense_dim))
    # Shared feed-forward: sparse → dense_dim
    w = rng.normal(0, 0.3, size=(sparse_dim, dense_dim))
    projected_sparse = sparse @ w
    concat = np.concatenate([projected_sparse, dense], axis=1)
    # Second shared FF → transformer width (hidden)
    w2 = rng.normal(0, 0.3, size=(concat.shape[1], hidden))
    transformer_in = concat @ w2

    frame = pd.DataFrame(
        {
            "token": TOKENS,
            "sparse→dense": [f"[{', '.join(f'{v:.2f}' for v in row)}]" for row in projected_sparse],
            "dense": [f"[{', '.join(f'{v:.2f}' for v in row)}]" for row in dense],
            "transformer in": [f"[{', '.join(f'{v:.2f}' for v in row)}]" for row in transformer_in],
        }
    )
    lines = [
        "Sparse features: one-hot word + multi-hot char n-grams (n ≤ 5).",
        "Dense features: frozen GloVe / BERT / ConveRT embeddings (plug-and-play).",
        "Shared FF: sparse → same width as dense, then concatenate.",
        "Shared FF again: concatenated vector → transformer width (paper: 256).",
        f"Toy dims here: sparse={sparse_dim}, dense={dense_dim}, transformer={hidden}.",
    ]
    return FeaturizeWalk(TOKENS, sparse_dim, dense_dim, frame, lines)


def intent_loss_walk() -> IntentLossWalk:
    """Dot-product intent loss with one positive and two negatives (Eq. 2)."""
    intents = ["play_game", "play_music", "set_alarm"]
    # Toy embeddings in R^3 (paper uses R^20)
    h_cls = np.array([0.8, 0.5, 0.1])
    h_labels = {
        "play_game": np.array([0.9, 0.4, 0.0]),
        "play_music": np.array([0.2, 0.7, 0.3]),
        "set_alarm": np.array([-0.1, 0.1, 0.9]),
    }
    sims = {name: float(h_cls @ vec) for name, vec in h_labels.items()}
    s_pos = sims["play_game"]
    negs = [sims["play_music"], sims["set_alarm"]]
    # L_I = -(S+ - log(e^{S+} + Σ e^{S-}))
    denom = np.exp(s_pos) + sum(np.exp(s) for s in negs)
    loss = float(-(s_pos - np.log(denom)))

    lines = [
        "Embed __CLS__ output and each intent label into the same space h ∈ R^d.",
        f"h_CLS = {np.array2string(h_cls, precision=2)}",
        f"S+ = h_CLS · h_play_game = {s_pos:.3f}",
        f"S- (play_music) = {sims['play_music']:.3f}",
        f"S- (set_alarm)  = {sims['set_alarm']:.3f}",
        "L_I = -( S+ − log( e^{S+} + Σ e^{S-} ) )",
        f"L_I = -( {s_pos:.3f} − log({denom:.3f}) ) = {loss:.3f}",
        "Inference: rank intents by the same dot product. Highest score wins.",
    ]
    frame = pd.DataFrame(
        {
            "intent": intents,
            "similarity": [sims[i] for i in intents],
            "role": ["positive", "negative", "negative"],
        }
    )
    return IntentLossWalk(intents, sims, loss, lines, frame)


def mask_loss_walk() -> MaskLossWalk:
    """One masked token, same dot-product form as intent (Eq. 3)."""
    tokens = ["play", "ping", "pong"]
    masked_index = 1
    masked_as = "__MASK__"
    h_mask = np.array([0.6, 0.4, 0.2])
    vocab = {
        "ping": np.array([0.7, 0.3, 0.1]),
        "play": np.array([0.1, 0.8, 0.2]),
        "music": np.array([0.0, 0.2, 0.9]),
    }
    sims = {w: float(h_mask @ vec) for w, vec in vocab.items()}
    s_pos = sims["ping"]
    negs = [sims["play"], sims["music"]]
    denom = np.exp(s_pos) + sum(np.exp(s) for s in negs)
    loss = float(-(s_pos - np.log(denom)))
    lines = [
        "Pick ~15% of tokens. Of those: 70% → __MASK__, 10% → random, 20% keep.",
        f"Here token[{masked_index}] = 'ping' is shown as {masked_as}.",
        "Transformer output a_MASK is embedded; compare to token label embeddings.",
        "Same loss shape as intent: pull the true token close, push negatives away.",
        f"L_M = {loss:.3f}",
    ]
    return MaskLossWalk(tokens, masked_index, masked_as, sims, loss, lines)
