"""Teaching CRF aligned with the code walkthrough video (triplet potentials + DP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import logsumexp

START = "START"

TAG_GLOSSARY: dict[str, dict[str, str]] = {
    "DET": {
        "full_name": "Determiner",
        "means": "Points to a noun — **the**, **a**, **an**.",
        "example_words": "the",
    },
    "NOUN": {
        "full_name": "Noun",
        "means": "Person, place, or thing — **dog**, **cat**.",
        "example_words": "dog",
    },
    "VERB": {
        "full_name": "Verb",
        "means": "Action or state — **barks**, **sat**.",
        "example_words": "barks",
    },
}


TAGS = ["DET", "NOUN", "VERB"]

# Tiny labeled corpus for the CRF lab — real systems use thousands of sentences like this.
CRF_TRAINING_EXAMPLES: list[dict[str, str]] = [
    {"sentence": "the dog", "gold tags": "DET → NOUN"},
    {"sentence": "the cat", "gold tags": "DET → NOUN"},
    {"sentence": "a dog barks", "gold tags": "DET → NOUN → VERB"},
    {"sentence": "the dog barks", "gold tags": "DET → NOUN → VERB"},
    {"sentence": "cats run", "gold tags": "NOUN → VERB"},
]


def triplet_key(prev_tag: str, tag: str, word: str) -> tuple[str, str, str]:
    """Label–label–word potential key (video 4)."""
    return (prev_tag, tag, word.lower())


@dataclass
class TripletCRF:
    """Minimal linear-chain CRF with exp(triplet) potentials and forward DP."""

    tags: list[str] = field(default_factory=lambda: list(TAGS))
    potentials: dict[tuple[str, str, str], float] = field(default_factory=dict)
    tag2idx: dict[str, int] = field(init=False)
    idx2tag: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = list(self.tags)

    def potential(self, prev_tag: str, tag: str, word: str) -> float:
        """Triplet score passed through exp(), as in the code video."""
        raw = self.potentials.get(triplet_key(prev_tag, tag, word), 0.0)
        return float(np.exp(raw))

    def path_score(self, sentence: list[str], tag_seq: list[str]) -> float:
        """Numerator: multiply triplet potentials along a tag path."""
        if len(sentence) != len(tag_seq):
            raise ValueError("sentence and tag_seq length mismatch")
        score = 1.0
        prev = START
        for word, tag in zip(sentence, tag_seq):
            score *= self.potential(prev, tag, word)
            prev = tag
        return float(score)

    def _forward_log(self, sentence: list[str]) -> tuple[float, np.ndarray]:
        """Log-space forward pass for the partition function."""
        n = len(sentence)
        t_count = len(self.tags)
        alpha = np.full((n, t_count), -np.inf)
        for t_idx, tag in enumerate(self.tags):
            alpha[0, t_idx] = np.log(max(self.potential(START, tag, sentence[0]), 1e-300))
        for i in range(1, n):
            word = sentence[i]
            for curr_idx, tag in enumerate(self.tags):
                terms = [
                    alpha[i - 1, prev_idx]
                    + np.log(max(self.potential(prev_tag, tag, word), 1e-300))
                    for prev_idx, prev_tag in enumerate(self.tags)
                ]
                alpha[i, curr_idx] = float(logsumexp(terms))
        log_z = float(logsumexp(alpha[-1]))
        return log_z, alpha

    def partition(self, sentence: list[str]) -> float:
        """Denominator Z(x): sum of scores over all tag sequences."""
        log_z, _ = self._forward_log(sentence)
        return float(np.exp(log_z))

    def probability(self, sentence: list[str], tag_seq: list[str]) -> float:
        log_z, _ = self._forward_log(sentence)
        log_num = np.log(max(self.path_score(sentence, tag_seq), 1e-300))
        return float(np.exp(log_num - log_z))

    def enumerate_paths(self, sentence: list[str]) -> list[tuple[list[str], float]]:
        """Naive enumeration for tiny examples (two-label demo)."""
        return [
            (list(combo), self.path_score(sentence, list(combo)))
            for combo in product(self.tags, repeat=len(sentence))
        ]

    def enumerate_paths_table(self, sentence: list[str]) -> list[dict[str, str | float]]:
        """All tag paths with scores and probabilities, sorted by score."""
        paths = self.enumerate_paths(sentence)
        z = self.partition(sentence)
        rows = []
        for tags, score in paths:
            rows.append(
                {
                    "tag path": " → ".join(tags),
                    "path score": round(score, 2),
                    "P(path | sentence)": round(score / z if z > 0 else 0.0, 2),
                }
            )
        return sorted(rows, key=lambda r: float(r["path score"]), reverse=True)

    def viterbi(self, sentence: list[str]) -> tuple[list[str], np.ndarray]:
        """Best tag sequence — same DP idea as the HMM Viterbi video."""
        n = len(sentence)
        t_count = len(self.tags)
        scores = np.full((n, t_count), -np.inf)
        back = np.zeros((n, t_count), dtype=int)

        for t_idx, tag in enumerate(self.tags):
            scores[0, t_idx] = np.log(max(self.potential(START, tag, sentence[0]), 1e-300))
        for i in range(1, n):
            word = sentence[i]
            for curr_idx, tag in enumerate(self.tags):
                best = -np.inf
                best_prev = 0
                for prev_idx, prev_tag in enumerate(self.tags):
                    cand = scores[i - 1, prev_idx] + np.log(
                        max(self.potential(prev_tag, tag, word), 1e-300)
                    )
                    if cand > best:
                        best = cand
                        best_prev = prev_idx
                scores[i, curr_idx] = best
                back[i, curr_idx] = best_prev

        last = int(np.argmax(scores[-1]))
        path_idx = [last]
        for i in range(n - 1, 0, -1):
            last = int(back[i, last])
            path_idx.append(last)
        path_idx.reverse()
        return [self.idx2tag[i] for i in path_idx], scores

    def forward_step_table(self, sentence: list[str]) -> pd.DataFrame:
        """Detailed breakdown of forward DP accumulation at each word step."""
        log_z, alpha = self._forward_log(sentence)
        rows = []
        for step, word in enumerate(sentence):
            for t_idx, tag in enumerate(self.tags):
                log_val = float(alpha[step, t_idx])
                exp_val = float(np.exp(log_val))
                if step == 0:
                    how = f"ψ(START, {tag}, '{word}') = {self.potential(START, tag, word):.1f}"
                else:
                    prev_word = sentence[step - 1]
                    terms = [
                        f"α_{{{step-1}}}({p})·ψ({p},{tag},'{word}')"
                        for p in self.tags
                    ]
                    how = " + ".join(terms)
                rows.append(
                    {
                        "step": step + 1,
                        "word": word,
                        "tag": tag,
                        "log α": round(log_val, 3),
                        "α = exp(log α)": round(exp_val, 3),
                        "incoming sum computation": how,
                    }
                )
        return pd.DataFrame(rows)

    def viterbi_step_table(self, sentence: list[str]) -> pd.DataFrame:
        """Detailed breakdown of Viterbi DP max score and backpointer at each step."""
        path, scores = self.viterbi(sentence)
        n = len(sentence)
        rows = []
        for step, word in enumerate(sentence):
            for t_idx, tag in enumerate(self.tags):
                best_score = float(scores[step, t_idx])
                is_on_winning_path = (path[step] == tag)
                if step == 0:
                    how = f"log ψ(START, {tag}, '{word}') = {best_score:.2f}"
                    best_from = "START"
                else:
                    # find which prev tag gave this best score
                    prev_scores = [
                        scores[step - 1, p_idx] + np.log(max(self.potential(p_tag, tag, word), 1e-300))
                        for p_idx, p_tag in enumerate(self.tags)
                    ]
                    best_prev_idx = int(np.argmax(prev_scores))
                    best_from = self.tags[best_prev_idx]
                    how = f"max over prev tags → came from {best_from} (score {best_score:.2f})"

                rows.append(
                    {
                        "step": step + 1,
                        "word": word,
                        "tag": tag,
                        "Viterbi log score": round(best_score, 3),
                        "best prev tag": best_from,
                        "winning path?": "★ WINNER" if is_on_winning_path else "",
                        "how chosen": how,
                    }
                )
        return pd.DataFrame(rows)

    def training_gradient_step(
        self,
        sentence: list[str],
        gold_tags: list[str],
        learning_rate: float = 0.1,
    ) -> dict:
        """Compute exact NLL loss, feature counts, and weight updates for one sentence."""
        all_paths = self.enumerate_paths(sentence)
        z = self.partition(sentence)
        log_z, _ = self._forward_log(sentence)

        # 1. Gold path score and loss
        gold_score = self.path_score(sentence, gold_tags)
        loss = log_z - np.log(max(gold_score, 1e-300))

        # 2. Empirical feature counts on gold path
        gold_triplets: dict[tuple[str, str, str], float] = {}
        prev = START
        for w, t in zip(sentence, gold_tags):
            k = triplet_key(prev, t, w)
            gold_triplets[k] = gold_triplets.get(k, 0.0) + 1.0
            prev = t

        # 3. Model path probabilities
        path_probs: list[tuple[list[str], float, float]] = []
        for tags, score in all_paths:
            prob = score / z if z > 0 else 0.0
            path_probs.append((tags, score, prob))

        # 4. Expected feature counts across all paths
        expected_triplets: dict[tuple[str, str, str], float] = {}
        for tags, _, prob in path_probs:
            prev = START
            for w, t in zip(sentence, tags):
                k = triplet_key(prev, t, w)
                expected_triplets[k] = expected_triplets.get(k, 0.0) + prob
                prev = t

        # 5. Build gradient and update table for all relevant triplets
        all_keys = set(gold_triplets.keys()) | set(expected_triplets.keys()) | set(self.potentials.keys())
        table_rows = []
        for k in sorted(all_keys):
            curr_w = self.potentials.get(k, 0.0)
            emp = gold_triplets.get(k, 0.0)
            exp_val = expected_triplets.get(k, 0.0)
            # Gradient of NLL = E[f] - f_gold
            grad = exp_val - emp
            new_w = curr_w - learning_rate * grad
            action = "UP ↑ (Boost gold)" if grad < -0.01 else ("DOWN ↓ (Penalize wrong)" if grad > 0.01 else "STABLE ↔")
            table_rows.append(
                {
                    "triplet feature": f"({k[0]}, {k[1]}, '{k[2]}')",
                    "curr weight": round(curr_w, 3),
                    "gold count f*": round(emp, 2),
                    "model expected E[f]": round(exp_val, 3),
                    "gradient (E[f] - f*)": round(grad, 3),
                    "update direction": action,
                    f"new weight (η={learning_rate})": round(new_w, 3),
                }
            )

        return {
            "sentence": sentence,
            "gold_tags": gold_tags,
            "loss": loss,
            "z": z,
            "gold_score": gold_score,
            "p_gold": gold_score / z if z > 0 else 0.0,
            "update_table": pd.DataFrame(table_rows),
        }


def demo_dog_barks_model() -> TripletCRF:
    """*the dog barks* example from the code video."""
    model = TripletCRF()
    model.potentials = {
        triplet_key(START, "DET", "the"): np.log(2.0),
        triplet_key("DET", "NOUN", "dog"): np.log(3.0),
        triplet_key("NOUN", "VERB", "barks"): np.log(2.0),
        triplet_key(START, "DET", "dog"): np.log(0.5),
        triplet_key("DET", "DET", "dog"): np.log(0.5),
        triplet_key("DET", "NOUN", "barks"): np.log(0.5),
        triplet_key("NOUN", "NOUN", "barks"): np.log(0.5),
        triplet_key("VERB", "VERB", "barks"): np.log(0.5),
    }
    return model


def demo_the_dog_two_label() -> TripletCRF:
    """Two-label *the dog* naive probability demo from video 4."""
    model = TripletCRF(tags=["DET", "NOUN"])
    model.potentials = {
        triplet_key(START, "DET", "the"): np.log(1.0),
        triplet_key("DET", "NOUN", "dog"): np.log(7.0),
        triplet_key(START, "DET", "dog"): np.log(1.0),
        triplet_key("DET", "DET", "dog"): np.log(1.0),
    }
    return model
