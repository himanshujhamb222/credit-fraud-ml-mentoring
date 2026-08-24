"""Toy Hidden Markov Models for the CRF teaching lab."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

START = "START"


@dataclass(frozen=True)
class ToyHMM:
    """Happy/sad mood HMM with shirt-color observations."""

    states: list[str]
    observations: list[str]
    start: np.ndarray
    trans: np.ndarray
    emit: np.ndarray

    @classmethod
    def happy_sad(cls) -> ToyHMM:
        """Professor mood example: hidden Happy/Sad, observed red/green/blue shirts."""
        states = ["Happy", "Sad"]
        observations = ["red", "green", "blue"]
        start = np.array([0.4, 0.6], dtype=float)
        trans = np.array([[0.7, 0.3], [0.5, 0.5]], dtype=float)
        emit = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.3, 0.5],
            ],
            dtype=float,
        )
        return cls(states=states, observations=observations, start=start, trans=trans, emit=emit)

    def _obs_index(self, symbol: str) -> int:
        try:
            return self.observations.index(symbol)
        except ValueError as exc:
            raise ValueError(
                f"Unknown observation {symbol!r}; expected one of {self.observations}"
            ) from exc

    def viterbi(self, obs_seq: list[str]) -> tuple[list[str], np.ndarray]:
        """Return best hidden-state path and log-score trellis."""
        if not obs_seq:
            return [], np.zeros((0, len(self.states)))

        n = len(obs_seq)
        t_count = len(self.states)
        log_start = np.log(self.start)
        log_trans = np.log(self.trans)
        obs_idx = [self._obs_index(o) for o in obs_seq]
        log_emit = np.log(self.emit[:, obs_idx]).T

        scores = np.full((n, t_count), -np.inf)
        back = np.zeros((n, t_count), dtype=int)
        scores[0] = log_start + log_emit[0]
        for i in range(1, n):
            for curr in range(t_count):
                candidates = scores[i - 1] + log_trans[:, curr]
                back[i, curr] = int(np.argmax(candidates))
                scores[i, curr] = candidates[back[i, curr]] + log_emit[i, curr]

        last = int(np.argmax(scores[-1]))
        path_idx = [last]
        for i in range(n - 1, 0, -1):
            last = int(back[i, last])
            path_idx.append(last)
        path_idx.reverse()
        return [self.states[i] for i in path_idx], scores

    @staticmethod
    def sample_demo_sequence() -> list[str]:
        """Three-day shirt sequence used in the lab (green → blue → red)."""
        return ["green", "blue", "red"]

    def viterbi_step_table(
        self,
        obs_seq: list[str],
        path: list[str] | None = None,
    ) -> pd.DataFrame:
        """Day-by-day joint-probability breakdown for a hidden-state path."""
        if path is None:
            path, _ = self.viterbi(obs_seq)
        if len(path) != len(obs_seq):
            raise ValueError("path and obs_seq length mismatch")

        rows: list[dict[str, str | float]] = []
        log_total = 0.0
        for i, (state, obs) in enumerate(zip(path, obs_seq)):
            s_idx = self.states.index(state)
            o_idx = self._obs_index(obs)
            if i == 0:
                p_a = float(self.start[s_idx])
                p_b = float(self.emit[s_idx, o_idx])
                factor = f"P({state}) × P({obs} | {state})"
                value = f"{p_a:.1f} × {p_b:.1f}"
            else:
                prev = path[i - 1]
                prev_idx = self.states.index(prev)
                p_a = float(self.trans[prev_idx, s_idx])
                p_b = float(self.emit[s_idx, o_idx])
                factor = f"P({state} | {prev}) × P({obs} | {state})"
                value = f"{p_a:.1f} × {p_b:.1f}"
            log_step = float(np.log(p_a) + np.log(p_b))
            log_total += log_step
            rows.append(
                {
                    "step": f"Day {i + 1}",
                    "factor": factor,
                    "value": value,
                    "running log score": round(log_total, 2),
                }
            )
        return pd.DataFrame(rows)

    def viterbi_day_candidates(self, obs_seq: list[str], day_index: int) -> pd.DataFrame:
        """Per-day competing scores for teaching — all moods (day 1) or all incoming paths."""
        if day_index < 0 or day_index >= len(obs_seq):
            raise ValueError("day_index out of range")

        obs = obs_seq[day_index]
        o_idx = self._obs_index(obs)
        rows: list[dict[str, str | float]] = []

        if day_index == 0:
            for s_idx, state in enumerate(self.states):
                p_start = float(self.start[s_idx])
                p_emit = float(self.emit[s_idx, o_idx])
                log_score = float(np.log(p_start) + np.log(p_emit))
                rows.append(
                    {
                        "mood": state,
                        "from": START,
                        "log score": round(log_score, 2),
                        "how computed": (
                            f"log P({state}) + log P({obs}|{state}) "
                            f"= log({p_start:.1f}) + log({p_emit:.1f})"
                        ),
                    }
                )
            return pd.DataFrame(rows).sort_values("log score", ascending=False)

        _, trellis = self.viterbi(obs_seq)
        prev_scores = trellis[day_index - 1]
        for curr_idx, curr in enumerate(self.states):
            for prev_idx, prev in enumerate(self.states):
                p_trans = float(self.trans[prev_idx, curr_idx])
                p_emit = float(self.emit[curr_idx, o_idx])
                prev_score = float(prev_scores[prev_idx])
                log_score = float(prev_score + np.log(p_trans) + np.log(p_emit))
                rows.append(
                    {
                        "mood": curr,
                        "from": prev,
                        "log score": round(log_score, 2),
                        "how computed": (
                            f"score({prev}) + log P({curr}|{prev}) + log P({obs}|{curr}) "
                            f"= {prev_score:.2f} + log({p_trans:.1f}) + log({p_emit:.1f})"
                        ),
                    }
                )
        return pd.DataFrame(rows).sort_values(["mood", "log score"], ascending=[True, False])

    def top_two_paths(self, obs_seq: list[str]) -> pd.DataFrame:
        """Best and runner-up mood paths for teaching."""
        return self.enumerate_all_paths(obs_seq).head(2)

    def path_joint_probability(self, obs_seq: list[str], path: list[str]) -> float:
        """Joint probability P(path, obs_seq) as a plain product of table entries."""
        if len(path) != len(obs_seq):
            raise ValueError("path and obs_seq length mismatch")
        prob = 1.0
        for i, (state, obs) in enumerate(zip(path, obs_seq)):
            s_idx = self.states.index(state)
            o_idx = self._obs_index(obs)
            if i == 0:
                prob *= float(self.start[s_idx]) * float(self.emit[s_idx, o_idx])
            else:
                prev_idx = self.states.index(path[i - 1])
                prob *= float(self.trans[prev_idx, s_idx]) * float(self.emit[s_idx, o_idx])
        return float(prob)

    def enumerate_all_paths(self, obs_seq: list[str]) -> pd.DataFrame:
        """Every mood path with its joint probability (brute-force view)."""
        rows: list[dict[str, str | float]] = []
        for path in product(self.states, repeat=len(obs_seq)):
            path_list = list(path)
            rows.append(
                {
                    "mood path": " → ".join(path_list),
                    "joint probability": self.path_joint_probability(obs_seq, path_list),
                }
            )
        return pd.DataFrame(rows).sort_values("joint probability", ascending=False)

    def viterbi_day_candidates_prob(self, obs_seq: list[str], day_index: int) -> pd.DataFrame:
        """Per-day competing paths using plain probabilities (teaching)."""
        if day_index < 0 or day_index >= len(obs_seq):
            raise ValueError("day_index out of range")

        obs = obs_seq[day_index]
        o_idx = self._obs_index(obs)
        rows: list[dict[str, str | float]] = []

        if day_index == 0:
            for s_idx, state in enumerate(self.states):
                p_start = float(self.start[s_idx])
                p_emit = float(self.emit[s_idx, o_idx])
                prob = p_start * p_emit
                rows.append(
                    {
                        "mood": state,
                        "from": START,
                        "probability": prob,
                        "how computed": f"P({state}) × P({obs}|{state}) = {p_start:.1f} × {p_emit:.1f}",
                    }
                )
            return pd.DataFrame(rows).sort_values("probability", ascending=False)

        _, trellis = self.viterbi(obs_seq)
        prev_probs = np.exp(trellis[day_index - 1])
        for curr_idx, curr in enumerate(self.states):
            p_emit = float(self.emit[curr_idx, o_idx])
            for prev_idx, prev in enumerate(self.states):
                p_trans = float(self.trans[prev_idx, curr_idx])
                prev_prob = float(prev_probs[prev_idx])
                prob = prev_prob * p_trans * p_emit
                rows.append(
                    {
                        "mood": curr,
                        "from": prev,
                        "probability": prob,
                        "how computed": (
                            f"P(best→{prev}) × P({curr}|{prev}) × P({obs}|{curr}) "
                            f"= {prev_prob:.3f} × {p_trans:.1f} × {p_emit:.1f}"
                        ),
                    }
                )
        return pd.DataFrame(rows).sort_values(["mood", "probability"], ascending=[True, False])

    def viterbi_pruning_pair(
        self,
        obs_seq: list[str],
        day_index: int,
    ) -> dict[str, str | float | list[str]]:
        """Two competing routes into the same mood — for teaching why pruning is safe."""
        if day_index < 1:
            raise ValueError("pruning comparison needs day_index >= 1")

        obs = obs_seq[day_index]
        prev_obs = obs_seq[day_index - 1]
        candidates = self.viterbi_day_candidates_prob(obs_seq, day_index)
        mood = str(candidates.loc[candidates["probability"].idxmax(), "mood"])
        group = candidates[candidates["mood"] == mood].sort_values("probability", ascending=False)
        kept = group.iloc[0]
        dropped = group.iloc[1] if len(group) > 1 else None
        if dropped is None:
            raise ValueError(f"only one route into {mood} on day {day_index + 1}")

        kept_from = str(kept["from"])
        drop_from = str(dropped["from"])
        kept_prev = float(kept["probability"]) / (
            float(self.trans[self.states.index(kept_from), self.states.index(mood)])
            * float(self.emit[self.states.index(mood), self._obs_index(obs)])
        )
        drop_prev = float(dropped["probability"]) / (
            float(self.trans[self.states.index(drop_from), self.states.index(mood)])
            * float(self.emit[self.states.index(mood), self._obs_index(obs)])
        )
        p_trans_keep = float(self.trans[self.states.index(kept_from), self.states.index(mood)])
        p_trans_drop = float(self.trans[self.states.index(drop_from), self.states.index(mood)])
        p_emit = float(self.emit[self.states.index(mood), self._obs_index(obs)])

        result: dict[str, str | float] = {
            "day": day_index + 1,
            "shirt": obs,
            "prev_shirt": prev_obs,
            "mood": mood,
            "kept_from": kept_from,
            "drop_from": drop_from,
            "kept_prev": kept_prev,
            "drop_prev": drop_prev,
            "p_trans_keep": p_trans_keep,
            "p_trans_drop": p_trans_drop,
            "p_emit": p_emit,
            "kept_prob": float(kept["probability"]),
            "drop_prob": float(dropped["probability"]),
            "ratio": float(kept["probability"]) / float(dropped["probability"]),
        }
        if day_index == 1:
            k_idx = self.states.index(kept_from)
            d_idx = self.states.index(drop_from)
            o_prev = self._obs_index(prev_obs)
            result["kept_p_start"] = float(self.start[k_idx])
            result["kept_p_emit_prev"] = float(self.emit[k_idx, o_prev])
            result["drop_p_start"] = float(self.start[d_idx])
            result["drop_p_emit_prev"] = float(self.emit[d_idx, o_prev])
        return result

    def viterbi_step_table_prob(
        self,
        obs_seq: list[str],
        path: list[str] | None = None,
    ) -> pd.DataFrame:
        """Day-by-day probability breakdown for the winning (or given) path."""
        if path is None:
            path, _ = self.viterbi(obs_seq)
        if len(path) != len(obs_seq):
            raise ValueError("path and obs_seq length mismatch")

        rows: list[dict[str, str | float]] = []
        running = 1.0
        for i, (state, obs) in enumerate(zip(path, obs_seq)):
            s_idx = self.states.index(state)
            o_idx = self._obs_index(obs)
            if i == 0:
                p_a = float(self.start[s_idx])
                p_b = float(self.emit[s_idx, o_idx])
                factor = f"P({state}) × P({obs} | {state})"
                value = f"{p_a:.1f} × {p_b:.1f}"
            else:
                prev = path[i - 1]
                prev_idx = self.states.index(prev)
                p_a = float(self.trans[prev_idx, s_idx])
                p_b = float(self.emit[s_idx, o_idx])
                factor = f"P({state} | {prev}) × P({obs} | {state})"
                value = f"{p_a:.1f} × {p_b:.1f}"
            running *= p_a * p_b
            rows.append(
                {
                    "step": f"Day {i + 1}",
                    "shirt": obs,
                    "mood": state,
                    "factor": factor,
                    "multiply": value,
                    "running probability": round(running, 4),
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class PosTaggingHMM:
    """POS HMM for *the fans watch the race*."""

    tags: list[str]
    words: list[str]
    start_tag: np.ndarray
    trans: np.ndarray
    emit: np.ndarray
    sentence: list[str]

    @classmethod
    def fans_watch_race(cls) -> PosTaggingHMM:
        tags = ["DET", "NOUN", "VERB"]
        sentence = ["the", "fans", "watch", "the", "race"]
        tag2idx = {t: i for i, t in enumerate(tags)}

        start_tag = np.array([0.8, 0.1, 0.1], dtype=float)
        trans = np.array(
            [
                [0.0, 0.9, 0.1],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        )
        # Columns follow sentence position (two distinct "the" tokens).
        emit = np.full((len(tags), len(sentence)), 1e-9, dtype=float)
        emit[tag2idx["DET"], 0] = 0.2
        emit[tag2idx["DET"], 3] = 0.2
        emit[tag2idx["NOUN"], 1] = 0.1
        emit[tag2idx["VERB"], 1] = 0.2
        emit[tag2idx["NOUN"], 2] = 0.3
        emit[tag2idx["VERB"], 2] = 0.15
        emit[tag2idx["NOUN"], 4] = 0.1
        emit[tag2idx["VERB"], 4] = 0.3

        return cls(
            tags=tags,
            words=sentence,
            start_tag=start_tag,
            trans=trans,
            emit=emit,
            sentence=sentence,
        )

    def viterbi(self, sentence: list[str] | None = None) -> tuple[list[str], np.ndarray]:
        sentence = sentence or self.sentence
        if sentence != self.sentence:
            raise ValueError("POS demo supports the fixed lab sentence only.")
        n = len(sentence)
        t_count = len(self.tags)
        log_start = np.log(self.start_tag)
        log_trans = np.where(self.trans > 0, np.log(self.trans), -np.inf)
        log_emit = np.where(self.emit > 0, np.log(self.emit), -np.inf).T

        scores = np.full((n, t_count), -np.inf)
        back = np.zeros((n, t_count), dtype=int)
        scores[0] = log_start + log_emit[0]
        for i in range(1, n):
            for curr in range(t_count):
                candidates = scores[i - 1] + log_trans[:, curr]
                back[i, curr] = int(np.argmax(candidates))
                scores[i, curr] = candidates[back[i, curr]] + log_emit[i, curr]

        last = int(np.argmax(scores[-1]))
        path_idx = [last]
        for i in range(n - 1, 0, -1):
            last = int(back[i, last])
            path_idx.append(last)
        path_idx.reverse()
        return [self.tags[i] for i in path_idx], scores

    def viterbi_step_table(
        self,
        sentence: list[str] | None = None,
        path: list[str] | None = None,
    ) -> pd.DataFrame:
        """Word-by-word log-score breakdown for the best (or given) tag path."""
        sentence = sentence or self.sentence
        if sentence != self.sentence:
            raise ValueError("POS demo supports the fixed lab sentence only.")
        if path is None:
            path, _ = self.viterbi(sentence)

        tag2idx = {t: i for i, t in enumerate(self.tags)}
        rows: list[dict[str, str | float]] = []
        log_total = 0.0
        for i, (tag, word) in enumerate(zip(path, sentence)):
            t_idx = tag2idx[tag]
            p_emit = float(self.emit[t_idx, i])
            if i == 0:
                p_start = float(self.start_tag[t_idx])
                factor = f"P({tag}) × P({word!r} | {tag})"
                value = f"{p_start:.1f} × {p_emit:.1f}"
                log_step = float(np.log(p_start) + np.log(p_emit))
            else:
                prev = path[i - 1]
                prev_idx = tag2idx[prev]
                p_trans = float(self.trans[prev_idx, t_idx])
                factor = f"P({tag} | {prev}) × P({word!r} | {tag})"
                value = f"{p_trans:.1f} × {p_emit:.1f}"
                log_step = float(np.log(p_trans) + np.log(p_emit))
            log_total += log_step
            rows.append(
                {
                    "word": f"{i + 1}: {word}",
                    "factor": factor,
                    "value": value,
                    "running log score": round(log_total, 2),
                }
            )
        return pd.DataFrame(rows)

    def viterbi_word_candidates(self, word_index: int) -> pd.DataFrame:
        """Show competing tag scores at one word position (for teaching)."""
        if word_index < 0 or word_index >= len(self.sentence):
            raise ValueError("word_index out of range")
        _, trellis = self.viterbi()
        word = self.sentence[word_index]
        rows = []
        for t_idx, tag in enumerate(self.tags):
            score = float(trellis[word_index, t_idx])
            p_emit = float(self.emit[t_idx, word_index])
            if word_index == 0:
                detail = f"log P({tag}) + log P({word!r}|{tag}) = log({self.start_tag[t_idx]:.1f}) + log({p_emit:.1f})"
            else:
                detail = f"best incoming path + log P({word!r}|{tag})"
            rows.append(
                {
                    "tag": tag,
                    "log score": round(score, 2),
                    "how computed": detail,
                }
            )
        return pd.DataFrame(rows).sort_values("log score", ascending=False)


def default_hmm() -> ToyHMM:
    return ToyHMM.happy_sad()
