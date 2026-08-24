"""CRF lab — self-contained lessons from HMM intuition to Python CRF code."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import streamlit as st

from st_compat import button, dataframe, plotly_chart

from credit_fraud.crf import (
    CRF_TRAINING_EXAMPLES,
    START,
    TAG_GLOSSARY,
    TripletCRF,
    demo_dog_barks_model,
    demo_the_dog_two_label,
    triplet_key,
)
from credit_fraud.crf_plots import (
    plot_complexity_visual,
    plot_crf_code_pipeline,
    plot_crf_forward_columns,
    plot_crf_forward_trellis,
    plot_crf_training_flow,
    plot_crf_viterbi_trellis,
    plot_forward_vs_viterbi_compare,
    plot_hmm_all_paths,
    plot_hmm_day_viterbi,
    plot_hmm_graph,
    plot_hmm_matrices,
    plot_hmm_problem_intro,
    plot_hmm_two_paths_compare,
    plot_hmm_viterbi_keep_drop,
    plot_hmm_viterbi_pruning,
    plot_hmm_viterbi_trellis,
    plot_hmm_vs_crf_contrast,
    plot_hmm_winning_path,
    plot_pos_emission_matrix,
    plot_pos_problem_intro,
    plot_pos_transition_matrix,
    plot_pos_viterbi_trellis,
)
from credit_fraud.hmm import PosTaggingHMM, ToyHMM

VIDEO_1 = "https://www.youtube.com/watch?v=fX5bYmnHqqE"
VIDEO_2 = "https://www.youtube.com/watch?v=IqXdjdOgXPM"
VIDEO_3 = "https://www.youtube.com/watch?v=rI3DQS0P2fk"
VIDEO_4 = "https://www.youtube.com/watch?v=XesdKt4J6Zc"

STEPS = [
    "1. Hidden Markov model",
    "2. Viterbi algorithm",
    "3. Conditional random fields",
    "4. CRF code walkthrough",
]


def _further_watching(url: str, title: str) -> None:
    with st.expander("Further watching"):
        st.markdown(f"[{title}]({url})")


def _path_from_label(label: str) -> list[str]:
    return label.split(" → ")


def _render_viterbi_pruning_math(hmm: ToyHMM, obs_seq: list[str], day_index: int) -> None:
    """Keep vs drop numbers for all target mood candidates next to the visual."""
    c = hmm.viterbi_day_candidates_prob(obs_seq, day_index)
    obs = obs_seq[day_index]
    day = day_index + 1

    st.markdown(f"**Day {day} transition evaluation (shirt: {obs})**")

    for target_mood in hmm.states:
        sub = c[c["mood"] == target_mood].sort_values("probability", ascending=False)
        kept = sub.iloc[0]
        dropped = sub.iloc[1] if len(sub) > 1 else None

        kept_from = str(kept["from"])
        kept_prob = float(kept["probability"])

        st.markdown(f"##### Target mood today: **{target_mood}**")
        col_k, col_d = st.columns(2)
        with col_k:
            st.markdown(f"**Keep:** :green-badge[{kept_from} → {target_mood}] = **{kept_prob:.4f}**")
            st.caption(f"{kept['how computed']}")
        if dropped is not None:
            drop_from = str(dropped["from"])
            drop_prob = float(dropped["probability"])
            with col_d:
                st.markdown(f"**Drop:** :red-badge[{drop_from} → {target_mood}] = **{drop_prob:.4f}**")
                st.caption(f"{dropped['how computed']}")
            ratio = kept_prob / drop_prob if drop_prob > 0 else float("inf")
            st.markdown(
                f"- **Decision:** Keep **{kept_from} → {target_mood}** ({kept_prob:.4f}), drop **{drop_from} → {target_mood}** ({drop_prob:.4f}) — kept route is **{ratio:.1f}×** stronger."
            )


def _render_mood_viterbi_step_by_step(
    hmm: ToyHMM,
    obs_seq: list[str],
    path: list[str],
    best_joint: float,
) -> None:
    """Day-by-day Viterbi walkthrough with probability tables."""
    st.markdown("- Each day highlights today's shirt color column in the emission matrix.")
    day1, day2, day3, summary = st.tabs(["Day 1", "Day 2", "Day 3", "Winning path"])

    with day1:
        obs = obs_seq[0]
        plotly_chart(plot_hmm_matrices(hmm, highlight_obs=obs), key="viterbi_mood_matrices_day1")
        c1 = hmm.viterbi_day_candidates_prob(obs_seq, 0)
        w1 = c1.iloc[0]
        st.markdown(f"**Day 1 scores** (shirt: **{obs}**)")
        for _, row in c1.iterrows():
            marker = " — wins Day 1" if row["mood"] == w1["mood"] else ""
            st.markdown(
                f"- **{row['mood']}:** P({row['mood']}) × P({obs}|{row['mood']}) = **{row['probability']:.3f}**{marker}"
            )

    with day2:
        obs = obs_seq[1]
        plotly_chart(plot_hmm_matrices(hmm, highlight_obs=obs), key="viterbi_mood_matrices_day2")
        plotly_chart(plot_hmm_viterbi_keep_drop(hmm, obs_seq, 1), key="viterbi_mood_keepdrop_day2")
        _render_viterbi_pruning_math(hmm, obs_seq, 1)

    with day3:
        obs = obs_seq[2]
        plotly_chart(plot_hmm_matrices(hmm, highlight_obs=obs), key="viterbi_mood_matrices_day3")
        plotly_chart(plot_hmm_viterbi_keep_drop(hmm, obs_seq, 2), key="viterbi_mood_keepdrop_day3")
        _render_viterbi_pruning_math(hmm, obs_seq, 2)

    with summary:
        plotly_chart(plot_hmm_winning_path(hmm, obs_seq, path), key="viterbi_mood_winning_path")
        st.markdown(
            f"""
- Best mood path: **{' → '.join(path)}**
- Joint probability: **{best_joint:.4f}**
"""
        )


def step_hmm() -> None:
    st.subheader("Hidden Markov model")

    hmm = ToyHMM.happy_sad()
    obs_seq = ToyHMM.sample_demo_sequence()
    path, _ = hmm.viterbi(obs_seq)
    all_paths = hmm.enumerate_all_paths(obs_seq)
    best_joint = float(all_paths.iloc[0]["joint probability"])

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        [
            "1. Overview",
            "2. Model diagram",
            "3. Probability tables",
            "4. Joint probability",
            "5. Joint example",
            "6. Two paths worked example",
            "7. All mood paths",
        ]
    )

    with t1:
        plotly_chart(plot_hmm_problem_intro(obs_seq))
        st.markdown(
            """
- **Green arrows:** Shirt color depends on today's hidden mood ($P(\\text{shirt} \\mid \\text{mood})$).
- **Blue arrows:** Today's mood depends on yesterday's mood ($P(\\text{mood}_t \\mid \\text{mood}_{t-1})$).
- **Goal:** Infer the most likely sequence of hidden moods given observed shirts.
"""
        )

    with t2:
        plotly_chart(plot_hmm_graph(hmm))
        st.markdown(
            """
- **START:** Day 1 prior ($P(\\text{Happy})=0.4, P(\\text{Sad})=0.6$).
- **States:** Hidden moods (Happy, Sad).
- **Emissions:** Visible shirt colors (red, green, blue).
"""
        )

    with t3:
        plotly_chart(plot_hmm_matrices(hmm))
        st.markdown(
            """
- **Transition matrix (left):** $P(\\text{mood}_t \\mid \\text{mood}_{t-1})$. Rows = previous mood, Columns = next mood.
- **Emission matrix (right):** $P(\\text{shirt} \\mid \\text{mood})$. Rows = mood, Columns = shirt color.
- Each row sums to **1.0**.
"""
        )

    with t4:
        st.markdown(
            r"""
**Joint Probability Formula:**

$$P(\mathbf{x}, \mathbf{y}) = P(y_1) \cdot P(x_1 \mid y_1) \prod_{t=2}^N P(y_t \mid y_{t-1}) \cdot P(x_t \mid y_t)$$

- **Prior $P(y_1)$:** START probability for Day 1.
- **Transition $P(y_t \mid y_{t-1})$:** Step-to-step state progression.
- **Emission $P(x_t \mid y_t)$:** Probability of generating the observed item.
"""
        )

    with t5:
        joint_example = (
            float(hmm.start[1])
            * float(hmm.emit[1, hmm.observations.index("green")])
            * float(hmm.trans[1, 1])
            * float(hmm.emit[1, hmm.observations.index("blue")])
            * float(hmm.trans[1, 0])
            * float(hmm.emit[0, hmm.observations.index("red")])
        )
        st.markdown(
            f"""
**Worked Example:** Path **Sad → Sad → Happy** with shirts **green → blue → red**:

```text
= P(Sad) × P(green | Sad) × P(Sad | Sad) × P(blue | Sad) × P(Happy | Sad) × P(red | Happy)
= 0.6 × 0.3 × 0.5 × 0.5 × 0.5 × 0.8
= {joint_example:.4f}
```
"""
        )

    with t6:
        top_two = hmm.top_two_paths(obs_seq)
        path_a_label = str(top_two.iloc[0]["mood path"])
        path_b_label = str(top_two.iloc[1]["mood path"])
        path_a = _path_from_label(path_a_label)
        path_b = _path_from_label(path_b_label)
        prob_a = float(top_two.iloc[0]["joint probability"])
        prob_b = float(top_two.iloc[1]["joint probability"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Best path: {path_a_label}** (Joint: **{prob_a:.4f}**)")
            dataframe(hmm.viterbi_step_table_prob(obs_seq, path_a), hide_index=True)
        with col_b:
            st.markdown(f"**Runner-up: {path_b_label}** (Joint: **{prob_b:.4f}**)")
            dataframe(hmm.viterbi_step_table_prob(obs_seq, path_b), hide_index=True)

        plotly_chart(plot_hmm_two_paths_compare(hmm, obs_seq, path_a, path_b))

    with t7:
        plotly_chart(plot_hmm_all_paths(hmm, obs_seq, best_path=path))
        st.markdown(
            f"""
- **2 moods × 3 days = {len(all_paths)} total paths**.
- Top path **{' → '.join(path)}** wins with probability **{best_joint:.4f}**.
- Step 2 (Viterbi) finds this exact winner without scoring every combination!
"""
        )

    _further_watching(VIDEO_1, "Hidden Markov Model : Data Science Concepts")


def step_viterbi() -> None:
    st.subheader("Viterbi algorithm")

    hmm = ToyHMM.happy_sad()
    obs_seq = ToyHMM.sample_demo_sequence()
    mood_path, _ = hmm.viterbi(obs_seq)
    all_paths = hmm.enumerate_all_paths(obs_seq)
    best_joint = float(all_paths.iloc[0]["joint probability"])

    pos_hmm = PosTaggingHMM.fans_watch_race()
    path, _ = pos_hmm.viterbi()
    step_table = pos_hmm.viterbi_step_table(path=path)
    word1_candidates = pos_hmm.viterbi_word_candidates(0)

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
        [
            "1. Why Viterbi & Complexity",
            "2. Mood HMM — step by step",
            "3. Mood HMM — trellis",
            "4. POS overview",
            "5. POS transition rules",
            "6. POS emission rules",
            "7. POS step-by-step scores",
            "8. POS trellis chart",
        ]
    )

    with t1:
        plotly_chart(plot_complexity_visual())
        st.markdown(
            r"""
### Why Complexity is $\mathcal{O}(N \cdot K^2)$ (The Math Explained)

1. **At each token step $t \to t+1$:**
   - There are $K$ possible destination states at step $t+1$.
   - To compute the best incoming path into **one** destination state, we must check incoming paths from all $K$ source states at step $t$.
   - For all $K$ destination states: $K \text{ states} \times K \text{ predecessors} = \mathbf{K^2}$ local operations per token step.
2. **Across the whole sequence of length $N$:**
   - Total operations $= N \text{ tokens} \times K^2 = \mathbf{\mathcal{O}(N \cdot K^2)}$.
3. **Compare with Brute Force ($\mathcal{O}(K^N)$):**
   - Brute force generates all $K \times K \times \dots \times K = K^N$ full paths.
   - For POS tagging with $N=20$ words and $K=45$ tags:
     - **Brute force:** $45^{20} \approx 1.19 \times 10^{33}$ operations (impossible).
     - **Viterbi DP:** $20 \times 45^2 = 40,500$ operations ($< 1\text{ millisecond}$ — **$2.9 \times 10^{28}\times$ faster!**).
"""
        )
        plotly_chart(plot_hmm_viterbi_pruning(hmm, obs_seq, 1), key="viterbi_mood_why_prune")
        _render_viterbi_pruning_math(hmm, obs_seq, 1)

    with t2:
        _render_mood_viterbi_step_by_step(hmm, obs_seq, mood_path, best_joint)

    with t3:
        plotly_chart(plot_hmm_viterbi_trellis(hmm, obs_seq, mood_path), key="viterbi_mood_trellis")
        st.markdown(
            """
- **Nodes (dots):** Maximum partial probability entering each mood at each day.
- **Thick black line:** Backtraced winning path (**Sad → Sad → Happy**).
- Weaker incoming paths to the same node are pruned immediately.
"""
        )

    with t4:
        plotly_chart(plot_pos_problem_intro(pos_hmm.sentence))
        tag_rows = [
            {"tag": tag, "name": info["full_name"], "means": info["means"]}
            for tag, info in TAG_GLOSSARY.items()
        ]
        dataframe(pd.DataFrame(tag_rows), hide_index=True)

    with t5:
        plotly_chart(plot_pos_transition_matrix(pos_hmm))
        st.caption("P(next tag | previous tag). Rows = previous tag, Columns = next tag. 0.0 = forbidden transition.")

    with t6:
        plotly_chart(plot_pos_emission_matrix(pos_hmm))
        st.caption("P(word | tag). Zero emission = word cannot be this tag (immediately pruned).")

    with t7:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(f"**Sentence:** *{' '.join(pos_hmm.sentence)}*")
            st.markdown(f"**Winning tag path:** {' → '.join(path)}")
            p_start = float(pos_hmm.start_tag[0])
            p_emit = float(pos_hmm.emit[0, 0])
            log_score = float(np.log(p_start) + np.log(p_emit))
            st.markdown(
                f"""
**Word 1 (*the*) — DET score:**
```text
log P(DET) + log P("the" | DET) = log({p_start:.1f}) + log({p_emit:.1f}) = {log_score:.2f}
```
"""
            )
        with col_right:
            st.markdown(
                r"""
**Why NOUN is pruned at Word 1:**
- $P(\text{"the"} \mid \text{NOUN}) = 0.0 \implies \log = -\infty$.
- Score is $-\infty \implies$ eliminated immediately!
"""
            )

        st.markdown("**All tag candidates at Word 1 (DET wins):**")
        dataframe(word1_candidates, hide_index=True)

        st.markdown("**Full Viterbi path (Word by Word):**")
        dataframe(step_table, hide_index=True)

    with t8:
        plotly_chart(plot_pos_viterbi_trellis(pos_hmm, path))
        st.markdown(
            """
- **Complexity:** $5 \text{ words} \times 3^2 \text{ tags}^2 = 45$ evaluations vs 243 full paths.
- Viterbi keeps strictly the best partial route entering each tag node.
"""
        )

    _further_watching(VIDEO_2, "The Viterbi Algorithm : Natural Language Processing")


def step_crf() -> None:
    st.subheader("Conditional random fields")

    model = demo_dog_barks_model()
    sentence3 = ["the", "dog", "barks"]

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "1. HMM vs CRF",
            "2. Scoring formula & features",
            "3. How Z(x) is calculated (Forward DP)",
            "4. How best path is chosen (Viterbi)",
            "5. How training works (Learning weights)",
        ]
    )

    with t1:
        plotly_chart(plot_hmm_vs_crf_contrast())
        st.markdown(
            r"""
| Characteristic | HMM (Generative) | CRF (Discriminative) |
|---|---|---|
| **What it models** | Joint $P(\mathbf{x}, \mathbf{y})$ (story generation) | Conditional $P(\mathbf{y} \mid \mathbf{x})$ (tag prediction) |
| **Parameters** | Fixed conditional probability tables | Learned weights $\mathbf{w}$ on feature functions $f(\mathbf{x}, \mathbf{y})$ |
| **Feature context** | Local only: $y_t$ depends solely on $x_t$ | Global: features can inspect any words in the sentence |
| **Training goal** | Count occurrences (MLE) | Maximize conditional log-likelihood $\sum \log P(\mathbf{y}^* \mid \mathbf{x})$ |
"""
        )

    with t2:
        st.markdown(
            r"""
### Linear-Chain CRF Scoring Formula

$$P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^N \sum_{j} w_j f_j(y_{t-1}, y_t, \mathbf{x}, t)\right)$$

**Key Components:**
- **Numerator $\exp(\text{score}(\mathbf{x}, \mathbf{y}))$:** Unnormalized affinity of tag sequence $\mathbf{y}$ for sentence $\mathbf{x}$.
- **Denominator $Z(\mathbf{x})$ (Partition function):** Sum of scores across all $K^N$ possible tag sequences so probabilities sum to $1.0$.
- **Feature $f_j$:** Binary/numeric rule, e.g. $\mathbb{I}(\text{prev}=\text{DET}, \text{curr}=\text{NOUN}, \text{word}=\text{"dog"})$.
- **Weight $w_j$:** Learned importance (positive = rewards, negative = penalizes).
"""
        )

        st.markdown(
            """
| Feature Template | Meaning | Weight Role |
|---|---|---|
| `(START, DET, "the")` | Sentence begins with determiner *the* | High positive $\to$ encourages DET at start |
| `(DET, NOUN, "dog")` | NOUN follows DET on word *dog* | High positive $\to$ strongly encourages noun phrase |
| `(NOUN, VERB, "barks")` | VERB follows NOUN on word *barks* | High positive $\to$ encourages verb action |
| `(DET, DET, "dog")` | Consecutive determiners on *dog* | Negative $\to$ strongly suppresses error |
"""
        )

    with t3:
        plotly_chart(plot_crf_forward_trellis(model, sentence3))
        st.markdown(
            r"""
### How $Z(\mathbf{x})$ is Calculated: Forward DP ($\mathcal{O}(N \cdot K^2)$)

**The Challenge:** $Z(\mathbf{x}) = \sum_{\mathbf{y} \in \mathcal{Y}^N} \exp(\text{score}(\mathbf{x}, \mathbf{y}))$ sums $K^N$ paths (exponential).

**The Solution (Distributive Law):** $(a + b) \cdot c = a \cdot c + b \cdot c \implies$ push sums inside products!

1. Define forward variable $\alpha(t, k) = $ total score mass of all partial paths ending in tag $k$ at word $t$.
2. **Base step ($t=1$):**
   $$\alpha(1, k) = \psi(\text{START}, k, x_1) = \exp(w_{(\text{START}, k, x_1)})$$
3. **Recurrence ($t > 1$):**
   $$\alpha(t, k) = \sum_{j=1}^K \Big( \alpha(t-1, j) \cdot \psi(j, k, x_t) \Big)$$
   $$\text{In log space: } \log \alpha(t, k) = \text{logsumexp}_{j=1}^K \Big( \log \alpha(t-1, j) + w_{(j, k, x_t)} \Big)$$
4. **Total Partition Function:**
   $$Z(\mathbf{x}) = \sum_{k=1}^K \alpha(N, k) \quad\iff\quad \log Z(\mathbf{x}) = \text{logsumexp}_{k=1}^K \big( \log \alpha(N, k) \big)$$
"""
        )

    with t4:
        plotly_chart(plot_forward_vs_viterbi_compare())
        plotly_chart(plot_crf_viterbi_trellis(model, sentence3))
        st.markdown(
            r"""
### How the Best Path is Chosen: Viterbi Decoding ($\mathcal{O}(N \cdot K^2)$)

**Goal:** Find $\mathbf{y}^* = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x}) = \arg\max_{\mathbf{y}} \text{score}(\mathbf{x}, \mathbf{y})$.

**Same DP as Forward, but replace $\sum$ with $\max$:**
1. **Recurrence:**
   $$V(t, k) = \max_{j=1}^K \Big( V(t-1, j) + w_{(j, k, x_t)} \Big)$$
   $$\text{backpointer}(t, k) = \arg\max_{j=1}^K \Big( V(t-1, j) + w_{(j, k, x_t)} \Big)$$
2. **Termination:** Pick best final tag $y_N^* = \arg\max_k V(N, k)$.
3. **Backtracking:** Follow backpointers backwards from $y_N^*$ to $y_1^*$.
"""
        )

    with t5:
        plotly_chart(plot_crf_training_flow())
        st.markdown(
            r"""
### How CRF Training Works (Learning the Weights)

**1. Training Objective: Negative Log-Likelihood (NLL) Loss**
For a gold labeled pair $(\mathbf{x}, \mathbf{y}^*)$, minimize:
$$\mathcal{L}(\mathbf{w}) = -\log P(\mathbf{y}^* \mid \mathbf{x}) = \log Z(\mathbf{x}; \mathbf{w}) - \text{score}(\mathbf{x}, \mathbf{y}^*; \mathbf{w})$$

**2. The Gradient Formula (Model Expectation vs Empirical Count)**
$$\frac{\partial \mathcal{L}}{\partial w_j} = \underbrace{\mathbb{E}_{\mathbf{y} \sim P(\mathbf{y} \mid \mathbf{x})}[f_j(\mathbf{x}, \mathbf{y})]}_{\text{Model Expected Feature Count}} - \underbrace{f_j(\mathbf{x}, \mathbf{y}^*)}_{\text{Empirical Gold Feature Count}}$$

**3. Intuitive Weight Update:**
$$w_j \leftarrow w_j - \eta \cdot \left( \mathbb{E}_{P(\mathbf{y} \mid \mathbf{x})}[f_j] - f_j^* \right)$$
- **If $\mathbb{E}[f_j] < f_j^*$:** Model under-predicts feature $j \to$ Gradient is negative $\to$ Weight $w_j$ **increases $\uparrow$** (boosts gold path).
- **If $\mathbb{E}[f_j] > f_j^*$:** Model over-predicts feature $j$ on wrong paths $\to$ Gradient is positive $\to$ Weight $w_j$ **decreases $\downarrow$** (penalizes wrong paths).
"""
        )

        st.markdown("**Sample Supervised Training Sentences:**")
        dataframe(pd.DataFrame(CRF_TRAINING_EXAMPLES), hide_index=True)

    _further_watching(VIDEO_3, "Conditional Random Fields (CRF)")


def step_code() -> None:
    st.subheader("CRF code walkthrough")

    model = demo_dog_barks_model()
    sentence3 = ["the", "dog", "barks"]
    gold3 = ["DET", "NOUN", "VERB"]
    path_score = model.path_score(sentence3, gold3)
    z = model.partition(sentence3)
    prob = model.probability(sentence3, gold3)
    rows3 = model.enumerate_paths_table(sentence3)
    forward_table = model.forward_step_table(sentence3)
    viterbi_table = model.viterbi_step_table(sentence3)
    grad_data = model.training_gradient_step(sentence3, gold3)

    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "1. Pipeline architecture",
            "2. Path score (numerator)",
            "3. Forward DP → Z(x) (denominator)",
            "4. Viterbi decode (best path)",
            "5. Training in code (loss & gradients)",
            "6. Source code",
        ]
    )

    with t1:
        plotly_chart(plot_crf_code_pipeline())
        st.markdown(
            r"""
1. **`path_score()`:** Multiplies triplet potentials $\psi = \exp(w)$ along one path $\to$ Numerator.
2. **`_forward_log()`:** Fast DP in log space $\to$ Computes partition function $Z(\mathbf{x})$ in $\mathcal{O}(N \cdot K^2)$.
3. **`probability()`:** Computes $\exp(\log \text{score} - \log Z) \implies \frac{\text{score}}{Z(\mathbf{x})}$.
4. **`viterbi()`:** DP using $\max$ and backpointers $\to$ Returns highest-scoring tag sequence.
5. **`training_gradient_step()`:** Computes $\nabla \mathcal{L} = \mathbb{E}[f] - f^*$ to update weights.
"""
        )

    with t2:
        st.markdown(
            f"""
**Sentence:** *{' '.join(sentence3)}* · **Gold Tags:** *{' → '.join(gold3)}*

$$\\text{{Numerator Score}} = \\psi(\\text{{START}}, \\text{{DET}}, \\text{{"the"}}) \\times \\psi(\\text{{DET}}, \\text{{NOUN}}, \\text{{"dog"}}) \\times \\psi(\\text{{NOUN}}, \\text{{VERB}}, \\text{{"barks"}})$$
$$= 2.0 \\times 3.0 \\times 2.0 = \\mathbf{{{path_score:.0f}}}$$
"""
        )
        pot_rows = [
            {
                "triplet feature": f"({prev}, {tag}, '{word}')",
                "log weight w": round(model.potentials.get(triplet_key(prev, tag, word), 0.0), 3),
                "potential ψ = exp(w)": model.potential(prev, tag, word),
            }
            for prev, tag, word in [
                (START, "DET", "the"),
                ("DET", "NOUN", "dog"),
                ("NOUN", "VERB", "barks"),
            ]
        ]
        dataframe(pd.DataFrame(pot_rows), hide_index=True)

        st.markdown("**All 27 tag paths (sorted by score):**")
        dataframe(pd.DataFrame(rows3), hide_index=True)

    with t3:
        plotly_chart(plot_crf_forward_trellis(model, sentence3))
        st.markdown(
            f"""
- **Partition Function $Z(\\mathbf{{x}}) = \\sum \\alpha_3 = \\mathbf{{{z:.0f}}}$** ($log Z = {np.log(z):.2f}$).
- **Gold Path Probability:** $P(\\text{{DET NOUN VERB}} \\mid \\text{{the dog barks}}) = \\frac{{{path_score:.0f}}}{{{z:.0f}}} = \\mathbf{{{prob:.2%}}}$.
"""
        )
        st.markdown("**Step-by-step Forward DP accumulation table:**")
        dataframe(forward_table, hide_index=True)

    with t4:
        plotly_chart(plot_crf_viterbi_trellis(model, sentence3))
        path_pred, _ = model.viterbi(sentence3)
        st.markdown(
            f"""
- **Winning Decoded Path:** **{' → '.join(path_pred)}** (matches gold truth!).
- **How chosen:** Each cell preserves the maximum incoming score and stores a backpointer.
"""
        )
        st.markdown("**Step-by-step Viterbi max score and backpointer table:**")
        dataframe(viterbi_table, hide_index=True)

    with t5:
        st.markdown(
            f"""
**Supervised Training Step on:** *{' '.join(sentence3)}* (Gold: *{' → '.join(gold3)}*)

- **Negative Log-Likelihood Loss:** $\\mathcal{{L}} = \\log Z - \\log \\text{{score}}^* = {np.log(z):.2f} - {np.log(path_score):.2f} = \\mathbf{{{grad_data['loss']:.3f}}}$
- **Model Gold Probability:** $\\mathbf{{{grad_data['p_gold']:.1%}}}$
"""
        )
        st.markdown("**Feature Weights, Expectations, Gradients & Updates:**")
        dataframe(grad_data["update_table"], hide_index=True)
        st.caption("Gradient = Expected count under model − Empirical count in gold. Weight update: w_new = w − η · Gradient.")

    with t6:
        st.markdown("**1. Triplet Key and Path Score (Numerator)**")
        st.code(inspect.getsource(triplet_key), language="python")
        st.code(
            inspect.getsource(TripletCRF.potential)
            + "\n"
            + inspect.getsource(TripletCRF.path_score),
            language="python",
        )
        st.markdown("**2. Forward DP and Partition Function Z(x) (Denominator)**")
        st.code(inspect.getsource(TripletCRF._forward_log), language="python")
        st.code(inspect.getsource(TripletCRF.partition), language="python")
        st.markdown("**3. Viterbi Decoding (Best Path Extraction)**")
        st.code(inspect.getsource(TripletCRF.viterbi), language="python")
        st.markdown("**4. Training Gradient Calculation (Loss & Updates)**")
        st.code(inspect.getsource(TripletCRF.training_gradient_step), language="python")

    _further_watching(VIDEO_4, "Implementing a basic CRF in Python")


STEP_RENDERERS = {
    STEPS[0]: step_hmm,
    STEPS[1]: step_viterbi,
    STEPS[2]: step_crf,
    STEPS[3]: step_code,
}


def _shift_step(delta: int) -> None:
    current = st.session_state.get("crf_step_radio", STEPS[0])
    idx = STEPS.index(current) if current in STEPS else 0
    st.session_state.crf_step_radio = STEPS[max(0, min(len(STEPS) - 1, idx + delta))]


def render() -> None:
    if st.session_state.get("crf_step_radio") not in STEPS:
        st.session_state.crf_step_radio = STEPS[0]

    st.title("CRF lab")
    st.caption(
        "Four self-contained lessons from moods and shirts to Python CRF code. "
        "Use the sidebar or Next/Previous buttons to move through the steps."
    )

    current = st.sidebar.radio(
        "Session step",
        options=STEPS,
        index=0,
        key="crf_step_radio",
    )
    idx = STEPS.index(current) if current in STEPS else 0

    prev_col, _, next_col = st.columns([1, 4, 1])
    with prev_col:
        button(
            "Previous",
            key="crf_prev",
            disabled=idx == 0,
            on_click=_shift_step,
            args=(-1,),
        )
    with next_col:
        button(
            "Next",
            key="crf_next",
            disabled=idx == len(STEPS) - 1,
            on_click=_shift_step,
            args=(1,),
        )

    st.markdown(f"**{current}** · step {idx + 1} of {len(STEPS)}")
    STEP_RENDERERS[current]()
