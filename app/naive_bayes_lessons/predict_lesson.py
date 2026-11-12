"""predict(): independence assumption and Laplace word probabilities."""

from __future__ import annotations

import streamlit as st

from naive_bayes_lab.walks import smoothed_example, training_walk
from naive_bayes_lessons.ui import working


def render() -> None:
    st.header("predict(): derive the word probabilities")
    walk = training_walk()
    assume, laplace = st.tabs(
        ["Step 3.1 — Naive assumption", "Step 3.2 — Laplace smoothing"]
    )

    with assume:
        st.markdown(
            r"For a new review like **good movie** we need "
            r"\(P(review\mid +)\) and \(P(review\mid -)\). "
            r"How do we get the probability of a whole sentence?"
        )
        st.markdown("Chain rule (exact):")
        st.latex(
            r"P(w_1,\ldots,w_n\mid c)=\prod_{i=1}^{n}"
            r"P(w_i\mid w_1,\ldots,w_{i-1},c)"
        )
        st.markdown(
            r"Naive Bayes assumes that, **given the class**, words are independent. "
            r"Replace each conditional with \(P(w_i\mid c)\):"
        )
        st.latex(r"P(review\mid c)\approx\prod_i P(w_i\mid c)")
        st.markdown("For **good movie**:")
        st.latex(r"P(review\mid +)\approx P(good\mid +)\,P(movie\mid +)")
        st.latex(r"P(review\mid -)\approx P(good\mid -)\,P(movie\mid -)")
        st.markdown("That is why we counted words separately for each sentiment.")

    with laplace:
        st.markdown("Without smoothing, relative frequency:")
        st.latex(
            r"P(good\mid +)\approx\frac{\text{count of good in +}}"
            r"{\text{total word occurrences in +}}=\frac{1}{4}"
        )
        st.markdown(
            r"Problem: **great** never appears in negative reviews, so "
            r"\(P(great\mid -)=0\). Any review with *great* would get "
            r"zero probability under negative, no matter what else it says."
        )
        st.markdown(
            r"**Fix — Laplace smoothing.** Add \(\alpha=1\) to every word count. "
            rf"Vocabulary size \(V={walk.vocab_size}\), so add \(V\) to the denominator:"
        )
        st.latex(
            r"P(w\mid c)=\frac{\operatorname{count}(w,c)+\alpha}"
            r"{\operatorname{totalWords}(c)+\alpha V}"
        )

        examples = st.columns(3)
        with examples[0]:
            st.markdown("**good | +**")
            working(smoothed_example("good", 1))
        with examples[1]:
            st.markdown("**movie | +**")
            working(smoothed_example("movie", 1))
        with examples[2]:
            st.markdown("**movie | −**")
            working(smoothed_example("movie", 0))

        st.code(
            "alpha = 1\n"
            "V = len(vocab)\n"
            "p_word_pos = (word_counts[1][word] + alpha) / (\n"
            "    total_words[1] + alpha * V\n"
            ")\n"
            "p_word_neg = (word_counts[0][word] + alpha) / (\n"
            "    total_words[0] + alpha * V\n"
            ")",
            language="python",
        )
        st.caption("Every vocabulary word now has a nonzero probability in both classes.")
