"""Why we accumulate log-odds instead of multiplying tiny probabilities."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Why calculate log-odds?")
    st.markdown("From Bayes' theorem, the posterior odds are:")
    st.latex(
        r"\frac{P(+\mid review)}{P(-\mid review)}"
        r"=\frac{P(+)}{P(-)}\cdot\frac{P(review\mid +)}{P(review\mid -)}"
    )
    st.markdown("With the Naive Bayes product over words:")
    st.latex(
        r"=\frac{P(+)}{P(-)}\prod_i\frac{P(w_i\mid +)}{P(w_i\mid -)}"
    )
    st.markdown(
        r"Take \(\log\). Because \(\log(ab)=\log a+\log b\), multiplication "
        "becomes addition:"
    )
    st.latex(
        r"\mathrm{score}(review)="
        r"\log\frac{P(+)}{P(-)}"
        r"+\sum_i\log\frac{P(w_i\mid +)}{P(w_i\mid -)}"
    )
    st.markdown("This score is exactly the log of the posterior odds.")

    st.subheader("How to read the score")
    st.markdown(
        """
- **Positive score** → posterior odds > 1 → positive is more likely.
- **Negative score** → posterior odds < 1 → negative is more likely.
- **Zero score** → odds = 1 → both classes equally likely.
- Bonus: we avoid multiplying many tiny probabilities (underflow).
"""
    )
