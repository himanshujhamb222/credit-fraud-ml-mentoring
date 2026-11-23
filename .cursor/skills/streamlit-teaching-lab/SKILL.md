---
name: streamlit-teaching-lab
description: Builds short, crisp, equation-first teaching labs for the Streamlit dashboard with minimal prose, worked numbers, interactive visuals, and punchy interview Q&As. Use whenever adding, refactoring, or editing a teaching lab or lesson page in this Streamlit dashboard.
---

# Concise Equation-First Streamlit Teaching Labs

Build every lab page to be **scannable in seconds**: minimal prose, clear equations first, worked numbers, and interactive visuals.

## Hard Content Rules

1. **Minimal text:** No paragraph longer than 1-2 short sentences (<= 25 words). Cut filler words and textbook storytelling.
2. **Equations first:** Lead concepts with `st.latex(...)` and a 3-bullet symbol legend instead of prose explanations.
3. **Short bullets only:** Max 3-5 bullets per section, <= 12 words per bullet.
4. **Show, don't tell:** Replace explanations with:
   - Compact `working([...])` arithmetic blocks (3-5 lines)
   - Interactive sliders + Vega/Altair charts (`altair_chart`)
   - Compact 3-5 row tables (`dataframe(..., hide_index=True)`)
   - Small Mermaid diagrams (`mermaid_chart`) for flows/architectures
5. **Crisp interview Q&As:** Keep each answer to 1-2 tight sentences + 1-line **Takeaway**.
6. **Raw strings for LaTeX:** Always use `r"""..."""` or `r"..."` for any string containing LaTeX backslashes to prevent Python `SyntaxWarning` or `\t` tab bugs.

## Standard Algorithm Lesson Layout (4 Tabs)

Keep every core algorithm inside 4 clean tabs covering all key elements with low cognitive load:

1. **Intuition & math:** 1-line intuition -> `st.latex(...)` equations -> 3-bullet symbol/threshold reading guide.
2. **Interactive demo:** 3-bullet learning summary -> sliders + metrics + chart + 4-line worked numbers.
3. **Code & limits:** Short runnable Python snippet (10-20 lines) + compact 3-4 row limitations table.
4. **Interview prep:** Revealable Q&A expanders (`Conceptual`, `Mathematical`, `Numerical`, `Scenario`) with 1-2 sentence answers and a 1-line takeaway.

## Good vs Bad Copy Example

**Bad (too text-heavy):**
```python
st.markdown(
    "When we want to figure out whether a transaction is unusual, we can look at "
    "how far away it sits from the typical historical average for that customer. "
    "Because different customers spend in different dollar ranges, we divide by "
    "the standard deviation so the result becomes dimensionless..."
)
```

**Good (short, equation-first):**
```python
st.markdown("Measures how many standard deviations a transaction sits from the mean.")
st.latex(r"z_i = \frac{x_i - \mu}{\sigma}")
st.markdown(
    r"""
- **$|z| \le 2.0$:** normal (~95%)
- **$|z| > 2.5$:** suspicious tail
- **$|z| > 3.0$:** strong anomaly (~0.3%)
"""
)
```

## Repo Integration Checklist

When adding a new `<topic>` lab:

- [ ] Put pure math, datasets, and models in `src/<topic>_lab/`
- [ ] Put lesson modules in `app/<topic>_lessons/` (using `st_compat` helpers and `@st.cache_data`)
- [ ] Create page wrapper `app/app_pages/<topic>_lab.py` with sidebar topic radio + Previous/Next buttons
- [ ] Register the page in `app/streamlit_app.py` and add a 1-line bullet in `README.md`
- [ ] Verify with `.venv/bin/python -W error::SyntaxWarning` compile check and `streamlit.testing.v1.AppTest` across every lesson
