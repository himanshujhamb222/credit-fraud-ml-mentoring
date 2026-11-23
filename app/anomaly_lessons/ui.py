"""Shared compact UI helpers for the Financial Anomaly Detection lab."""

from __future__ import annotations

import streamlit as st

from anomaly_lab.interview_bank import InterviewQA

_QTYPE_BADGE = {
    "Conceptual": ":blue-badge[Conceptual]",
    "Mathematical": ":violet-badge[Mathematical]",
    "Numerical": ":orange-badge[Numerical]",
    "Scenario": ":green-badge[Scenario]",
}


def lesson_header(
    title: str,
    subtitle: str,
    *,
    priority_badge: str,
    interview_focus: list[str] | None = None,
) -> None:
    """Compact lesson header: badge, title, 1-line subtitle, and crisp interview focus bullets."""
    st.markdown(priority_badge)
    st.header(title)
    st.caption(subtitle)
    if interview_focus:
        with st.container(border=True):
            st.markdown(":material/school: **Interview focus**")
            st.markdown("\n".join(f"- {item}" for item in interview_focus))


def working(lines: list[str]) -> None:
    """Compact monospaced arithmetic block."""
    st.code("\n".join(lines), language="text")


def render_interview_qa(
    qas: list[InterviewQA],
    *,
    key_prefix: str,
    heading: str = "Interview prep",
) -> None:
    """Render short, scannable interview Q&A cards."""
    st.subheader(heading)
    reveal_all = st.toggle("Reveal all answers", value=False, key=f"{key_prefix}_reveal_all")
    for qa in qas:
        badge = _QTYPE_BADGE.get(qa.q_type, f":gray-badge[{qa.q_type}]")
        with st.expander(f"{badge}  {qa.question}", expanded=reveal_all):
            st.markdown(qa.answer)
            st.info(f"**Takeaway:** {qa.key_point}", icon=":material/lightbulb:")
