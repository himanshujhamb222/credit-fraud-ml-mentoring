"""Shared display helpers for the Naive Bayes lessons."""

from __future__ import annotations

import streamlit as st


def working(lines: list[str]) -> None:
    st.code("\n".join(lines), language="text")
