"""Shared display helpers for the DIET lessons."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from st_compat import image

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "diet"


def figure(name: str) -> None:
    image(str(ASSETS / name))


def working(lines: list[str]) -> None:
    st.code("\n".join(lines), language="text")
