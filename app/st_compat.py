"""Streamlit helpers that work on 1.32 (session conda) and newer versions."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import streamlit as st


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _plotly_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(st.plotly_chart).parameters
    out = dict(kwargs)
    if "width" in params:
        out.setdefault("width", "stretch")
        out.pop("use_container_width", None)
    else:
        out.pop("width", None)
        out["use_container_width"] = True
    return out


def plotly_chart(fig: Any, **kwargs: Any) -> None:
    st.plotly_chart(fig, **_plotly_kwargs(kwargs))


def plotly_in(slot: Any, fig: Any, **kwargs: Any) -> None:
    slot.plotly_chart(fig, **_plotly_kwargs(kwargs))


def _dataframe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(st.dataframe).parameters
    out = dict(kwargs)
    width = out.get("width")
    if width == "stretch" or (width is not None and not isinstance(width, int)):
        out.pop("width", None)
        if "use_container_width" in params:
            out["use_container_width"] = True
    if "hide_index" not in params:
        out.pop("hide_index", None)
    return out


def _arrow_safe(data: Any) -> Any:
    """Stringify mixed object columns so Streamlit/Arrow can serialize the table."""
    if not isinstance(data, pd.DataFrame):
        return data
    out = data.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    return out


def dataframe(data: Any, **kwargs: Any) -> None:
    st.dataframe(_arrow_safe(data), **_dataframe_kwargs(kwargs))


def dataframe_in(slot: Any, data: Any, **kwargs: Any) -> None:
    slot.dataframe(_arrow_safe(data), **_dataframe_kwargs(kwargs))


def mermaid_chart(spec: str, **kwargs: Any) -> None:
    if hasattr(st, "mermaid_chart"):
        params = inspect.signature(st.mermaid_chart).parameters
        if "width" in params:
            kwargs.setdefault("width", "stretch")
        st.mermaid_chart(spec, **kwargs)
        return
    st.markdown(f"```mermaid\n{spec.strip()}\n```")


def metric(label: str, value: Any, **kwargs: Any) -> None:
    if not _accepts(st.metric, "border"):
        kwargs.pop("border", None)
    st.metric(label, value, **kwargs)


def button(label: str, **kwargs: Any) -> bool:
    if not _accepts(st.button, "icon"):
        kwargs.pop("icon", None)
    if not _accepts(st.button, "use_container_width"):
        kwargs.pop("use_container_width", None)
    return bool(st.button(label, **kwargs))


def form_submit_button(label: str, **kwargs: Any) -> bool:
    if not _accepts(st.form_submit_button, "icon"):
        kwargs.pop("icon", None)
    if not _accepts(st.form_submit_button, "use_container_width"):
        kwargs.pop("use_container_width", None)
    return bool(st.form_submit_button(label, **kwargs))


def choice(label: str, options: list[str], *, default: str | None = None, key: str, horizontal: bool = False) -> str:
    """segmented_control when available, otherwise radio."""
    if hasattr(st, "segmented_control"):
        kwargs: dict[str, Any] = {"key": key}
        if _accepts(st.segmented_control, "default") and default is not None:
            kwargs["default"] = default
        picked = st.segmented_control(label, options=options, **kwargs)
        return picked or default or options[0]
    index = options.index(default) if default in options else 0
    return str(st.radio(label, options, index=index, key=key, horizontal=horizontal))
