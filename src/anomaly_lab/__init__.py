"""AI/ML Financial Anomaly Detection teaching and interview-prep engine."""

from __future__ import annotations

from anomaly_lab.data import (
    BENCHMARK_FEATURES,
    HAND_2D_TX,
    HAND_TX_DF,
    make_financial_benchmark,
    make_temporal_sequence,
    make_transaction_graph,
)

__all__ = [
    "BENCHMARK_FEATURES",
    "HAND_2D_TX",
    "HAND_TX_DF",
    "make_financial_benchmark",
    "make_temporal_sequence",
    "make_transaction_graph",
]
