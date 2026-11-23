"""Unit tests and headless Streamlit smoke tests for the Financial Anomaly Detection lab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP = ROOT / "app"
for p in (SRC, APP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anomaly_lab.autoencoder import NumpyAutoencoder, customer_vs_global_demo, run_autoencoder_demo
from anomaly_lab.data import (
    BENCHMARK_FEATURES,
    HAND_2D_TX,
    HAND_TX_DF,
    make_financial_benchmark,
    make_temporal_sequence,
    make_transaction_graph,
)
from anomaly_lab.evaluation import CORE_MODEL_NAMES, normalize_scores, precision_at_k_table, run_six_model_benchmark
from anomaly_lab.interview_bank import CORE_QA_BY_TOPIC, CROSS_ALGORITHM_SCENARIOS, all_interview_questions
from anomaly_lab.isolation_forest import anomaly_score, expected_path_length, run_2d_isolation_demo
from anomaly_lab.one_class_svm import rbf_similarity, run_2d_ocsvm_demo
from anomaly_lab.statistical import iqr_analysis, mahalanobis_analysis, outlier_masking_demo, zscore_analysis
from anomaly_lab.supporting import (
    run_graph_message_passing_demo,
    run_lof_dense_vs_sparse_demo,
    run_temporal_sequence_demo,
)


def test_zscore_and_outlier_masking() -> None:
    amounts = HAND_TX_DF["amount"].tolist()
    before, after = outlier_masking_demo(amounts, extreme_amount=5000.0, threshold=2.5)
    # T10 ($490) should be flagged before the $5000 spike and masked afterward
    assert bool(before.table.iloc[-1]["is_anomaly"]) is True
    assert float(after.table.iloc[-2]["z_score"]) < float(before.table.iloc[-1]["z_score"])
    assert bool(after.table.iloc[-2]["is_anomaly"]) is False


def test_iqr_robustness_under_extreme_spike() -> None:
    amounts = HAND_TX_DF["amount"].tolist()
    base = iqr_analysis(amounts, k=1.5)
    spiked = iqr_analysis([*amounts, 50000.0], k=1.5)
    # T10 ($490) remains flagged even with a $50k extreme spike
    assert bool(base.table.iloc[-1]["is_anomaly"]) is True
    assert bool(spiked.table.iloc[-2]["is_anomaly"]) is True
    assert bool(spiked.table.iloc[-1]["is_anomaly"]) is True


def test_mahalanobis_reduces_to_euclidean_for_identity_covariance() -> None:
    rng = np.random.default_rng(0)
    # Construct points with exact zero mean and identity covariance
    X = rng.normal(size=(200, 2))
    X = (X - X.mean(axis=0))
    cov = np.cov(X, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    X_white = X @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    res = mahalanobis_analysis(X_white, reg=0.0)
    np.testing.assert_allclose(
        res.table["mahalanobis_dist"].to_numpy(),
        res.table["euclidean_z_dist"].to_numpy(),
        atol=1e-2,
    )


def test_mahalanobis_catches_correlation_break() -> None:
    X = HAND_2D_TX[["amount", "distance_km"]].to_numpy(dtype=float)
    fit_mask = (HAND_2D_TX["label"] == "Normal").to_numpy()
    res = mahalanobis_analysis(X, fit_mask=fit_mask)
    assert bool(res.table.iloc[-1]["is_anomaly"]) is True
    assert float(res.table.iloc[-1]["mahalanobis_dist"]) > float(res.table.iloc[-1]["euclidean_z_dist"])


def test_isolation_forest_scoring_and_demo() -> None:
    c256 = expected_path_length(256)
    assert 10.0 < c256 < 10.6
    # When E[h(x)] == c(n), anomaly score must equal 0.5
    np.testing.assert_allclose(anomaly_score(c256, 256), 0.5, atol=1e-6)
    assert float(anomaly_score(2.0, 256)) > 0.8
    assert float(anomaly_score(20.0, 256)) < 0.3

    demo = run_2d_isolation_demo(n_trees=30, psi=32, contamination=0.07)
    norm_path = demo.points[demo.points["true_role"] == "Normal"]["avg_path_length"].mean()
    anom_path = demo.points[demo.points["true_role"] == "Anomaly"]["avg_path_length"].mean()
    assert anom_path < norm_path


def test_ocsvm_demo_and_rbf() -> None:
    from anomaly_lab.one_class_svm import build_1d_spotlight_demo

    spot = build_1d_spotlight_demo(gamma=1.8, rho=0.45)
    probes = spot.probes.set_index("amount")
    assert probes.loc[52.0, "verdict"] == "Normal (Inside)"
    assert probes.loc[240.0, "verdict"] == "Anomaly (Outside)"

    sim = rbf_similarity(np.array([0.0, 0.0]), np.array([[0.0, 0.0], [5.0, 5.0]]), gamma=0.5)
    assert sim[0] == 1.0 and sim[1] < 1e-6
    res = run_2d_ocsvm_demo(kernel="rbf", nu=0.08, gamma=0.5)
    assert res.n_support_vectors > 0
    # Dual property: SV fraction >= nu (up to finite-sample rounding)
    assert res.sv_fraction >= res.nu - 0.03
    assert int(res.points[res.points["true_role"] == "Anomaly"]["flagged"].sum()) >= 4


def test_autoencoder_trains_and_separates_anomalies() -> None:
    df = make_financial_benchmark()
    run = run_autoencoder_demo(df, BENCHMARK_FEATURES, bottleneck_dim=2, epochs=160, train_normal_only=True)
    # Training loss must decrease
    assert run.loss_history["train_mse"].iloc[-1] < run.loss_history["train_mse"].iloc[0]
    # Anomalies must have substantially higher reconstruction error than normal transactions
    assert run.anomaly_mean_mse > run.normal_mean_mse * 2.5

    cvg = customer_vs_global_demo(df, BENCHMARK_FEATURES)
    assert len(cvg) == 2


def test_supporting_lof_graph_temporal() -> None:
    lof_res = run_lof_dense_vs_sparse_demo(k=6)
    local_score = float(lof_res.points[lof_res.points["tx_id"] == "LOCAL_OUT"]["lof_score"].iloc[0])
    assert local_score > 1.5

    nodes, edges = make_transaction_graph()
    out_nodes, _ = run_graph_message_passing_demo(nodes, edges, steps=2)
    ring_risk = out_nodes[(out_nodes["entity"] == "Customer") & (out_nodes["group"] == "Ring")]["final_risk"].mean()
    norm_risk = out_nodes[(out_nodes["entity"] == "Customer") & (out_nodes["group"] == "Normal")]["final_risk"].mean()
    assert ring_risk > norm_risk

    seq_res = run_temporal_sequence_demo(make_temporal_sequence(), window=4)
    assert "hidden_norm" in seq_res.columns
    burst = seq_res[seq_res["is_seq_anomaly"] == 1]
    assert int(burst["point_flag"].sum()) == 0
    assert int(burst["sequence_flag"].sum()) >= 3


def test_six_model_evaluation_and_interview_bank() -> None:
    df = make_financial_benchmark()
    comp = run_six_model_benchmark(df, contamination=0.05)
    assert set(comp.metrics_df["Algorithm"]) == set(CORE_MODEL_NAMES)
    assert (comp.metrics_df["PR-AUC"] >= 0.0).all() and (comp.metrics_df["PR-AUC"] <= 1.0).all()
    assert (comp.metrics_df["ROC-AUC"] >= 0.5).all() and (comp.metrics_df["ROC-AUC"] <= 1.0).all()

    normed = normalize_scores(comp.scores_df["Autoencoder"].to_numpy(), "percentile")
    assert normed.min() >= 0.0 and normed.max() <= 1.0

    pk = precision_at_k_table(
        df["is_anomaly"].to_numpy(dtype=int),
        {m: comp.scores_df[m].to_numpy(dtype=float) for m in CORE_MODEL_NAMES},
        k=20,
    )
    assert len(pk) == len(CORE_MODEL_NAMES)

    assert len(CROSS_ALGORITHM_SCENARIOS) == 9
    assert len(CORE_QA_BY_TOPIC) >= 7
    assert len(all_interview_questions()) >= 35


def test_streamlit_app_renders_all_ten_lessons() -> None:
    from streamlit.testing.v1 import AppTest

    from app_pages.anomaly_lab import STEPS

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Switch sidebar page to Anomaly detection lab
    at.sidebar.radio(key="nav_page").set_value("Anomaly detection lab").run()
    assert not at.exception

    for step in STEPS:
        at.sidebar.radio(key="anom_step_radio").set_value(step).run()
        assert not at.exception, f"Streamlit exception on step {step!r}: {at.exception}"
