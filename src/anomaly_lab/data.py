"""Deterministic financial transaction fixtures for anomaly detection lessons."""

from __future__ import annotations

import numpy as np
import pandas as pd

BENCHMARK_FEATURES = [
    "amount",
    "tx_velocity_24h",
    "distance_km",
    "hour_of_day",
    "merchant_risk",
]

# Small 10-transaction table for exact hand-calculable Z-score / IQR walkthroughs.
HAND_TX_DF = pd.DataFrame(
    [
        {"tx_id": "T01", "merchant": "Coffee Roasters", "amount": 42.0, "hour": 8, "note": "Normal"},
        {"tx_id": "T02", "merchant": "Fresh Market", "amount": 55.0, "hour": 12, "note": "Normal"},
        {"tx_id": "T03", "merchant": "City Transit", "amount": 48.0, "hour": 9, "note": "Normal"},
        {"tx_id": "T04", "merchant": "Pharmacy Plus", "amount": 63.0, "hour": 17, "note": "Normal"},
        {"tx_id": "T05", "merchant": "Bistro Nine", "amount": 51.0, "hour": 19, "note": "Normal"},
        {"tx_id": "T06", "merchant": "Bookshop", "amount": 59.0, "hour": 15, "note": "Normal"},
        {"tx_id": "T07", "merchant": "Fuel Station", "amount": 46.0, "hour": 11, "note": "Normal"},
        {"tx_id": "T08", "merchant": "Bakery Co", "amount": 54.0, "hour": 10, "note": "Normal"},
        {"tx_id": "T09", "merchant": "Streaming Hub", "amount": 61.0, "hour": 20, "note": "Normal"},
        {"tx_id": "T10", "merchant": "Luxury Electronics", "amount": 490.0, "hour": 2, "note": "Suspected fraud"},
    ]
)

# Small 2D correlated dataset: monthly card spend ($) rises with credit limit ($k).
# T12 sits inside each individual feature's range but violates the joint correlation.
HAND_2D_TX = pd.DataFrame(
    [
        {"tx_id": "M01", "amount": 120.0, "distance_km": 4.0, "label": "Normal"},
        {"tx_id": "M02", "amount": 180.0, "distance_km": 6.5, "label": "Normal"},
        {"tx_id": "M03", "amount": 240.0, "distance_km": 8.0, "label": "Normal"},
        {"tx_id": "M04", "amount": 310.0, "distance_km": 10.5, "label": "Normal"},
        {"tx_id": "M05", "amount": 370.0, "distance_km": 12.0, "label": "Normal"},
        {"tx_id": "M06", "amount": 430.0, "distance_km": 14.5, "label": "Normal"},
        {"tx_id": "M07", "amount": 500.0, "distance_km": 16.5, "label": "Normal"},
        {"tx_id": "M08", "amount": 560.0, "distance_km": 18.0, "label": "Normal"},
        {"tx_id": "M09", "amount": 620.0, "distance_km": 20.5, "label": "Normal"},
        {"tx_id": "M10", "amount": 690.0, "distance_km": 22.5, "label": "Normal"},
        {"tx_id": "M11", "amount": 410.0, "distance_km": 13.5, "label": "Normal"},
        {"tx_id": "M12", "amount": 640.0, "distance_km": 3.5, "label": "Correlation anomaly"},
    ]
)


def make_financial_benchmark(n_normal: int = 460, n_anomalies: int = 24, seed: int = 42) -> pd.DataFrame:
    """Create a realistic multi-customer financial transaction benchmark.

    Normal transactions come from two customer tiers:
    - Everyday spenders (~70%): $20-$140, low velocity, local distance, daytime hours.
    - Affluent travelers (~30%): $180-$520, moderate velocity, wider distance, varied hours.

    Anomalies span five realistic financial patterns so every algorithm's
    strengths and blind spots show up clearly during model comparison.
    """
    rng = np.random.default_rng(seed)

    n_everyday = int(n_normal * 0.70)
    n_affluent = n_normal - n_everyday

    # Everyday customer cluster
    ev_amount = rng.lognormal(mean=4.05, sigma=0.35, size=n_everyday)  # ~$40-$120
    ev_vel = np.clip(rng.normal(2.2, 0.9, size=n_everyday), 0.5, 6.0)
    ev_dist = np.clip(ev_amount * 0.08 + rng.normal(3.0, 2.0, size=n_everyday), 0.5, 25.0)
    ev_hour = np.clip(rng.normal(14.5, 3.2, size=n_everyday), 7.0, 22.0)
    ev_risk = np.clip(rng.normal(0.18, 0.08, size=n_everyday), 0.02, 0.48)
    ev_cust = [f"C{101 + (i % 6)}" for i in range(n_everyday)]

    # Affluent customer cluster (higher baseline spend + travel)
    af_amount = rng.lognormal(mean=5.65, sigma=0.28, size=n_affluent)  # ~$210-$450
    af_vel = np.clip(rng.normal(3.8, 1.1, size=n_affluent), 1.0, 8.0)
    af_dist = np.clip(af_amount * 0.09 + rng.normal(10.0, 4.5, size=n_affluent), 5.0, 70.0)
    af_hour = np.clip(rng.normal(16.0, 3.5, size=n_affluent), 8.0, 23.0)
    af_risk = np.clip(rng.normal(0.24, 0.09, size=n_affluent), 0.05, 0.52)
    af_cust = [f"C{201 + (i % 4)}" for i in range(n_affluent)]

    normal_df = pd.DataFrame(
        {
            "customer_id": ev_cust + af_cust,
            "customer_tier": ["Everyday"] * n_everyday + ["Affluent"] * n_affluent,
            "amount": np.concatenate([ev_amount, af_amount]),
            "tx_velocity_24h": np.concatenate([ev_vel, af_vel]),
            "distance_km": np.concatenate([ev_dist, af_dist]),
            "hour_of_day": np.concatenate([ev_hour, af_hour]),
            "merchant_risk": np.concatenate([ev_risk, af_risk]),
            "is_anomaly": 0,
            "anomaly_type": "Normal",
        }
    )

    # Injected financial anomalies across 4 archetypes
    per_type = max(1, n_anomalies // 4)
    rem = n_anomalies - 4 * per_type

    # 1. High-amount account takeover
    n1 = per_type + rem
    a1 = pd.DataFrame(
        {
            "customer_id": [f"C{101 + (i % 6)}" for i in range(n1)],
            "customer_tier": "Everyday",
            "amount": rng.uniform(1100, 2400, size=n1),
            "tx_velocity_24h": rng.uniform(3.5, 8.0, size=n1),
            "distance_km": rng.uniform(45, 160, size=n1),
            "hour_of_day": rng.uniform(0.5, 4.5, size=n1),
            "merchant_risk": rng.uniform(0.68, 0.95, size=n1),
            "is_anomaly": 1,
            "anomaly_type": "High-amount takeover",
        }
    )

    # 2. Card-testing velocity burst (small amounts, huge velocity, high merchant risk)
    a2 = pd.DataFrame(
        {
            "customer_id": [f"C{101 + (i % 6)}" for i in range(per_type)],
            "customer_tier": "Everyday",
            "amount": rng.uniform(15, 65, size=per_type),
            "tx_velocity_24h": rng.uniform(13.0, 22.0, size=per_type),
            "distance_km": rng.uniform(2, 18, size=per_type),
            "hour_of_day": rng.uniform(1.0, 5.0, size=per_type),
            "merchant_risk": rng.uniform(0.70, 0.96, size=per_type),
            "is_anomaly": 1,
            "anomaly_type": "Velocity burst",
        }
    )

    # 3. Correlation break (normal-ish amount, extreme distance + high merchant risk)
    a3 = pd.DataFrame(
        {
            "customer_id": [f"C{201 + (i % 4)}" for i in range(per_type)],
            "customer_tier": "Affluent",
            "amount": rng.uniform(80, 190, size=per_type),
            "tx_velocity_24h": rng.uniform(8.5, 14.0, size=per_type),
            "distance_km": rng.uniform(135, 240, size=per_type),
            "hour_of_day": rng.uniform(1.5, 4.5, size=per_type),
            "merchant_risk": rng.uniform(0.62, 0.90, size=per_type),
            "is_anomaly": 1,
            "anomaly_type": "Geo + correlation break",
        }
    )

    # 4. Everyday customer spending at affluent-plus level at 3 AM
    a4 = pd.DataFrame(
        {
            "customer_id": [f"C{101 + (i % 6)}" for i in range(per_type)],
            "customer_tier": "Everyday",
            "amount": rng.uniform(520, 880, size=per_type),
            "tx_velocity_24h": rng.uniform(9.5, 15.0, size=per_type),
            "distance_km": rng.uniform(90, 170, size=per_type),
            "hour_of_day": rng.uniform(0.5, 3.8, size=per_type),
            "merchant_risk": rng.uniform(0.72, 0.95, size=per_type),
            "is_anomaly": 1,
            "anomaly_type": "Off-hours profile jump",
        }
    )

    df = pd.concat([normal_df, a1, a2, a3, a4], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "tx_id", [f"TX{i + 1:04d}" for i in range(len(df))])
    for col in BENCHMARK_FEATURES:
        df[col] = df[col].round(2)
    return df


def make_transaction_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small entity graph containing normal activity and a shared-device fraud ring."""
    nodes = pd.DataFrame(
        [
            # Normal cluster (left)
            {"id": "Cust_A", "entity": "Customer", "x": 1.0, "y": 3.4, "base_risk": 0.10, "group": "Normal"},
            {"id": "Dev_1", "entity": "Device", "x": 2.2, "y": 3.4, "base_risk": 0.08, "group": "Normal"},
            {"id": "Merch_Grocer", "entity": "Merchant", "x": 3.5, "y": 3.4, "base_risk": 0.12, "group": "Normal"},
            {"id": "Cust_B", "entity": "Customer", "x": 1.0, "y": 2.1, "base_risk": 0.12, "group": "Normal"},
            {"id": "Dev_2", "entity": "Device", "x": 2.2, "y": 2.1, "base_risk": 0.10, "group": "Normal"},
            # Suspicious mule ring (right/bottom) sharing one device & high-risk merchant
            {"id": "Cust_X", "entity": "Customer", "x": 5.4, "y": 3.5, "base_risk": 0.35, "group": "Ring"},
            {"id": "Cust_Y", "entity": "Customer", "x": 5.4, "y": 2.3, "base_risk": 0.32, "group": "Ring"},
            {"id": "Cust_Z", "entity": "Customer", "x": 5.4, "y": 1.1, "base_risk": 0.38, "group": "Ring"},
            {"id": "Dev_Shared", "entity": "Device", "x": 7.0, "y": 2.3, "base_risk": 0.85, "group": "Ring"},
            {"id": "Merch_Crypto", "entity": "Merchant", "x": 8.6, "y": 2.3, "base_risk": 0.90, "group": "Ring"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "Cust_A", "target": "Dev_1", "relation": "uses_device"},
            {"source": "Cust_A", "target": "Merch_Grocer", "relation": "pays"},
            {"source": "Cust_B", "target": "Dev_2", "relation": "uses_device"},
            {"source": "Cust_B", "target": "Merch_Grocer", "relation": "pays"},
            # Ring: 3 unrelated accounts share one device and funnel funds to one merchant
            {"source": "Cust_X", "target": "Dev_Shared", "relation": "uses_device"},
            {"source": "Cust_Y", "target": "Dev_Shared", "relation": "uses_device"},
            {"source": "Cust_Z", "target": "Dev_Shared", "relation": "uses_device"},
            {"source": "Cust_X", "target": "Merch_Crypto", "relation": "pays"},
            {"source": "Cust_Y", "target": "Merch_Crypto", "relation": "pays"},
            {"source": "Cust_Z", "target": "Merch_Crypto", "relation": "pays"},
            {"source": "Dev_Shared", "target": "Merch_Crypto", "relation": "checkout_origin"},
        ]
    )
    return nodes, edges


def make_temporal_sequence(seed: int = 42) -> pd.DataFrame:
    """Create a 32-step customer sequence with an anomalous behavioral burst window.

    Steps 22-26 have completely ordinary dollar amounts ($72-$105) so a single-transaction
    Z-score on amount misses them, yet the sequence breaks temporal rhythm: 2-3 AM hours
    and rapid 3-minute gaps between consecutive transactions.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(1, 33)
    amounts = rng.normal(78.0, 14.0, size=32).clip(35.0, 125.0)
    hours = rng.normal(15.0, 2.5, size=32).clip(9.0, 21.0)
    gap_mins = rng.normal(420.0, 95.0, size=32).clip(150.0, 720.0)
    is_seq_anomaly = np.zeros(32, dtype=int)

    # Inject a 5-transaction sequence anomaly at steps 22..26 (indices 21..25)
    burst = slice(21, 26)
    amounts[burst] = np.array([88.0, 92.0, 95.0, 91.0, 98.0])  # looks normal alone!
    hours[burst] = np.array([2.1, 2.2, 2.25, 2.3, 2.4])
    gap_mins[burst] = np.array([4.0, 3.0, 2.5, 3.5, 4.0])
    is_seq_anomaly[burst] = 1

    return pd.DataFrame(
        {
            "step": steps,
            "amount": np.round(amounts, 2),
            "hour_of_day": np.round(hours, 2),
            "gap_minutes": np.round(gap_mins, 1),
            "is_seq_anomaly": is_seq_anomaly,
            "status": np.where(is_seq_anomaly == 1, "Anomalous sequence", "Normal rhythm"),
        }
    )
