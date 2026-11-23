"""Concise, interview-ready Q&A bank for Financial Anomaly Detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterviewQA:
    topic: str
    q_type: str  # "Conceptual" | "Mathematical" | "Numerical" | "Scenario"
    question: str
    answer: str
    key_point: str


CORE_QA_BY_TOPIC: dict[str, list[InterviewQA]] = {
    "Z-score": [
        InterviewQA(
            topic="Z-score",
            q_type="Conceptual",
            question="What does Z-score assume, and why do transaction amounts break it?",
            answer="It assumes roughly symmetric, bell-shaped data. Financial amounts are heavily right-skewed, so a single mean/std distorts tail cutoffs.",
            key_point="Fix skewness with `log1p(amount)` or switch to median/MAD (Modified Z-score).",
        ),
        InterviewQA(
            topic="Z-score",
            q_type="Mathematical",
            question="Why does one huge fraud spike mask other anomalies (masking effect)?",
            answer="One extreme spike inflates both $\\mu$ and $\\sigma$ (squared deviations), shrinking $z = (x - \\mu)/\\sigma$ for moderate anomalies below the cutoff.",
            key_point="Mean and std have a 0% breakdown point.",
        ),
        InterviewQA(
            topic="Z-score",
            q_type="Numerical",
            question=r"Customer mean = \$60, std = \$15. What is $Z$ for a \$112.50 charge, and does $|Z| > 3$ flag it?",
            answer=r"$z = (112.50 - 60) / 15 = 3.50$. Since $|3.50| > 3.0$, it is flagged (~0.27% Gaussian tail).",
            key_point=r"Benchmarks: $|z| > 2$ (~4.6%), $|z| > 2.5$ (~1.2%), $|z| > 3$ (~0.27%).",
        ),
        InterviewQA(
            topic="Z-score",
            q_type="Scenario",
            question="How do you pick the Z-score threshold in live card alerting?",
            answer="Calibrate to analyst capacity and FP vs FN cost using rolling per-customer `log1p(amount)` Z-scores rather than hardcoding 3.0.",
            key_point="Thresholds are business/capacity decisions, not fixed constants.",
        ),
    ],
    "IQR": [
        InterviewQA(
            topic="IQR",
            q_type="Conceptual",
            question="Why is IQR more robust to extreme fraud spikes than Z-score?",
            answer=r"IQR uses only $Q_1$ and $Q_3$ (middle 50%). Changing the max spend from \$500 to \$500k does not move either fence.",
            key_point="25% breakdown point vs Z-score's 0%.",
        ),
        InterviewQA(
            topic="IQR",
            q_type="Mathematical",
            question=r"Where does the 1.5 multiplier in $Q_3 + 1.5 \cdot \text{IQR}$ come from?",
            answer=r"On a standard Normal, $Q_1 = -0.6745, Q_3 = +0.6745$ ($\text{IQR} = 1.349$), placing fences at $\pm 2.698\sigma$ (~0.7% tail).",
            key_point=r"$k=1.5 \approx \pm 2.7\sigma$; $k=3.0 \approx \pm 4.7\sigma$ for extreme spikes.",
        ),
        InterviewQA(
            topic="IQR",
            q_type="Numerical",
            question=r"If $Q_1 = \$40$ and $Q_3 = \$70$, what are the fences? Is \$120 anomalous?",
            answer=r"$\text{IQR} = 30$. Lower $= 40 - 45 = -\$5$, Upper $= 70 + 45 = \$115$. Since $\$120 > \$115$, it is flagged.",
            key_point="Lower fences often go negative on positive spend.",
        ),
        InterviewQA(
            topic="IQR",
            q_type="Scenario",
            question="When does IQR fail in finance?",
            answer="Univariate only (misses joint anomalies), discrete/sparse columns where $Q_1 = Q_3 \\Rightarrow \\text{IQR} = 0$, and multimodal spending (coffee + rent).",
            key_point="Always guard against $\\text{IQR} = 0$ and multivariate blind spots.",
        ),
    ],
    "Mahalanobis": [
        InterviewQA(
            topic="Mahalanobis",
            q_type="Conceptual",
            question="Why does Euclidean distance fail on correlated features, and how does Mahalanobis fix it?",
            answer="Euclidean draws a scale-sensitive circle ignoring correlation. Mahalanobis divides by $\\Sigma$, turning the boundary into a correlation-aligned ellipse.",
            key_point="Mahalanobis = Euclidean distance after whitening (decorrelating + scaling).",
        ),
        InterviewQA(
            topic="Mahalanobis",
            q_type="Mathematical",
            question="What happens if $\\Sigma$ is singular (collinear features or $n < p$)?",
            answer="$\\Sigma^{-1}$ does not exist (det $= 0$). Fix with ridge/Ledoit-Wolf shrinkage ($\\Sigma + \\lambda I$), PCA, or dropping redundant columns.",
            key_point="Always mention shrinkage ($\\Sigma + \\lambda I$) or PCA for singular covariance.",
        ),
        InterviewQA(
            topic="Mahalanobis",
            q_type="Numerical",
            question="If $\\Sigma = \\text{diag}(4, 9)$ (uncorrelated), what does $D_M(x)$ simplify to?",
            answer="$D_M(x) = \\sqrt{(x_1-\\mu_1)^2/4 + (x_2-\\mu_2)^2/9} = \\sqrt{z_1^2 + z_2^2}$ — standardized Euclidean distance.",
            key_point="Diagonal $\\Sigma \\Rightarrow$ standardized Euclidean; $\\Sigma = I \\Rightarrow$ raw Euclidean.",
        ),
        InterviewQA(
            topic="Mahalanobis",
            q_type="Scenario",
            question="How do you set the Mahalanobis cutoff, and what is the training pitfall?",
            answer="Under normality, $D_M^2 \\sim \\chi^2_p$, so use $\\sqrt{\\chi^2_{p, 0.975}}$. Pitfall: contaminated data stretches $\\Sigma$ — use robust MCD or clean normal rows.",
            key_point="$D_M^2 \\sim \\chi^2_p$ for threshold; MCD for robust covariance.",
        ),
    ],
    "Isolation Forest": [
        InterviewQA(
            topic="Isolation Forest",
            q_type="Conceptual",
            question="Why do anomalies have shorter path lengths than normal points?",
            answer="Anomalies sit alone in sparse regions, so 1-2 random feature splits isolate them immediately; dense normal clusters need many cuts.",
            key_point="Isolates anomalies directly without density estimation.",
        ),
        InterviewQA(
            topic="Isolation Forest",
            q_type="Mathematical",
            question="Why divide by $c(n)$ in $s(x, n) = 2^{-E[h(x)] / c(n)}$, and how do you read $s$?",
            answer="$c(n)$ is the average BST path length (via harmonic numbers) so scores normalize across subsample size $n$. $s \\to 1$: anomaly; $s \\approx 0.5$: normal; $s \\to 0$: dense core.",
            key_point="$E[h] = c(n) \\Rightarrow s = 0.5$; shorter path $\\Rightarrow s \\to 1$.",
        ),
        InterviewQA(
            topic="Isolation Forest",
            q_type="Conceptual",
            question="Why use small subsamples ($\\psi = 256$) instead of all 10M rows per tree?",
            answer="Two wins: (1) $O(T \\psi \\log \\psi)$ speed independent of $N$; (2) reduces **swamping** (normal points crowding outliers) and **masking** (clustered fraud hiding itself).",
            key_point="Small $\\psi = 256$ improves BOTH speed and accuracy.",
        ),
        InterviewQA(
            topic="Isolation Forest",
            q_type="Numerical",
            question="With $c(256) \\approx 10.24$, compute $s$ for $E[h] = 3.5$ vs $E[h] = 11.0$.",
            answer="For $3.5$: $s = 2^{-3.5/10.24} \\approx 0.79$ (anomaly). For $11.0$: $s = 2^{-11.0/10.24} \\approx 0.47$ (normal).",
            key_point="Short path ($< c(n)$) gives $s > 0.5$; long path gives $s < 0.5$.",
        ),
        InterviewQA(
            topic="Isolation Forest",
            q_type="Scenario",
            question="What does `contamination` change in sklearn, and what is a key weakness of Isolation Forest?",
            answer="`contamination` only shifts the `predict()` cutoff percentile — trees never change. Weakness: axis-aligned cuts create rectangular ghost zones on diagonals (fixed by Extended Isolation Forest).",
            key_point="Contamination = cutoff only; axis-aligned cuts struggle on diagonal/local clusters.",
        ),
    ],
    "One-Class SVM": [
        InterviewQA(
            topic="One-Class SVM",
            q_type="Conceptual",
            question="How does One-Class SVM work without fraud labels, and what are support vectors?",
            answer="Instead of separating two classes, it learns the tightest smooth fence around normal data. Support vectors are the few boundary points ('fence posts') that define the fence — interior points get weight 0 and are ignored at scoring time.",
            key_point="Support vectors = fence posts; new points are scored only by similarity to them.",
        ),
        InterviewQA(
            topic="One-Class SVM",
            q_type="Mathematical",
            question="What are the two roles of $\\nu$ in One-Class SVM?",
            answer="$\\nu \\in (0, 1]$ is simultaneously an **upper bound** on the training outlier fraction and a **lower bound** on the support vector fraction.",
            key_point="Memorize: $\\text{train outliers} \\le \\nu \\le \\text{SV fraction}$.",
        ),
        InterviewQA(
            topic="One-Class SVM",
            q_type="Conceptual",
            question="What happens in an RBF One-Class SVM when $\\gamma$ is too large vs too small?",
            answer="Large $\\gamma$: tiny spiky spotlights around each training point (overfitting, high false positives). Small $\\gamma$: overly wide loose blanket (underfitting, missed fraud).",
            key_point="$\\gamma$ controls spotlight width in $K(x, s) = \\exp(-\\gamma \\|x - s\\|^2)$.",
        ),
        InterviewQA(
            topic="One-Class SVM",
            q_type="Numerical",
            question="With $\\gamma = 0.5$, compare RBF similarity at distance $0.2$ vs $4.0$.",
            answer="$\\exp(-0.5 \\cdot 0.04) = 0.980$ vs $\\exp(-0.5 \\cdot 16) = 0.0003$. Far anomalies get $K \\approx 0 \\Rightarrow f(x) \\approx -\\rho < 0$ (flagged).",
            key_point="RBF warmth drops exponentially with squared distance from support vectors.",
        ),
        InterviewQA(
            topic="One-Class SVM",
            q_type="Scenario",
            question="Why is One-Class SVM rarely run directly on 5M daily transactions?",
            answer="Kernel training scales $O(n^2)$-$O(n^3)$ and requires strict scaling + clean normal data. Use Nystroem + SGD OC-SVM, or switch to Isolation Forest / Autoencoders.",
            key_point="Main limits: $O(n^2)$-$O(n^3)$ scaling + mandatory feature scaling.",
        ),
    ],
    "Autoencoders": [
        InterviewQA(
            topic="Autoencoders",
            q_type="Conceptual",
            question="Why do we need a bottleneck and normal-only training?",
            answer="A tight bottleneck ($k < d$) blocks identity copying. Normal-only training keeps fraud off the manifold, raising MSE.",
            key_point="Compression learns normal structure; normal-only training isolates fraud.",
        ),
        InterviewQA(
            topic="Autoencoders",
            q_type="Mathematical",
            question="Write reconstruction MSE and the weight update rule.",
            answer=r"$\mathcal{L} = \frac{1}{d}\sum (x_j - \hat{x}_j)^2$. Gradients flow back via chain rule: $W \leftarrow W - \eta \nabla_W \mathcal{L}$.",
            key_point="MSE gives both the anomaly score and per-feature explainability.",
        ),
        InterviewQA(
            topic="Autoencoders",
            q_type="Numerical",
            question=r"$x = [0.2, 3.0, 0.1]$, $\hat{x} = [0.3, 0.4, 0.2]$. Find MSE and top driver.",
            answer=r"Squared errors $= [0.01, 6.76, 0.01] \Rightarrow \text{MSE} = 6.78 / 3 = 2.26$. Feature 2 drives 99.7%.",
            key_point=r"Per-feature $(x_j - \hat{x}_j)^2$ gives instant analyst reason codes.",
        ),
        InterviewQA(
            topic="Autoencoders",
            q_type="Scenario",
            question="What happens if the bottleneck is too wide, and how do you fix it?",
            answer="The model copies inputs blindly, giving fraud low MSE. Fix with smaller $k$, L2/dropout, Denoising/VAE, or early stopping.",
            key_point="Over-capacity $\\Rightarrow$ identity copy $\\Rightarrow$ silent false negatives.",
        ),
        InterviewQA(
            topic="Autoencoders",
            q_type="Scenario",
            question="Customer-level vs global autoencoder — which scales in production?",
            answer="Global mixes spend tiers; per-user models fail cold-start. Feed customer-relative ratios into one shared model.",
            key_point="Customer-relative features inside a shared model solve scale and personalization.",
        ),
    ],
    "Supporting (LOF, Graph, Temporal)": [
        InterviewQA(
            topic="Supporting (LOF, Graph, Temporal)",
            q_type="Conceptual",
            question="When does LOF beat global detectors like Z-score or Isolation Forest?",
            answer="When normal clusters have different densities. LOF compares local density to $k$ neighbors, flagging $\\text{LOF} \\gg 1$.",
            key_point="$\\text{LOF} \\approx 1$ normal; $\\text{LOF} \\gg 1$ local outlier.",
        ),
        InterviewQA(
            topic="Supporting (LOF, Graph, Temporal)",
            q_type="Numerical",
            question="Point $p$ has $\\text{lrd}(p) = 0.20$; its $k$ neighbors average $\\text{lrd} = 0.80$. Find LOF.",
            answer="$\\text{LOF}(p) = 0.80 / 0.20 = 4.0$. Since $4.0 > 1.5$, $p$ is a strong local outlier.",
            key_point="LOF is simply neighbor density divided by your own density.",
        ),
        InterviewQA(
            topic="Supporting (LOF, Graph, Temporal)",
            q_type="Scenario",
            question="When do tabular detectors miss fraud that a Graph/GNN catches?",
            answer=r"Mule rings and synthetic IDs: ordinary \$90 transfers all linked to one shared device or merchant.",
            key_point="Graphs catch collusive links invisible in single rows.",
        ),
        InterviewQA(
            topic="Supporting (LOF, Graph, Temporal)",
            q_type="Mathematical",
            question="How does GNN message passing work, and why limit hops to 1-3?",
            answer="Each hop blends a node's features with degree-normalized neighbor features. Too many hops cause over-smoothing.",
            key_point="1-3 hops capture fraud rings without blurring node risk.",
        ),
        InterviewQA(
            topic="Supporting (LOF, Graph, Temporal)",
            q_type="Scenario",
            question="When do temporal models (LSTM / Transformer) add value?",
            answer=r"When event order or timing breaks normal habits (e.g., five \$90 buys 2 min apart at 2 AM).",
            key_point="Sequence models flag broken trajectories, not just big amounts.",
        ),
    ],
}


CROSS_ALGORITHM_SCENARIOS: list[InterviewQA] = [
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="1. Isolation Forest vs One-Class SVM: when and why?",
        answer="**Isolation Forest** for large tabular data ($O(T\\psi\\log\\psi)$, no scaling, tolerates slight noise). **One-Class SVM (RBF)** for clean medium data ($<20\\text{k}$) needing a tight curved boundary.",
        key_point="Scale & speed $\\to$ Isolation Forest; tight smooth boundary on clean medium data $\\to$ OC-SVM.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="2. Z-score vs IQR vs Mahalanobis: which should you choose?",
        answer="**Z-score:** 1D symmetric (or after `log1p`). **IQR:** 1D skewed or spike-contaminated. **Mahalanobis:** multi-feature correlated data.",
        key_point="1D symmetric $\\to$ Z-score; 1D skewed/noisy $\\to$ IQR; multi-D correlated $\\to$ Mahalanobis.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Conceptual",
        question="3. Why can an autoencoder reconstruct an anomalous transaction well?",
        answer="(1) Over-capacity / wide bottleneck (identity copy); (2) contaminated training data containing fraud; (3) anomaly lies on another segment's normal manifold.",
        key_point="Tight bottleneck + clean normal data + customer-relative features.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="4. How would you detect unusual spending for an individual customer?",
        answer="Compute rolling per-customer baselines (median/MAD of `log1p(amount)`, usual merchants/hours), shrink cold-start users to peer segments, and feed relative ratios into Isolation Forest or Autoencoder.",
        key_point="Customer-relative features + peer fallback beat 10M separate models.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="5. How would you handle highly skewed transaction amounts?",
        answer="Apply `log1p(amount)` or Yeo-Johnson before Z-score, Mahalanobis, OC-SVM, or Autoencoders; or use rank/quartile methods (IQR, Modified Z-score, Isolation Forest).",
        key_point="`log1p(amount)` + robust scaling before distance or neural models.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="6. How do you detect & evaluate anomalies with very few fraud labels?",
        answer="Train unsupervised/normal-only, then measure **Precision@K** on analyst-reviewed top alerts, run **synthetic anomaly injection**, and track alert-rate stability.",
        key_point="Top-K analyst review (Precision@K) + synthetic anomaly injection.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Mathematical",
        question="7. How do you compare models whose anomaly scores are on different scales?",
        answer="Align direction (higher = riskier), convert each score to **empirical percentile ranks** $[0, 1]$ (ECDF) on a normal reference set, and compare rank metrics (**PR-AUC** / ROC-AUC).",
        key_point="Align sign $\\to$ empirical percentile rank (ECDF) before blending.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Conceptual",
        question="8. Can an anomaly be legitimate? Can fraud look completely normal?",
        answer=r"Yes to both — **anomaly $\neq$ fraud**. Big travel/gifts are legitimate anomalies (use step-up 2FA); a stolen card buying \$45 groceries looks normal alone (needs device/graph/velocity signals).",
        key_point="Match friction to confidence: 2FA for rare anomalies, hard block for high-confidence fraud.",
    ),
    InterviewQA(
        topic="Cross-algorithm & Scenarios",
        q_type="Scenario",
        question="9. How would you combine multiple anomaly detection techniques?",
        answer="Fast statistical rules (ms) + diverse unsupervised detectors (Isolation Forest + Autoencoder + Graph) converted to percentile ranks, then blend or pass them as **features** into the supervised fraud model.",
        key_point="Percentile-normalized ensemble + feed anomaly scores into the supervised model.",
    ),
]


def all_interview_questions() -> list[InterviewQA]:
    """Return every interview Q&A across all topics plus cross-algorithm scenarios."""
    out: list[InterviewQA] = []
    for items in CORE_QA_BY_TOPIC.values():
        out.extend(items)
    out.extend(CROSS_ALGORITHM_SCENARIOS)
    return out
