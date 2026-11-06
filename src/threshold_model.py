import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Dict, Any, Optional, Tuple


def fit_evt_threshold(
    scores: np.ndarray,
    initial_quantile: float = 0.90,
    risk_level: float = 0.02,
) -> float:
    """
    Fits Extreme Value Theory (EVT) Peaks-Over-Threshold model using a Generalized Pareto
    Distribution (GPD) on tail exceedances.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of continuous anomaly scores in [0, 1].
    initial_quantile : float, default 0.90
        Quantile u defining the start of the heavy tail.
    risk_level : float, default 0.02
        Target tail probability (unconditional risk).
        
    Returns
    -------
    float
        Statistically justified threshold tau*.
    """
    s = np.asarray(scores, dtype=float).flatten()
    s = s[~np.isnan(s)]
    
    u = float(np.percentile(s, initial_quantile * 100.0))
    exceedances = s[s > u] - u
    n_total = len(s)
    n_u = len(exceedances)
    
    if n_u < 10:
        # Fallback if too few tail samples
        return float(np.percentile(s, (1.0 - risk_level) * 100.0))
        
    try:
        # Fit GPD: c is shape (xi), scale is beta, loc fixed at 0
        c_shape, loc, scale = stats.genpareto.fit(exceedances, floc=0.0)
        
        # Unconditional tail probability: P(S > tau) = (n_u / n_total) * P(Y > tau - u) = risk_level
        # P(Y > tau - u) = risk_level * (n_total / n_u)
        conditional_tail_prob = risk_level * (n_total / n_u)
        conditional_tail_prob = max(1e-6, min(0.999, conditional_tail_prob))
        
        y_tau = float(stats.genpareto.ppf(1.0 - conditional_tail_prob, c_shape, loc=0.0, scale=scale))
        tau = u + y_tau
    except Exception:
        # Robust fallback
        tau = float(np.percentile(s, (1.0 - risk_level) * 100.0))
        
    # Boundary clamps: must be >= u and < max(s)
    max_val = float(np.max(s))
    min_val = u
    tau = max(min_val, min(tau, max_val - 1e-4))
    return float(tau)


def fit_gmm_threshold(
    scores: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> float:
    """
    Fits a 2-component Gaussian Mixture Model (Normal vs Anomalous) and finds the
    decision boundary where posterior probabilities cross.
    """
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    s = s[~np.isnan(s)].reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(s)
    
    # Sort components by mean: 0 = normal (lower), 1 = anomalous (higher)
    means_flat = gmm.means_.flatten()
    order = np.argsort(means_flat)
    
    mu0 = float(means_flat[order[0]])
    mu1 = float(means_flat[order[1]])
    
    covs = gmm.covariances_.reshape(n_components, -1)
    sigma0 = float(np.sqrt(covs[order[0]][0]))
    sigma1 = float(np.sqrt(covs[order[1]][0]))
    
    weights = gmm.weights_.flatten()
    pi0 = float(weights[order[0]])
    pi1 = float(weights[order[1]])
    
    # Search for crossing point in [mu0, mu1]
    xs = np.linspace(mu0, mu1, 1000)
    pdf0 = pi0 * stats.norm.pdf(xs, mu0, sigma0)
    pdf1 = pi1 * stats.norm.pdf(xs, mu1, sigma1)
    diff = pdf1 - pdf0
    
    # Zero crossing index
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        tau = float(xs[idx])
    else:
        # Fallback to midway or equal density approximation
        tau = float((mu0 * sigma1 + mu1 * sigma0) / (sigma0 + sigma1))
        
    return tau


def gaussian_closed_form(
    mu: float,
    sigma: float,
    c_false_alarm: float,
    c_missed: float,
) -> float:
    """
    Bayesian critical fractile closed-form threshold for Gaussian normal baseline:
    tau* = mu + sigma * Phi^{-1}( C_missed / (C_false_alarm + C_missed) )
    """
    fractile = c_missed / (c_false_alarm + c_missed)
    z_star = float(stats.norm.ppf(fractile))
    tau = mu + sigma * z_star
    return float(tau)


def fit_cost_optimal_threshold(
    scores: np.ndarray,
    c_false_alarm: float = 50.0,
    c_missed: float = 200.0,
    n_grid: int = 500,
) -> float:
    """
    Numerically derives the cost-minimizing threshold tau* using soft GMM posteriors
    as probabilistic ground truth.
    
    Total Cost = C_fa * FPR(tau) + C_miss * FNR(tau)
    """
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    s = s[~np.isnan(s)].reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(s)
    means_flat = gmm.means_.flatten()
    order = np.argsort(means_flat)
    
    # Probability of being anomaly (component 1)
    posteriors = gmm.predict_proba(s)[:, order[1]]
    
    s_flat = s.flatten()
    s_min, s_max = float(np.min(s_flat)), float(np.max(s_flat))
    candidates = np.linspace(s_min, s_max, n_grid)
    
    norm_weight_sum = np.sum(1.0 - posteriors) + 1e-8
    anom_weight_sum = np.sum(posteriors) + 1e-8
    
    best_cost = float("inf")
    best_tau = float(candidates[n_grid // 2])
    
    for tau in candidates:
        # Predicted anomaly if s > tau
        pred_anom = s_flat > tau
        # False alarm: s > tau when actually normal
        fpr = np.sum((1.0 - posteriors)[pred_anom]) / norm_weight_sum
        # Missed: s <= tau when actually anomaly
        fnr = np.sum(posteriors[~pred_anom]) / anom_weight_sum
        
        cost = c_false_alarm * fpr + c_missed * fnr
        if cost < best_cost:
            best_cost = cost
            best_tau = tau
            
    return float(best_tau)


def apply_dual_signal_gate(
    df: pd.DataFrame,
    tau: float,
    rank_threshold: float = 0.25,
    score_col: str = "ANOMALY_SCORE",
    rank_col: str = "PEER_RANK_PCT",
) -> pd.DataFrame:
    """
    Applies the Dual-Signal Decision Gate:
    Alert iff: (ANOMALY_SCORE > tau) AND (PEER_RANK_PCT < rank_threshold).
    Rejects benign fleet-wide transients where rank remains high despite score spike.
    """
    res = df.copy()
    is_score_high = res[score_col] > tau
    is_rank_low = res[rank_col] < rank_threshold
    res["DUAL_GATE_ALERT"] = is_score_high & is_rank_low
    return res


def compare_thresholds(
    scores: np.ndarray,
    c_fa: float = 50.0,
    c_miss: float = 200.0,
) -> Dict[str, Any]:
    """
    Calculates and compares all mathematical thresholds alongside the arbitrary 98th percentile.
    """
    s = np.asarray(scores, dtype=float).flatten()
    s = s[~np.isnan(s)]
    n = len(s)
    
    tau_evt = fit_evt_threshold(s)
    tau_gmm = fit_gmm_threshold(s)
    tau_cost = fit_cost_optimal_threshold(s, c_false_alarm=c_fa, c_missed=c_miss)
    tau_naive = float(np.percentile(s, 98.0))
    
    methods = {
        "EVT": tau_evt,
        "GMM": tau_gmm,
        "Cost-Optimal": tau_cost,
        "Naive-98th": tau_naive,
    }
    
    summary = {}
    for name, tau in methods.items():
        flagged = int(np.sum(s > tau))
        summary[name] = {
            "threshold": float(tau),
            "flagged_count": flagged,
            "flagged_pct": float(flagged / n * 100.0) if n else 0.0,
        }
        
    return summary
