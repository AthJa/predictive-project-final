import pytest
import numpy as np
import pandas as pd
from src.threshold_model import (
    fit_evt_threshold,
    fit_gmm_threshold,
    fit_cost_optimal_threshold,
    gaussian_closed_form,
    apply_dual_signal_gate,
    compare_thresholds,
)

class TestThresholdModel:
    def test_evt_threshold_in_tail(self, synthetic_anomaly_scores):
        """EVT threshold must lie within the upper tail and below the maximum score."""
        tau = fit_evt_threshold(synthetic_anomaly_scores, initial_quantile=0.90, risk_level=0.02)
        q90 = float(np.percentile(synthetic_anomaly_scores, 90))
        max_s = float(np.max(synthetic_anomaly_scores))
        
        assert isinstance(tau, float)
        assert tau >= q90, f"Expected EVT threshold >= 90th percentile ({q90:.3f}), got {tau:.3f}"
        assert tau < max_s, f"Expected EVT threshold < max score ({max_s:.3f}), got {tau:.3f}"

    def test_gmm_threshold_separates_clusters(self, synthetic_anomaly_scores):
        """GMM threshold on bimodal distribution (modes at 0.20 and 0.70) must land in the valley [0.35, 0.60]."""
        tau = fit_gmm_threshold(synthetic_anomaly_scores, n_components=2)
        assert isinstance(tau, float)
        assert 0.35 <= tau <= 0.60, f"Expected GMM threshold between modes [0.35, 0.60], got {tau:.3f}"

    def test_cost_optimal_monotonicity(self, synthetic_anomaly_scores):
        """Higher missed-anomaly cost must lower tau*; higher false-alarm cost must raise tau*."""
        tau_base = fit_cost_optimal_threshold(
            synthetic_anomaly_scores, c_false_alarm=50.0, c_missed=200.0
        )
        # Increasing cost of missing anomalies -> be more cautious (lower threshold)
        tau_high_miss = fit_cost_optimal_threshold(
            synthetic_anomaly_scores, c_false_alarm=50.0, c_missed=800.0
        )
        assert tau_high_miss <= tau_base, (
            f"Expected tau_high_miss ({tau_high_miss:.3f}) <= tau_base ({tau_base:.3f})"
        )
        
        # Increasing cost of false alarm dispatch -> be more conservative (raise threshold)
        tau_high_fa = fit_cost_optimal_threshold(
            synthetic_anomaly_scores, c_false_alarm=400.0, c_missed=200.0
        )
        assert tau_high_fa >= tau_base, (
            f"Expected tau_high_fa ({tau_high_fa:.3f}) >= tau_base ({tau_base:.3f})"
        )

    def test_gaussian_closed_form(self):
        """gaussian_closed_form must match analytical Bayesian critical fractile formula."""
        mu, sigma = 0.20, 0.04
        c_fa, c_miss = 50.0, 200.0
        # fractile = 200 / 250 = 0.80. Normal quantile for 0.80 is ~0.8416
        tau = gaussian_closed_form(mu, sigma, c_fa, c_miss)
        expected = mu + sigma * 0.84162123
        assert tau == pytest.approx(expected, rel=1e-3)

    def test_dual_signal_gate_suppression(self):
        """
        Dual gate must require BOTH:
          1. ANOMALY_SCORE > tau
          2. PEER_RANK_PCT < 0.25
        Crucially, a passing cloud event (high score + normal rank) must NOT alert.
        """
        df = pd.DataFrame([
            {"INVERTER_ID": "INV_NORM", "ANOMALY_SCORE": 0.15, "PEER_RANK_PCT": 0.50},  # Normal
            {"INVERTER_ID": "INV_CLOUD", "ANOMALY_SCORE": 0.65, "PEER_RANK_PCT": 0.50}, # Cloud event (suppressed!)
            {"INVERTER_ID": "INV_TRUE", "ANOMALY_SCORE": 0.65, "PEER_RANK_PCT": 0.10},  # True fault (alert!)
            {"INVERTER_ID": "INV_LAG", "ANOMALY_SCORE": 0.15, "PEER_RANK_PCT": 0.10},   # Minor laggard (no alert)
        ])
        
        df_gated = apply_dual_signal_gate(df, tau=0.50, rank_threshold=0.25)
        assert "DUAL_GATE_ALERT" in df_gated.columns
        alerts = dict(zip(df_gated["INVERTER_ID"], df_gated["DUAL_GATE_ALERT"]))
        
        assert alerts["INV_NORM"] is False or alerts["INV_NORM"] == 0
        assert alerts["INV_CLOUD"] is False or alerts["INV_CLOUD"] == 0, "Cloud transient must be suppressed!"
        assert bool(alerts["INV_TRUE"]) is True, "True fault must trigger dual gate alert!"
        assert alerts["INV_LAG"] is False or alerts["INV_LAG"] == 0

    def test_compare_thresholds_output(self, synthetic_anomaly_scores):
        """compare_thresholds must return comparative table with all 4 methods."""
        summary = compare_thresholds(synthetic_anomaly_scores)
        assert isinstance(summary, dict)
        for method in ["EVT", "GMM", "Cost-Optimal", "Naive-98th"]:
            assert method in summary
            assert "threshold" in summary[method]
            assert "flagged_count" in summary[method]
            assert "flagged_pct" in summary[method]
