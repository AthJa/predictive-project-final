import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from src.peer_ranking import compute_peer_ranks


def inject_fault(
    df: pd.DataFrame,
    scenario: str,
    target_inverter: Optional[str] = None,
    severity: float = 0.35,
    t_start_hour: int = 10,
    t_end_hour: int = 15,
) -> pd.DataFrame:
    """
    Injects realistic solar fleet faults into telemetry data for demonstration and testing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Fleet telemetry DataFrame.
    scenario : str
        One of 'soiling', 'cloud_transient', or 'inverter_trip'.
    target_inverter : str, optional
        Target inverter for localized faults ('soiling', 'inverter_trip').
    severity : float, default 0.35
        Fractional power drop for soiling or cloud dip.
    t_start_hour : int, default 10
        Starting hour of fault (0-23).
    t_end_hour : int, default 15
        Ending hour of fault (0-23).
        
    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with faulted telemetry, updated residuals, and adjusted anomaly scores.
    """
    res = df.copy()
    res["DATE_TIME"] = pd.to_datetime(res["DATE_TIME"])
    
    hours = res["DATE_TIME"].dt.hour
    in_time_window = (hours >= t_start_hour) & (hours <= t_end_hour)
    
    # Initialize ANOMALY_SCORE if missing
    if "ANOMALY_SCORE" not in res.columns:
        res["ANOMALY_SCORE"] = 0.20
        
    if scenario == "soiling":
        if target_inverter is None:
            target_inverter = res["INVERTER_ID"].iloc[0]
            
        mask = in_time_window & (res["INVERTER_ID"] == target_inverter) & (res["IRRADIATION"] > 0.05)
        # Attenuate target inverter power
        res.loc[mask, "AC_POWER"] = res.loc[mask, "AC_POWER"] * (1.0 - severity)
        # Target inverter gets a higher TadGAN anomaly score
        res.loc[mask, "ANOMALY_SCORE"] = np.clip(
            res.loc[mask, "ANOMALY_SCORE"] + 0.50 + severity * 0.3, 0.0, 0.95
        )
        
    elif scenario == "cloud_transient":
        # Fleet-wide transient: sudden cloud bank dips irradiance and generation across all inverters
        mask = in_time_window & (res["IRRADIATION"] > 0.05)
        res.loc[mask, "IRRADIATION"] = res.loc[mask, "IRRADIATION"] * (1.0 - severity)
        res.loc[mask, "AC_POWER"] = res.loc[mask, "AC_POWER"] * (1.0 - severity)
        
        # In a cloud transient, naive reconstruction models that expected clear sky will see a large dip
        # and spike the anomaly score for ALL inverters!
        res.loc[mask, "ANOMALY_SCORE"] = np.clip(
            res.loc[mask, "ANOMALY_SCORE"] + 0.45 + severity * 0.25, 0.0, 0.95
        )
        
    elif scenario == "inverter_trip":
        if target_inverter is None:
            target_inverter = res["INVERTER_ID"].iloc[0]
            
        mask = (hours >= t_start_hour) & (res["INVERTER_ID"] == target_inverter)
        res.loc[mask, "AC_POWER"] = 0.0
        res.loc[mask, "ANOMALY_SCORE"] = 0.92
        
    else:
        raise ValueError(f"Unknown fault scenario: {scenario}. Choose from 'soiling', 'cloud_transient', 'inverter_trip'.")

    # Update residual
    res["MODEL_RESIDUAL"] = res["AC_POWER"] - res["PREDICTED_AC_POWER"]
    # Recompute instantaneous peer ranks with updated power
    res = compute_peer_ranks(res, value_col="AC_POWER")
    return res


def evaluate_detection_comparison(
    sim_df: pd.DataFrame,
    tau_math: float = 0.50,
    tau_naive: float = 0.50,
    rank_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Compares Naive Threshold Detection vs. Our Dual-Signal Gate on simulated data.
    Demonstrates false alarm suppression during fleet-wide weather events.
    """
    scores = sim_df["ANOMALY_SCORE"].to_numpy()
    ranks = sim_df["PEER_RANK_PCT"].to_numpy()
    
    # Naive alarm: score alone exceeds threshold
    naive_mask = scores > tau_naive
    naive_count = int(np.sum(naive_mask))
    
    # Dual-signal gate: score > tau_math AND peer_rank < 25th percentile
    dual_mask = (scores > tau_math) & (ranks < rank_threshold)
    dual_count = int(np.sum(dual_mask))
    
    # Weather-suppressed: score > tau_math BUT peer rank remained healthy (>= 0.25)
    suppressed_mask = (scores > tau_math) & (ranks >= rank_threshold)
    suppressed_count = int(np.sum(suppressed_mask))
    
    reduction_pct = 0.0
    if naive_count > 0:
        reduction_pct = (naive_count - dual_count) / naive_count * 100.0
        
    return {
        "total_records": len(sim_df),
        "naive_alarms": naive_count,
        "dual_gate_alarms": dual_count,
        "weather_suppressed_count": suppressed_count,
        "false_alarm_reduction_pct": float(reduction_pct),
    }


def get_demo_presets() -> Dict[str, Dict[str, Any]]:
    """
    Provides 4 pre-configured demonstration presets for quick 1-click presentation walkthroughs.
    """
    return {
        "Clear Baseline": {
            "title": "Clear Day Normal Baseline",
            "scenario": None,
            "description": "Baseline clear day with full solar generation and healthy peer consensus.",
            "target_inverter": None,
            "severity": 0.0,
            "expected_verdict": "NORMAL",
        },
        "Passing Cloud": {
            "title": "Passing Cloud Transient (Weather Suppression)",
            "scenario": "cloud_transient",
            "description": "Fleet-wide cloud transient dips irradiance and generation. Naive threshold alarms falsely; Dual-Gate suppresses.",
            "target_inverter": None,
            "severity": 0.60,
            "expected_verdict": "WEATHER_SUPPRESSED",
        },
        "Local Soiling": {
            "title": "Inverter Soiling / Degradation",
            "scenario": "soiling",
            "description": "Target inverter suffers 35% localized power loss. Peer rank drops below 25th pct, triggering dual-signal alert.",
            "target_inverter": "INV_05",
            "severity": 0.35,
            "expected_verdict": "CONFIRMED_ANOMALY",
        },
        "Inverter Trip": {
            "title": "Inverter Open-Circuit Trip (Zero Power)",
            "scenario": "inverter_trip",
            "description": "Target inverter trips to 0.0 kW during solar noon peak. Triggers critical alarm immediately.",
            "target_inverter": "INV_12",
            "severity": 1.0,
            "expected_verdict": "CONFIRMED_ANOMALY",
        },
    }


class FleetSimulator:
    """
    Simulation state machine managing chronological playback and real-time state calculation.
    """
    def __init__(self, telemetry_df: pd.DataFrame):
        self.df = telemetry_df.copy()
        self.df["DATE_TIME"] = pd.to_datetime(self.df["DATE_TIME"])
        self.timestamps = sorted(self.df["DATE_TIME"].unique())
        self.total_steps = len(self.timestamps)
        self.current_idx = 0

    def step(self, step_size: int = 1) -> int:
        self.current_idx = min(self.total_steps - 1, self.current_idx + step_size)
        return self.current_idx

    def jump_to(self, idx: int) -> int:
        self.current_idx = max(0, min(self.total_steps - 1, idx))
        return self.current_idx

    def get_current_state(
        self,
        target_inverter: Optional[str] = None,
        tau_math: float = 0.50,
        tau_naive: float = 0.50,
    ) -> Dict[str, Any]:
        cur_dt = self.timestamps[self.current_idx]
        cur_slice = self.df[self.df["DATE_TIME"] == cur_dt]
        
        fleet_power = float(cur_slice["AC_POWER"].sum())
        fleet_pred = float(cur_slice["PREDICTED_AC_POWER"].sum())
        mean_irrad = float(cur_slice["IRRADIATION"].mean())
        
        if target_inverter is None or target_inverter not in cur_slice["INVERTER_ID"].values:
            target_inverter = cur_slice["INVERTER_ID"].iloc[0]
            
        target_row = cur_slice[cur_slice["INVERTER_ID"] == target_inverter].iloc[0]
        p_act = float(target_row["AC_POWER"])
        p_exp = float(target_row["PREDICTED_AC_POWER"])
        score = float(target_row.get("ANOMALY_SCORE", 0.20))
        rank = float(target_row.get("PEER_RANK_PCT", 0.50))
        
        naive_alert = score > tau_naive
        dual_alert = (score > tau_math) and (rank < 0.25)
        
        if dual_alert:
            verdict = "CONFIRMED_ANOMALY"
        elif naive_alert and (rank >= 0.25):
            verdict = "WEATHER_SUPPRESSED"
        else:
            verdict = "NORMAL"
            
        return {
            "step_index": self.current_idx,
            "total_steps": self.total_steps,
            "DATE_TIME": cur_dt,
            "fleet_power": fleet_power,
            "fleet_expected": fleet_pred,
            "mean_irradiation": mean_irrad,
            "target_inverter": target_inverter,
            "target_power": p_act,
            "target_expected": p_exp,
            "anomaly_score": score,
            "peer_rank": rank,
            "naive_alert": naive_alert,
            "dual_gate_alert": dual_alert,
            "verdict": verdict,
        }
