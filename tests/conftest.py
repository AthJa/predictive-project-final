import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def synthetic_fleet_df():
    """
    Creates a synthetic 22-inverter fleet over 96 timestamps (1 day @ 15min).
    Includes daylight curves for irradiation & module temperature, expected AC power,
    and one intentionally degraded inverter ('INV_BAD') with ~35% lower output.
    """
    np.random.seed(42)
    timestamps = pd.date_range("2020-05-15 00:00:00", periods=96, freq="15min")
    inverter_ids = [f"INV_{i:02d}" for i in range(1, 22)] + ["INV_BAD"]
    
    # 96 points daylight curve (hour 6 to 19, i.e. steps 24 to 76)
    hours = np.linspace(0, 24, 96, endpoint=False)
    # Bell curve peaking at solar noon (hour 12)
    irrad_raw = np.maximum(0.0, np.sin((hours - 6) / 12 * np.pi)) ** 1.5
    irrad_curve = irrad_raw * 0.95  # peak ~0.95 kW/m²
    
    # Module temp curve: ambient ~25°C at night, up to ~55°C at peak solar noon
    mod_temp_curve = 25.0 + irrad_curve * 32.0 + np.random.normal(0, 0.5, 96)
    
    rows = []
    for t_idx, dt in enumerate(timestamps):
        irr = float(irrad_curve[t_idx])
        m_temp = float(mod_temp_curve[t_idx])
        
        # Base ideal AC power (kW)
        if irr > 0.01:
            base_ac = 750.0 * irr * (1.0 - 0.004 * (m_temp - 25.0))
            base_ac = max(0.0, base_ac)
        else:
            base_ac = 0.0
            
        for inv in inverter_ids:
            pred_ac = base_ac
            if inv == "INV_BAD" and irr > 0.05:
                # Injected degradation: ~35% power loss on INV_BAD
                actual_ac = base_ac * 0.65 + np.random.normal(0, 3.0)
            else:
                actual_ac = base_ac + (np.random.normal(0, 4.0) if irr > 0.01 else 0.0)
                
            actual_ac = max(0.0, actual_ac)
            residual = actual_ac - pred_ac
            
            rows.append({
                "DATE_TIME": dt,
                "PLANT_ID": 4135001,
                "INVERTER_ID": inv,
                "AC_POWER": actual_ac,
                "PREDICTED_AC_POWER": pred_ac,
                "MODEL_RESIDUAL": residual,
                "IRRADIATION": irr,
                "MODULE_TEMPERATURE": m_temp,
            })
            
    df = pd.DataFrame(rows)
    return df

@pytest.fixture
def synthetic_anomaly_scores():
    """
    Returns a reproducible bimodal anomaly score array:
    95% normal scores from N(0.20, 0.04), 5% anomaly scores from N(0.70, 0.08).
    """
    np.random.seed(42)
    n_total = 2000
    n_anom = int(0.05 * n_total)
    n_norm = n_total - n_anom
    
    norm_scores = np.random.normal(0.20, 0.04, n_norm)
    anom_scores = np.random.normal(0.70, 0.08, n_anom)
    
    scores = np.concatenate([norm_scores, anom_scores])
    scores = np.clip(scores, 0.001, 0.999)
    np.random.shuffle(scores)
    return scores
