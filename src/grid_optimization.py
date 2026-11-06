import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def build_environmental_grid(
    df: pd.DataFrame,
    irr_bins: int = 5,
    temp_bins: int = 5,
    min_samples: int = 5,
    irr_col: str = "IRRADIATION",
    temp_col: str = "MODULE_TEMPERATURE",
    power_col: str = "AC_POWER",
    residual_col: str = "MODEL_RESIDUAL",
) -> pd.DataFrame:
    """
    Constructs a 2D empirical operating grid over (IRRADIATION x MODULE_TEMPERATURE).
    Computes fleet median power, MAD, and model residual statistics per cell.
    """
    # Filter to active generation data
    daytime = df[df[irr_col] > 0.01].copy()
    if daytime.empty:
        daytime = df.copy()

    # Binning with duplicate handling
    try:
        irr_cats, irr_edges = pd.qcut(daytime[irr_col], q=irr_bins, retbins=True, duplicates="drop")
    except ValueError:
        irr_cats, irr_edges = pd.cut(daytime[irr_col], bins=irr_bins, retbins=True)

    try:
        temp_cats, temp_edges = pd.qcut(daytime[temp_col], q=temp_bins, retbins=True, duplicates="drop")
    except ValueError:
        temp_cats, temp_edges = pd.cut(daytime[temp_col], bins=temp_bins, retbins=True)

    daytime["irr_bin"] = irr_cats
    daytime["temp_bin"] = temp_cats

    rows = []
    for (ib, tb), grp in daytime.groupby(["irr_bin", "temp_bin"], observed=False):
        n = len(grp)
        if n < min_samples:
            continue

        p_med = float(grp[power_col].median())
        p_mad = float(np.median(np.abs(grp[power_col] - p_med))) + 1e-4

        r_med = float(grp[residual_col].median())
        r_mad = float(np.median(np.abs(grp[residual_col] - r_med))) + 1e-4

        rows.append({
            "irr_bin": str(ib),
            "temp_bin": str(tb),
            "irr_left": ib.left,
            "irr_right": ib.right,
            "temp_left": tb.left,
            "temp_right": tb.right,
            "irr_mid": (ib.left + ib.right) / 2.0,
            "temp_mid": (tb.left + tb.right) / 2.0,
            "median_power": p_med,
            "mad_power": p_mad,
            "median_residual": r_med,
            "mad_residual": r_mad,
            "sample_count": n,
        })

    grid_df = pd.DataFrame(rows)
    return grid_df


def compute_grid_residuals(
    df: pd.DataFrame,
    grid_stats: pd.DataFrame,
    irr_col: str = "IRRADIATION",
    temp_col: str = "MODULE_TEMPERATURE",
    residual_col: str = "MODEL_RESIDUAL",
) -> pd.DataFrame:
    """
    Normalizes model residuals against empirical condition grid.
    Adds 'GRID_Z_SCORE' column.
    """
    res = df.copy()
    z_scores = np.zeros(len(res), dtype=float)

    # For faster row-wise mapping, extract numpy boundaries
    irr_vals = res[irr_col].to_numpy()
    temp_vals = res[temp_col].to_numpy()
    res_vals = res[residual_col].to_numpy()

    # Pre-extract grid arrays
    g_irr_l = grid_stats["irr_left"].to_numpy()
    g_irr_r = grid_stats["irr_right"].to_numpy()
    g_tmp_l = grid_stats["temp_left"].to_numpy()
    g_tmp_r = grid_stats["temp_right"].to_numpy()
    g_r_med = grid_stats["median_residual"].to_numpy()
    g_r_mad = grid_stats["mad_residual"].to_numpy()
    g_irr_mid = grid_stats["irr_mid"].to_numpy()
    g_tmp_mid = grid_stats["temp_mid"].to_numpy()

    for i in range(len(res)):
        irr = irr_vals[i]
        tmp = temp_vals[i]
        residual = res_vals[i]

        if irr <= 0.01:
            z_scores[i] = 0.0
            continue

        # Find matching cell
        matches = np.where((irr > g_irr_l) & (irr <= g_irr_r) & (tmp > g_tmp_l) & (tmp <= g_tmp_r))[0]
        if len(matches) > 0:
            c_idx = matches[0]
            z_scores[i] = (residual - g_r_med[c_idx]) / g_r_mad[c_idx]
        else:
            # Fallback to closest Euclidean cell
            dist = (irr - g_irr_mid)**2 + ((tmp - g_tmp_mid) / 30.0)**2
            c_idx = np.argmin(dist)
            z_scores[i] = (residual - g_r_med[c_idx]) / g_r_mad[c_idx]

    res["GRID_Z_SCORE"] = z_scores
    return res


def optimize_dispatch(
    df: pd.DataFrame,
    ppa_rate: float = 0.08,
    maint_cost: float = 150.0,
    max_crews: int = 3,
    inverter_col: str = "INVERTER_ID",
    z_col: str = "GRID_Z_SCORE",
    residual_col: str = "MODEL_RESIDUAL",
    interval_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Allocates maintenance crews via greedy optimization on expected revenue loss.
    """
    records = []
    for inv, grp in df.groupby(inverter_col):
        # Deficit power in kW
        deficit = np.maximum(0.0, -grp[residual_col].to_numpy())
        lost_kwh = float(np.sum(deficit) * interval_hours)
        revenue_loss = float(lost_kwh * ppa_rate)
        mean_z = float(grp[z_col].mean())
        
        records.append({
            "INVERTER_ID": inv,
            "MEAN_GRID_Z": mean_z,
            "ESTIMATED_LOST_KWH": lost_kwh,
            "ESTIMATED_REVENUE_LOSS": revenue_loss,
            "MAINTENANCE_COST": maint_cost,
            "NET_BENEFIT": revenue_loss - maint_cost,
        })

    dispatch_df = pd.DataFrame(records)
    # Sort worst performers first (highest revenue loss / most negative z)
    dispatch_df = dispatch_df.sort_values(
        ["ESTIMATED_REVENUE_LOSS", "MEAN_GRID_Z"], ascending=[False, True]
    ).reset_index(drop=True)

    return dispatch_df.head(max_crews)


def get_grid_cell_lookup(
    grid_stats: pd.DataFrame,
    irradiation: float,
    module_temp: float,
) -> Dict[str, Any]:
    """
    Returns baseline expectation and confidence bounds for any (G, T) coordinate.
    """
    g_irr_l = grid_stats["irr_left"].to_numpy()
    g_irr_r = grid_stats["irr_right"].to_numpy()
    g_tmp_l = grid_stats["temp_left"].to_numpy()
    g_tmp_r = grid_stats["temp_right"].to_numpy()

    matches = np.where((irradiation > g_irr_l) & (irradiation <= g_irr_r) &
                       (module_temp > g_tmp_l) & (module_temp <= g_tmp_r))[0]

    if len(matches) > 0:
        row = grid_stats.iloc[matches[0]]
    else:
        # Closest cell
        g_irr_mid = grid_stats["irr_mid"].to_numpy()
        g_tmp_mid = grid_stats["temp_mid"].to_numpy()
        dist = (irradiation - g_irr_mid)**2 + ((module_temp - g_tmp_mid) / 30.0)**2
        row = grid_stats.iloc[np.argmin(dist)]

    return {
        "irr_bin": row["irr_bin"],
        "temp_bin": row["temp_bin"],
        "expected_power": float(row["median_power"]),
        "mad_power": float(row["mad_power"]),
        "expected_residual": float(row["median_residual"]),
        "mad_residual": float(row["mad_residual"]),
        "lower_bound_power": max(0.0, float(row["median_power"] - 2.5 * row["mad_power"])),
        "upper_bound_power": float(row["median_power"] + 2.5 * row["mad_power"]),
        "sample_count": int(row["sample_count"]),
    }
