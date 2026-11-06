import numpy as np
import pandas as pd
from typing import Optional


def compute_peer_ranks(
    df: pd.DataFrame,
    value_col: str = "AC_POWER",
    timestamp_col: str = "DATE_TIME",
    inverter_col: str = "INVERTER_ID",
) -> pd.DataFrame:
    """
    Computes instantaneous fleet percentile rankings for each inverter at each timestamp.
    
    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame containing timestamp, inverter identifier, and power/metric column.
    value_col : str, default 'AC_POWER'
        Column name of the variable to rank (higher = better performance).
    timestamp_col : str, default 'DATE_TIME'
        Timestamp column.
    inverter_col : str, default 'INVERTER_ID'
        Inverter ID column.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with 'PEER_RANK_PCT' in range [0.0, 1.0].
    """
    res = df.copy()
    
    def _rank_group(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n <= 1:
            return pd.Series(1.0, index=group.index)
        
        vals = group[value_col].to_numpy()
        # If all values are identical (e.g. 0.0 at night), assign neutral 0.5
        if np.all(vals == vals[0]):
            return pd.Series(0.5, index=group.index)
            
        # Rank from 1 (lowest) to N (highest), average ties
        ranks = group[value_col].rank(method="average", ascending=True)
        # Normalize to [0.0, 1.0]
        rank_pct = (ranks - 1.0) / (n - 1.0)
        return rank_pct

    res["PEER_RANK_PCT"] = (
        res.groupby(timestamp_col, group_keys=False)
        .apply(_rank_group, include_groups=False)
        .astype(float)
    )
    return res


def compute_rolling_peer_rank(
    df: pd.DataFrame,
    window_hours: int = 24,
    timestamp_col: str = "DATE_TIME",
    inverter_col: str = "INVERTER_ID",
    rank_col: str = "PEER_RANK_PCT",
) -> pd.DataFrame:
    """
    Computes the rolling time-window mean of the peer ranking per inverter.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PEER_RANK_PCT.
    window_hours : int, default 24
        Rolling window size in hours.
    timestamp_col : str, default 'DATE_TIME'
        Timestamp column.
    inverter_col : str, default 'INVERTER_ID'
        Inverter identifier column.
    rank_col : str, default 'PEER_RANK_PCT'
        Rank column to smooth.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'ROLLING_PEER_RANK'.
    """
    res = df.copy()
    res[timestamp_col] = pd.to_datetime(res[timestamp_col])
    res = res.sort_values([inverter_col, timestamp_col]).reset_index(drop=True)
    
    window_str = f"{window_hours}h"
    
    def _calc_rolling(grp: pd.DataFrame) -> pd.Series:
        # Time-indexed rolling mean with index preserved
        rolling_series = (
            grp.set_index(timestamp_col)[rank_col]
            .rolling(window=window_str, min_periods=1)
            .mean()
        )
        return pd.Series(rolling_series.to_numpy(), index=grp.index)

    res["ROLLING_PEER_RANK"] = (
        res.groupby(inverter_col, group_keys=False)
        .apply(_calc_rolling, include_groups=False)
        .astype(float)
    )
    return res


def flag_persistent_underperformers(
    df: pd.DataFrame,
    rank_threshold: float = 0.25,
    min_consecutive: int = 4,
    rank_col: str = "PEER_RANK_PCT",
    inverter_col: str = "INVERTER_ID",
    timestamp_col: str = "DATE_TIME",
    irrad_col: Optional[str] = "IRRADIATION",
) -> pd.DataFrame:
    """
    Flags inverters that stay below the rank threshold for at least `min_consecutive`
    intervals during active operating hours.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing peer rank columns.
    rank_threshold : float, default 0.25
        Percentile threshold below which an inverter is considered underperforming.
    min_consecutive : int, default 4
        Minimum consecutive intervals (e.g. 4 intervals @ 15min = 1 hour).
    irrad_col : str, optional
        If provided, only flag when irradiation > 0.05 to avoid nighttime artifacts.
        
    Returns
    -------
    pd.DataFrame
        DataFrame augmented with 'IS_UNDERPERFORMER' boolean flag.
    """
    res = df.copy()
    res[timestamp_col] = pd.to_datetime(res[timestamp_col])
    res = res.sort_values([inverter_col, timestamp_col]).reset_index(drop=True)
    
    # Check condition: low rank during active daylight (or unconditionally if irrad_col is None)
    if irrad_col and irrad_col in res.columns:
        is_low = (res[rank_col] < rank_threshold) & (res[irrad_col] > 0.05)
    else:
        is_low = res[rank_col] < rank_threshold
        
    def _flag_consecutive(group_low: pd.Series) -> pd.Series:
        # Rolling count of consecutive True values
        consec = group_low.astype(int).groupby((~group_low).cumsum()).cumsum()
        return consec >= min_consecutive

    flags = (
        is_low.groupby(res[inverter_col], group_keys=False)
        .apply(_flag_consecutive)
        .to_numpy()
    )
    
    res["IS_UNDERPERFORMER"] = flags
    return res
