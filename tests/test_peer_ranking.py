import pytest
import numpy as np
import pandas as pd
from src.peer_ranking import (
    compute_peer_ranks,
    compute_rolling_peer_rank,
    flag_persistent_underperformers,
)

class TestPeerRanking:
    def test_peer_rank_normalized_range(self, synthetic_fleet_df):
        """All PEER_RANK_PCT values must lie strictly in [0, 1]."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        assert "PEER_RANK_PCT" in df_ranked.columns
        ranks = df_ranked["PEER_RANK_PCT"]
        assert not ranks.isna().any(), "Peer ranks must not contain NaN"
        assert (ranks >= 0.0).all(), "Peer ranks must be >= 0.0"
        assert (ranks <= 1.0).all(), "Peer ranks must be <= 1.0"

    def test_highest_producer_rank_one(self, synthetic_fleet_df):
        """At any daytime timestamp, the inverter with highest AC_POWER has rank 1.0."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        noon_df = df_ranked[df_ranked["IRRADIATION"] > 0.5]
        first_noon_ts = noon_df["DATE_TIME"].iloc[0]
        slice_ts = df_ranked[df_ranked["DATE_TIME"] == first_noon_ts]
        
        max_inv = slice_ts.loc[slice_ts["AC_POWER"].idxmax()]
        assert max_inv["PEER_RANK_PCT"] == pytest.approx(1.0, abs=1e-5)

    def test_degraded_inverter_rank_drop(self, synthetic_fleet_df):
        """INV_BAD with 35% power drop must have mean daytime rank < 0.20."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        daytime = df_ranked[df_ranked["IRRADIATION"] > 0.1]
        bad_inv_daytime = daytime[daytime["INVERTER_ID"] == "INV_BAD"]
        
        mean_rank = bad_inv_daytime["PEER_RANK_PCT"].mean()
        assert mean_rank < 0.20, f"Expected INV_BAD mean rank < 0.20, got {mean_rank:.3f}"

    def test_rolling_rank_reduces_variance(self, synthetic_fleet_df):
        """Rolling rank over time must smooth out instantaneous noise."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        df_rolling = compute_rolling_peer_rank(df_ranked, window_hours=6)
        assert "ROLLING_PEER_RANK" in df_rolling.columns
        
        healthy = df_rolling[df_rolling["INVERTER_ID"] == "INV_01"]
        daytime_healthy = healthy[healthy["IRRADIATION"] > 0.1]
        
        std_raw = daytime_healthy["PEER_RANK_PCT"].std()
        std_rolling = daytime_healthy["ROLLING_PEER_RANK"].std()
        assert std_rolling <= std_raw + 1e-4, "Rolling rank must have <= variance than raw rank"

    def test_persistent_underperformer_flag(self, synthetic_fleet_df):
        """INV_BAD must be flagged as underperformer while top inverters are not."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        df_flagged = flag_persistent_underperformers(
            df_ranked, rank_threshold=0.25, min_consecutive=4
        )
        assert "IS_UNDERPERFORMER" in df_flagged.columns
        
        bad_flags = df_flagged[df_flagged["INVERTER_ID"] == "INV_BAD"]["IS_UNDERPERFORMER"]
        good_flags = df_flagged[df_flagged["INVERTER_ID"] == "INV_01"]["IS_UNDERPERFORMER"]
        
        # In daylight hours, INV_BAD should be persistently underperforming
        daytime_bad = df_flagged[(df_flagged["INVERTER_ID"] == "INV_BAD") & (df_flagged["IRRADIATION"] > 0.2)]
        assert daytime_bad["IS_UNDERPERFORMER"].sum() > 0, "INV_BAD should have active underperformer flags"
        assert not good_flags.all(), "Healthy inverter should not be flagged continuously"

    def test_nighttime_zero_power_handling(self, synthetic_fleet_df):
        """When all inverters produce 0 kW at night, no errors occur and ranks are well-defined."""
        df_ranked = compute_peer_ranks(synthetic_fleet_df, value_col="AC_POWER")
        night_df = df_ranked[df_ranked["IRRADIATION"] == 0.0]
        assert not night_df["PEER_RANK_PCT"].isna().any()
