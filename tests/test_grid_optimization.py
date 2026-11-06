import pytest
import numpy as np
import pandas as pd
from src.grid_optimization import (
    build_environmental_grid,
    compute_grid_residuals,
    optimize_dispatch,
    get_grid_cell_lookup,
)

class TestGridOptimization:
    def test_grid_building_dimensions(self, synthetic_fleet_df):
        """build_environmental_grid must create quantile bins and compute cell statistics."""
        grid_stats = build_environmental_grid(
            synthetic_fleet_df, irr_bins=5, temp_bins=5, min_samples=10
        )
        assert isinstance(grid_stats, pd.DataFrame)
        required_cols = [
            "irr_bin", "temp_bin", "median_power", "mad_power",
            "median_residual", "mad_residual", "sample_count"
        ]
        for col in required_cols:
            assert col in grid_stats.columns, f"Missing required column: {col}"
        assert (grid_stats["sample_count"] >= 10).all()

    def test_residual_normalization(self, synthetic_fleet_df):
        """Condition-normalized z-score must center around 0 for healthy fleet data."""
        grid_stats = build_environmental_grid(synthetic_fleet_df, irr_bins=5, temp_bins=5)
        df_norm = compute_grid_residuals(synthetic_fleet_df, grid_stats)
        
        assert "GRID_Z_SCORE" in df_norm.columns
        daytime = df_norm[df_norm["IRRADIATION"] > 0.1]
        
        # Robust fleet median z-score should be near 0
        median_z = daytime["GRID_Z_SCORE"].dropna().median()
        assert abs(median_z) < 0.5, f"Expected fleet median z-score near 0, got {median_z:.3f}"
        
        # Healthy inverters mean z-score should also be near 0
        healthy_daytime = daytime[daytime["INVERTER_ID"] != "INV_BAD"]
        mean_healthy_z = healthy_daytime["GRID_Z_SCORE"].dropna().mean()
        assert abs(mean_healthy_z) < 0.5, f"Expected healthy mean z-score near 0, got {mean_healthy_z:.3f}"

    def test_bad_inverter_negative_zscore(self, synthetic_fleet_df):
        """Injected underperforming inverter must exhibit strongly negative z-score (<-2.0)."""
        grid_stats = build_environmental_grid(synthetic_fleet_df, irr_bins=5, temp_bins=5)
        df_norm = compute_grid_residuals(synthetic_fleet_df, grid_stats)
        
        bad_inv = df_norm[(df_norm["INVERTER_ID"] == "INV_BAD") & (df_norm["IRRADIATION"] > 0.2)]
        mean_bad_z = bad_inv["GRID_Z_SCORE"].mean()
        assert mean_bad_z < -2.0, f"Expected INV_BAD z-score < -2.0, got {mean_bad_z:.3f}"

    def test_dispatch_crew_limit(self, synthetic_fleet_df):
        """optimize_dispatch must strictly enforce max_crews and prioritize worst inverter."""
        grid_stats = build_environmental_grid(synthetic_fleet_df, irr_bins=5, temp_bins=5)
        df_norm = compute_grid_residuals(synthetic_fleet_df, grid_stats)
        
        dispatch = optimize_dispatch(
            df_norm, ppa_rate=0.08, maint_cost=50.0, max_crews=3
        )
        assert isinstance(dispatch, pd.DataFrame)
        assert len(dispatch) <= 3, "Dispatched inverters must not exceed max_crews=3"
        assert dispatch.iloc[0]["INVERTER_ID"] == "INV_BAD", "INV_BAD must be prioritized #1"
        assert dispatch.iloc[0]["ESTIMATED_REVENUE_LOSS"] > 0

    def test_condition_lookup(self, synthetic_fleet_df):
        """get_grid_cell_lookup must retrieve cell expected power and bounds for given irradiance and temp."""
        grid_stats = build_environmental_grid(synthetic_fleet_df, irr_bins=5, temp_bins=5)
        lookup = get_grid_cell_lookup(grid_stats, irradiation=0.75, module_temp=48.0)
        
        assert isinstance(lookup, dict)
        assert "expected_power" in lookup
        assert "expected_residual" in lookup
        assert "mad_residual" in lookup
        assert lookup["expected_power"] > 200.0, "At 0.75 kW/m2 expected power should be positive"
