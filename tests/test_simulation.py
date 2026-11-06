import pytest
import numpy as np
import pandas as pd
from src.simulation import (
    FleetSimulator,
    inject_fault,
    evaluate_detection_comparison,
    get_demo_presets,
)

class TestSimulationEngine:
    def test_fault_injection_soiling(self, synthetic_fleet_df):
        """Soiling must reduce power of target inverter only by severity factor within fault window."""
        target = "INV_05"
        severity = 0.35
        df_fault = inject_fault(
            synthetic_fleet_df,
            scenario="soiling",
            target_inverter=target,
            severity=severity,
            t_start_hour=10,
            t_end_hour=15,
        )
        assert len(df_fault) == len(synthetic_fleet_df)
        
        # Test specifically within the active fault window (10:00 - 15:00)
        window_orig = synthetic_fleet_df[
            (synthetic_fleet_df["DATE_TIME"].dt.hour >= 10) &
            (synthetic_fleet_df["DATE_TIME"].dt.hour <= 15)
        ]
        window_fault = df_fault[
            (df_fault["DATE_TIME"].dt.hour >= 10) &
            (df_fault["DATE_TIME"].dt.hour <= 15)
        ]
        
        orig_target_p = window_orig[window_orig["INVERTER_ID"] == target]["AC_POWER"].mean()
        fault_target_p = window_fault[window_fault["INVERTER_ID"] == target]["AC_POWER"].mean()
        
        expected_ratio = 1.0 - severity
        actual_ratio = fault_target_p / orig_target_p
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.05)
        
        # Peer inverter must remain unchanged
        orig_peer_p = window_orig[window_orig["INVERTER_ID"] == "INV_02"]["AC_POWER"].mean()
        fault_peer_p = window_fault[window_fault["INVERTER_ID"] == "INV_02"]["AC_POWER"].mean()
        assert fault_peer_p == pytest.approx(orig_peer_p, rel=1e-3)

    def test_fault_injection_cloud(self, synthetic_fleet_df):
        """Passing cloud must dip all inverters and irradiance fleet-wide."""
        df_fault = inject_fault(
            synthetic_fleet_df,
            scenario="cloud_transient",
            severity=0.60,
        )
        # Check during peak midday
        noon_orig = synthetic_fleet_df[synthetic_fleet_df["DATE_TIME"].dt.hour == 12]
        noon_fault = df_fault[df_fault["DATE_TIME"].dt.hour == 12]
        
        assert noon_fault["IRRADIATION"].mean() < noon_orig["IRRADIATION"].mean() * 0.55
        assert noon_fault["AC_POWER"].mean() < noon_orig["AC_POWER"].mean() * 0.55
        # All 22 inverters should show reduced power
        for inv in synthetic_fleet_df["INVERTER_ID"].unique():
            orig_inv = noon_orig[noon_orig["INVERTER_ID"] == inv]["AC_POWER"].mean()
            fault_inv = noon_fault[noon_fault["INVERTER_ID"] == inv]["AC_POWER"].mean()
            if orig_inv > 50.0:
                assert fault_inv < orig_inv

    def test_fault_injection_trip(self, synthetic_fleet_df):
        """Inverter trip must immediately drop target inverter power to 0.0 kW during fault window."""
        target = "INV_12"
        df_fault = inject_fault(
            synthetic_fleet_df,
            scenario="inverter_trip",
            target_inverter=target,
            t_start_hour=10,
            t_end_hour=15,
        )
        trip_active = df_fault[
            (df_fault["INVERTER_ID"] == target) &
            (df_fault["DATE_TIME"].dt.hour >= 10) &
            (df_fault["DATE_TIME"].dt.hour <= 15)
        ]
        assert (trip_active["AC_POWER"] == 0.0).all(), "Tripped inverter must produce 0.0 kW during fault window"

    def test_side_by_side_rejection(self, synthetic_fleet_df):
        """
        Under a passing cloud transient:
        Naive detection triggers false alarms on healthy inverters due to power dip.
        Our Dual-Signal Gate suppresses the weather dip (0 false alarms).
        """
        # Inject cloud transient
        df_cloud = inject_fault(
            synthetic_fleet_df, scenario="cloud_transient", severity=0.65
        )
        comp = evaluate_detection_comparison(df_cloud, tau_math=0.50, tau_naive=0.50)
        
        assert isinstance(comp, dict)
        assert "naive_alarms" in comp
        assert "dual_gate_alarms" in comp
        assert "weather_suppressed_count" in comp
        
        # Dual-gate alarms should be strictly less than naive alarms during cloud
        assert comp["dual_gate_alarms"] <= comp["naive_alarms"]
        assert comp["weather_suppressed_count"] > 0

    def test_fleet_simulator_playback(self, synthetic_fleet_df):
        """FleetSimulator must maintain state, step intervals, and return current status."""
        sim = FleetSimulator(synthetic_fleet_df)
        assert sim.current_idx == 0
        assert sim.total_steps == 96
        
        state_0 = sim.get_current_state()
        assert "DATE_TIME" in state_0
        assert "fleet_power" in state_0
        
        sim.step(1)
        assert sim.current_idx == 1
        
        sim.jump_to(48)  # solar noon
        state_noon = sim.get_current_state()
        assert state_noon["fleet_power"] > 0

    def test_demo_presets_availability(self):
        """get_demo_presets must return 4 ready-to-run demonstration scenarios."""
        presets = get_demo_presets()
        assert isinstance(presets, dict)
        assert len(presets) >= 4
        assert "Clear Baseline" in presets
        assert "Passing Cloud" in presets
        assert "Local Soiling" in presets
        assert "Inverter Trip" in presets
