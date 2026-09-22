"""
Test endpoint detection using Pirani decline transition.

This test verifies that the find_endpoint() function correctly detects
the primary drying endpoint using the Pirani decline transition method,
which is the correct physics for this system where Pirani and CM never
fully converge.

Event: E-PHYS-TRANSITION-001
Phase: Phase 0 / Phase 1 blocked
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from core_pipeline import find_endpoint


def create_synthetic_transition_dataset(
    n_points=200,
    primary_baseline_pa=5.3,  # Pirani reads ~5.3 Pa above CM during primary drying
    post_baseline_pa=1.3,     # Pirani reads ~1.3 Pa above CM after primary drying
    transition_point=100,     # Index where transition begins
    sampling_interval_sec=60,
    noise_std=0.1,
):
    """
    Create a synthetic dataset that simulates the Pirani decline transition.
    
    The dataset has:
    - Pre-transition: Pirani reads ~5.3 Pa above CM (water vapor present)
    - Transition region: Pirani drops from 5.3 Pa to 1.3 Pa above CM
    - Post-transition: Pirani stabilizes at ~1.3 Pa above CM
    
    Parameters
    ----------
    n_points : int
        Total number of data points
    primary_baseline_pa : float
        Pirani-CM difference during primary drying (Pa)
    post_baseline_pa : float
        Pirani-CM difference after primary drying (Pa)
    transition_point : int
        Index where transition begins
    sampling_interval_sec : int
        Time between samples in seconds
    noise_std : float
        Standard deviation of noise added to readings
    
    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset with pirani_pa, capman_pa, timestamps, etc.
    expected_transition_idx : int
        Expected index of the transition point
    """
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = [
        start_time + pd.Timedelta(seconds=i * sampling_interval_sec)
        for i in range(n_points)
    ]
    
    # Capman pressure is constant
    capman_pa = np.full(n_points, 10.0)
    
    # Pirani pressure: starts high, then drops
    pirani_pa = np.zeros(n_points)
    transition_width = 20  # Number of points over which transition occurs
    
    for i in range(n_points):
        if i < transition_point:
            # Primary drying: high baseline
            pirani_pa[i] = capman_pa[i] + primary_baseline_pa + np.random.normal(0, noise_std)
        elif i < transition_point + transition_width:
            # Transition: smooth drop using sigmoid
            progress = (i - transition_point) / transition_width
            # Sigmoid transition
            factor = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
            pirani_pa[i] = capman_pa[i] + primary_baseline_pa - (primary_baseline_pa - post_baseline_pa) * factor + np.random.normal(0, noise_std)
        else:
            # Post-drying: low baseline
            pirani_pa[i] = capman_pa[i] + post_baseline_pa + np.random.normal(0, noise_std)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "Phase": [6] * n_points,
        "Cycle": [1] * n_points,
        "Step": [1] * n_points,
        "shelf_temp_k": [253.15] * n_points,
        "pressure_pa": capman_pa,
        "capman_pa": capman_pa,
        "pirani_pa": pirani_pa,
        "product_temp_k": [243.15] * n_points,
    })
    
    return df, transition_point


class TestPiraniDeclineTransition:
    """Tests for the Pirani decline transition endpoint detection."""
    
    def test_clear_transition_detection(self):
        """Test that a clear Pirani drop from 5.3 Pa to 1.3 Pa is detected."""
        df, expected_idx = create_synthetic_transition_dataset(
            n_points=200,
            primary_baseline_pa=5.3,
            post_baseline_pa=1.3,
            transition_point=100,
            sampling_interval_sec=60,
            noise_std=0.1,
        )
        
        # Use parameters appropriate for the dataset
        # With 60s sampling, 30 min window = 30 samples, 30 min sustain = 30 samples
        endpoint_ts, diagnostics = find_endpoint(
            df,
            rolling_window_minutes=15.0,  # Smaller window for this test
            sustain_minutes=15.0,
            min_baseline_fraction=0.25,
        )
        
        # Verify diagnostics are populated
        assert diagnostics['detected'] == True, "Should detect transition"
        assert diagnostics['primary_baseline'] is not None
        assert diagnostics['post_baseline'] is not None
        assert diagnostics['threshold'] is not None
        
        # Check baselines are approximately correct
        assert 4.0 < diagnostics['primary_baseline'] < 6.5, \
            f"Primary baseline {diagnostics['primary_baseline']:.2f} should be ~5.3 Pa"
        assert 0.5 < diagnostics['post_baseline'] < 2.5, \
            f"Post baseline {diagnostics['post_baseline']:.2f} should be ~1.3 Pa"
        
        # Threshold should be between the two baselines
        expected_threshold = (diagnostics['primary_baseline'] + diagnostics['post_baseline']) / 2
        assert abs(diagnostics['threshold'] - expected_threshold) < 0.01
        
        # Endpoint timestamp should be near the transition point
        expected_ts = df.iloc[expected_idx]["timestamp"]
        # Allow some tolerance for the rolling window and sustain requirements
        time_diff = abs((endpoint_ts - expected_ts).total_seconds())
        assert time_diff < 600, f"Endpoint should be within 10 min of transition (diff: {time_diff}s)"
    
    def test_no_transition_returns_last_timestamp(self):
        """Test that dataset with no drop returns last timestamp with warning."""
        n_points = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [
            start_time + pd.Timedelta(seconds=i * 60)
            for i in range(n_points)
        ]
        
        # No transition: Pirani stays at high baseline throughout
        capman_pa = np.full(n_points, 10.0)
        pirani_pa = capman_pa + 5.3 + np.random.normal(0, 0.1, n_points)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * n_points,
            "Cycle": [1] * n_points,
            "Step": [1] * n_points,
            "shelf_temp_k": [253.15] * n_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * n_points,
        })
        
        # Capture stderr to verify warning is logged
        import io
        import sys
        
        f = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = f
        
        try:
            endpoint_ts, diagnostics = find_endpoint(
                df,
                rolling_window_minutes=15.0,
                sustain_minutes=15.0,
                min_baseline_fraction=0.25,
            )
        finally:
            sys.stderr = old_stderr
        
        stderr_output = f.getvalue()
        
        # Should return last timestamp
        assert endpoint_ts == timestamps[-1], "Should return last timestamp when no transition"
        
        # Diagnostics should indicate no detection
        assert diagnostics['detected'] == False, "Should not detect transition"
        
        # Warning should be logged
        assert "warn" in stderr_output.lower() or "baseline" in stderr_output.lower(), \
            "Should log warning about no transition"
    
    def test_brief_dip_does_not_trigger_false_endpoint(self):
        """Test that a noisy dataset with brief dip does NOT trigger false endpoint."""
        n_points = 300
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [
            start_time + pd.Timedelta(seconds=i * 60)
            for i in range(n_points)
        ]
        
        capman_pa = np.full(n_points, 10.0)
        pirani_pa = np.zeros(n_points)
        
        # Create dataset with sustained high baseline but brief dip
        for i in range(n_points):
            if 100 <= i < 115:  # Brief 15-point dip (less than sustain requirement)
                pirani_pa[i] = capman_pa[i] + 1.3 + np.random.normal(0, 0.1)
            else:
                pirani_pa[i] = capman_pa[i] + 5.3 + np.random.normal(0, 0.3)  # Higher noise
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * n_points,
            "Cycle": [1] * n_points,
            "Step": [1] * n_points,
            "shelf_temp_k": [253.15] * n_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * n_points,
        })
        
        # Use sustain requirement longer than the dip
        # 15 min sustain = 15 points with 60s sampling, dip is only 15 points
        endpoint_ts, diagnostics = find_endpoint(
            df,
            rolling_window_minutes=5.0,  # Small window to catch the dip
            sustain_minutes=20.0,  # Require 20 min sustain (dip is only 15 min)
            min_baseline_fraction=0.25,
        )
        
        # Should NOT detect a sustained transition (dip is too brief)
        # Either returns last timestamp or a point after the dip
        # The key is that it shouldn't trigger at the brief dip
        
        # If detection occurred, it should be after the brief dip region
        if diagnostics['detected']:
            endpoint_idx = df[df['timestamp'] == endpoint_ts].index[0]
            # Should not be in the brief dip region (indices 100-115)
            assert endpoint_idx > 120 or endpoint_idx < 95, \
                f"Should not detect endpoint during brief dip (idx={endpoint_idx})"


class TestEdgeCases:
    """Test edge cases for endpoint detection."""
    
    def test_missing_columns(self):
        """Test fallback when pirani_pa or capman_pa columns are missing."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1min"),
            "Phase": [6] * 50,
            "Cycle": [1] * 50,
            "Step": [1] * 50,
            "shelf_temp_k": [253.15] * 50,
            "pressure_pa": [10.0] * 50,
            "product_temp_k": [243.15] * 50,
        })
        
        endpoint_ts, diagnostics = find_endpoint(df)
        
        assert endpoint_ts == df["timestamp"].iloc[-1], \
            "Should return last timestamp when pressure columns missing"
        assert diagnostics['detected'] == False
    
    def test_inverted_baselines(self):
        """Test handling when post-baseline > primary-baseline (inverted)."""
        n_points = 100
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [
            start_time + pd.Timedelta(seconds=i * 60)
            for i in range(n_points)
        ]
        
        capman_pa = np.full(n_points, 10.0)
        # Inverted: starts low, goes high (wrong physics)
        pirani_pa = np.concatenate([
            capman_pa[:50] + 1.3 + np.random.normal(0, 0.1, 50),
            capman_pa[50:] + 5.3 + np.random.normal(0, 0.1, 50),
        ])
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * n_points,
            "Cycle": [1] * n_points,
            "Step": [1] * n_points,
            "shelf_temp_k": [253.15] * n_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * n_points,
        })
        
        import io
        import sys
        
        f = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = f
        
        try:
            endpoint_ts, diagnostics = find_endpoint(
                df,
                rolling_window_minutes=10.0,
                sustain_minutes=10.0,
                min_baseline_fraction=0.25,
            )
        finally:
            sys.stderr = old_stderr
        
        # Should fall back to last timestamp due to inverted baselines
        assert endpoint_ts == timestamps[-1], "Should return last timestamp for inverted baselines"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
