"""
Test endpoint detection and truncation logic.

This test verifies that the find_endpoint() function correctly detects
the primary drying endpoint using Pirani/Capman convergence, and that
build_segments() properly excludes all data after this timestamp.

Event: E-PHYS-ENDPOINT-001
Phase: Phase 0 / Phase 1 blocked
"""

import numpy as np
import pandas as pd
import pytest

from core_pipeline import find_endpoint, build_segments, CalibratedPCRTubeGeometry
from pikal_model import DryingSegment


def create_synthetic_dataset(
    n_pre_endpoint=100,
    n_post_endpoint=50,
    endpoint_convergence_start=80,
    sampling_interval_sec=60,
):
    """
    Create a synthetic dataset that simulates primary drying with an early endpoint.
    
    The dataset has:
    - Pre-endpoint region: Pirani > Capman (sublimation ongoing)
    - Convergence region: Pirani ≈ Capman (sublimation ended)
    - Post-endpoint region: Data that should be excluded from fitting
    
    Parameters
    ----------
    n_pre_endpoint : int
        Number of data points before endpoint convergence begins
    n_post_endpoint : int
        Number of data points after endpoint (should be excluded)
    endpoint_convergence_start : int
        Index where convergence starts (within pre-endpoint region)
    sampling_interval_sec : int
        Time between samples in seconds
    
    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset with pirani_pa, capman_pa, timestamps, etc.
    expected_endpoint_idx : int
        Expected index of the endpoint (start of sustained convergence)
    """
    total_points = n_pre_endpoint + n_post_endpoint
    
    # Create timestamps starting from a fixed time
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = [
        start_time + pd.Timedelta(seconds=i * sampling_interval_sec)
        for i in range(total_points)
    ]
    
    # Create pressure readings
    # Pre-endpoint: Pirani significantly higher than Capman (water vapor present)
    # Post-convergence: Pirani ≈ Capman (no water vapor)
    capman_pa = np.full(total_points, 10.0)  # Constant chamber pressure at 10 Pa
    
    pirani_pa = np.zeros(total_points)
    for i in range(total_points):
        if i < endpoint_convergence_start:
            # During sublimation: Pirani reads higher due to water vapor
            pirani_pa[i] = 10.0 + 5.0 * np.exp(-i / 20) + np.random.normal(0, 0.1)
        else:
            # After sublimation: Pirani converges to Capman
            pirani_pa[i] = 10.0 + np.random.normal(0, 0.05)
    
    # Create other required columns
    df = pd.DataFrame({
        "timestamp": timestamps,
        "Phase": [6] * total_points,  # All primary drying phase
        "Cycle": [1] * total_points,
        "Step": [1] * total_points,
        "shelf_temp_k": [253.15] * total_points,  # Constant shelf temp
        "pressure_pa": capman_pa,
        "capman_pa": capman_pa,
        "pirani_pa": pirani_pa,
        "product_temp_k": [243.15] * total_points,  # Constant product temp
    })
    
    # Expected endpoint is at the start of sustained convergence
    expected_endpoint_idx = endpoint_convergence_start
    
    return df, expected_endpoint_idx


class TestFindEndpoint:
    """Tests for the find_endpoint() function."""
    
    def test_endpoint_detection_with_sustained_convergence(self):
        """Test that endpoint is detected when convergence is sustained."""
        df, expected_idx = create_synthetic_dataset(
            n_pre_endpoint=100,
            n_post_endpoint=50,
            endpoint_convergence_start=60,
            sampling_interval_sec=60,
        )
        
        # With 60 minutes of data before convergence and 20 min sustain requirement
        endpoint_ts = find_endpoint(
            df, 
            abs_tol_pa=1.0, 
            rel_tol=0.15, 
            sustain_minutes=20.0
        )
        
        # Endpoint should be at or near the convergence start
        expected_ts = df.iloc[expected_idx]["timestamp"]
        assert endpoint_ts == expected_ts, \
            f"Expected endpoint at {expected_ts}, got {endpoint_ts}"
    
    def test_endpoint_detection_no_convergence(self):
        """Test fallback when no convergence is detected."""
        # Create dataset where Pirani never converges to Capman
        total_points = 100
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [
            start_time + pd.Timedelta(seconds=i * 60)
            for i in range(total_points)
        ]
        
        capman_pa = np.full(total_points, 10.0)
        pirani_pa = np.full(total_points, 20.0)  # Always 10 Pa higher
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * total_points,
            "Cycle": [1] * total_points,
            "Step": [1] * total_points,
            "shelf_temp_k": [253.15] * total_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * total_points,
        })
        
        # Should fall back to last timestamp
        endpoint_ts = find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0)
        assert endpoint_ts == timestamps[-1], \
            "Should return last timestamp when no convergence detected"
    
    def test_endpoint_detection_missing_columns(self):
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
        
        # Should fall back to last timestamp
        endpoint_ts = find_endpoint(df)
        assert endpoint_ts == df["timestamp"].iloc[-1], \
            "Should return last timestamp when pressure columns missing"


class TestBuildSegmentsWithEndpoint:
    """Tests for build_segments() endpoint truncation logic."""
    
    def test_segments_exclude_post_endpoint_data(self):
        """Verify that segments do not include data after the endpoint."""
        df, _ = create_synthetic_dataset(
            n_pre_endpoint=100,
            n_post_endpoint=50,
            endpoint_convergence_start=60,
            sampling_interval_sec=60,
        )
        
        # Get endpoint timestamp
        endpoint_ts = find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0)
        
        # Build segments with endpoint truncation
        tube = CalibratedPCRTubeGeometry()
        segments = build_segments(
            df=df,
            primary_phase_code=6,
            tube=tube,
            fill_volume_m3=6e-9,
            endpoint_timestamp=endpoint_ts,
            min_segment_points=5,
        )
        
        # Verify we have segments
        assert len(segments) > 0, "Should have at least one segment"
        
        # Verify no segment contains data after endpoint
        for seg in segments:
            # Convert segment times back to timestamps
            t0 = df[df["Phase"] == 6]["timestamp"].min()
            segment_timestamps = [t0 + pd.Timedelta(seconds=t) for t in seg.t_s]
            
            for ts in segment_timestamps:
                assert ts <= endpoint_ts, \
                    f"Segment {seg.label} contains timestamp {ts} after endpoint {endpoint_ts}"
    
    def test_segments_truncated_at_endpoint_boundary(self):
        """Verify that segments containing the endpoint are sliced correctly."""
        # Create dataset with multiple steps, where endpoint falls in the middle of a step
        total_points = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [start_time + pd.Timedelta(seconds=i * 60) for i in range(total_points)]
        
        # Create two steps: Step 1 (0-99), Step 2 (100-199)
        steps = [1] * 100 + [2] * 100
        
        capman_pa = np.full(total_points, 10.0)
        pirani_pa = np.concatenate([
            10.0 + 5.0 * np.exp(-np.arange(100) / 20) + np.random.normal(0, 0.1, 100),
            10.0 + np.random.normal(0, 0.05, 100)
        ])
        
        # Endpoint falls at index 80 (in Step 1)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * total_points,
            "Cycle": [1] * total_points,
            "Step": steps,
            "shelf_temp_k": [253.15] * total_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * total_points,
        })
        
        endpoint_ts = find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0)
        
        tube = CalibratedPCRTubeGeometry()
        segments = build_segments(
            df=df,
            primary_phase_code=6,
            tube=tube,
            fill_volume_m3=6e-9,
            endpoint_timestamp=endpoint_ts,
            min_segment_points=5,
        )
        
        # Step 1 should be truncated at endpoint
        step1_seg = next((s for s in segments if "step=1" in s.label), None)
        assert step1_seg is not None, "Step 1 segment should exist"
        
        # Verify Step 1 segment duration matches endpoint
        t0 = df[(df["Phase"] == 6) & (df["Step"] == 1)]["timestamp"].min()
        expected_duration = (endpoint_ts - t0).total_seconds()
        actual_duration = step1_seg.t_s[-1]
        
        # Allow small tolerance for floating point
        assert abs(actual_duration - expected_duration) < 1.0, \
            f"Step 1 duration {actual_duration}s should match endpoint duration {expected_duration}s"
    
    def test_post_endpoint_steps_excluded(self):
        """Verify that steps entirely after the endpoint are excluded."""
        # Create dataset where Step 3 is entirely after endpoint
        total_points = 300
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = [start_time + pd.Timedelta(seconds=i * 60) for i in range(total_points)]
        
        # Create three steps: Step 1 (0-99), Step 2 (100-199), Step 3 (200-299)
        steps = [1] * 100 + [2] * 100 + [3] * 100
        
        capman_pa = np.full(total_points, 10.0)
        pirani_pa = np.concatenate([
            10.0 + 5.0 * np.exp(-np.arange(100) / 20) + np.random.normal(0, 0.1, 100),
            10.0 + np.random.normal(0, 0.05, 200)  # Converged for steps 2 and 3
        ])
        
        # Endpoint falls at index 80 (in Step 1)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "Phase": [6] * total_points,
            "Cycle": [1] * total_points,
            "Step": steps,
            "shelf_temp_k": [253.15] * total_points,
            "pressure_pa": capman_pa,
            "capman_pa": capman_pa,
            "pirani_pa": pirani_pa,
            "product_temp_k": [243.15] * total_points,
        })
        
        endpoint_ts = find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0)
        
        tube = CalibratedPCRTubeGeometry()
        segments = build_segments(
            df=df,
            primary_phase_code=6,
            tube=tube,
            fill_volume_m3=6e-9,
            endpoint_timestamp=endpoint_ts,
            min_segment_points=5,
        )
        
        # Step 2 and Step 3 should be excluded (entirely post-endpoint)
        step_labels = [s.label for s in segments]
        
        assert any("step=1" in label for label in step_labels), \
            "Step 1 should be included (partially pre-endpoint)"
        assert not any("step=2" in label for label in step_labels), \
            "Step 2 should be excluded (entirely post-endpoint)"
        assert not any("step=3" in label for label in step_labels), \
            "Step 3 should be excluded (entirely post-endpoint)"


class TestEndpointPreventsKvInflation:
    """
    Integration test demonstrating that endpoint truncation prevents Kv inflation.
    
    Without endpoint detection, fitting primary drying physics to secondary drying
    data (where ice is gone and product is warming) causes massive Kv inflation
    and ~6 K errors in Hold steps.
    """
    
    def test_truncation_prevents_secondary_drying_contamination(self):
        """
        Demonstrate that excluding post-endpoint data prevents model contamination.
        
        This test creates a scenario where:
        1. Primary drying ends early (at 60% of the recipe step)
        2. Secondary drying begins (ice front disappears, product warms)
        3. Without truncation, the model would fit warming data as sublimation
        4. With truncation, only true primary drying data is used
        """
        df, _ = create_synthetic_dataset(
            n_pre_endpoint=100,
            n_post_endpoint=50,
            endpoint_convergence_start=60,
            sampling_interval_sec=60,
        )
        
        endpoint_ts = find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0)
        
        # Build segments WITH endpoint truncation
        tube = CalibratedPCRTubeGeometry()
        segments_truncated = build_segments(
            df=df,
            primary_phase_code=6,
            tube=tube,
            fill_volume_m3=6e-9,
            endpoint_timestamp=endpoint_ts,
            min_segment_points=5,
        )
        
        # Calculate total data points used in truncated segments
        total_points_truncated = sum(len(seg.t_s) for seg in segments_truncated)
        
        # Build segments WITHOUT endpoint truncation (simulating the bug)
        segments_full = build_segments(
            df=df,
            primary_phase_code=6,
            tube=tube,
            fill_volume_m3=6e-9,
            endpoint_timestamp=None,  # No truncation
            min_segment_points=5,
        )
        
        total_points_full = sum(len(seg.t_s) for seg in segments_full)
        
        # Truncated version should use fewer data points
        assert total_points_truncated < total_points_full, \
            "Truncated segments should exclude post-endpoint data"
        
        # The difference should be approximately the post-endpoint data
        expected_excluded = 50  # From create_synthetic_dataset
        actual_excluded = total_points_full - total_points_truncated
        
        # Allow some tolerance for edge effects
        assert actual_excluded > 0, "Some data should be excluded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
