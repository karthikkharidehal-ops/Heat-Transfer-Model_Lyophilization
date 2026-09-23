"""
Test mass-balance endpoint closure constraint in joint fit.

Event: E-PHYS-MASSBAL-001
Phase: Phase 1 identification
Gate impact: RESOLVES_BLOCKER
"""
import numpy as np
from pikal_model import (
    DryingSegment, 
    fit_parameters_joint, 
    simulate_continuous_primary_drying,
    RHO_ICE
)
from geometry import PCRTubeGeometry


def create_synthetic_segment(
    duration_s: float,
    n_points: int = 50,
    Ts_k: float = 253.15,
    Pc_pa: float = 10.0,
    Tp_measured_k: float = 243.15,
    fill_volume_m3: float = 10e-9,
    label: str = "test_segment"
) -> DryingSegment:
    """Create a synthetic drying segment for testing."""
    t_s = np.linspace(0, duration_s, n_points)
    Ts_k_arr = np.full(n_points, Ts_k)
    Pc_pa_arr = np.full(n_points, Pc_pa)
    Tp_measured_arr = np.full(n_points, Tp_measured_k)
    
    tube = PCRTubeGeometry()
    
    return DryingSegment(
        t_s=t_s,
        Ts_k=Ts_k_arr,
        Pc_pa=Pc_pa_arr,
        Tp_measured_k=Tp_measured_arr,
        tube=tube,
        fill_volume_m3=fill_volume_m3,
        label=label,
        is_hold_step=True
    )


def test_mass_balance_positive_residual():
    """
    Test case with artificially slow sublimation must yield a positive mass residual.
    
    When sublimation is too slow (high Rp), the simulated ice won't deplete by the
    detected endpoint time, resulting in simulated_endpoint > detected_endpoint,
    hence a positive mass residual.
    """
    # Create a short segment where we expect ice to deplete
    endpoint_time_s = 600.0  # 10 minutes
    
    seg = create_synthetic_segment(
        duration_s=endpoint_time_s,
        n_points=30,
        Ts_k=253.15,  # -20C
        Pc_pa=10.0,
        Tp_measured_k=243.15,  # -30C
        fill_volume_m3=10e-9,  # 10 uL
        label="slow_sublimation"
    )
    
    # Use parameters that produce very slow sublimation (high Rp)
    # High Rp0 and A1 will starve sublimation, causing ice to persist
    initial_guess = dict(Kv=5.0, Rp0=1e7, A1=1e8, A2=100.0)
    
    fitted = fit_parameters_joint(
        segments=[seg],
        initial_guess=initial_guess,
        use_transient=True,
        endpoint_time_s=endpoint_time_s,
        mass_balance_weight=1.0
    )
    
    # Check that mass_balance_residual exists
    assert "mass_balance_residual" in fitted, "mass_balance_residual not in fitted result"
    assert "simulated_endpoint_time_s" in fitted, "simulated_endpoint_time_s not in fitted result"
    
    # With artificially slow sublimation, simulated endpoint should be > detected endpoint
    # yielding a positive residual
    mass_residual = fitted["mass_balance_residual"]
    print(f"[TEST] Slow sublimation case:")
    print(f"  Detected endpoint: {endpoint_time_s:.1f} s")
    print(f"  Simulated endpoint: {fitted['simulated_endpoint_time_s']:.1f} s")
    print(f"  Mass-balance residual: {mass_residual:.6f}")
    
    # Residual should be positive (simulated > detected)
    assert mass_residual > 0, f"Expected positive mass residual for slow sublimation, got {mass_residual}"
    print("  [PASS] Positive mass residual confirmed\n")


def test_mass_balance_zero_residual():
    """
    Test case tuned to deplete exactly at the endpoint must yield |residual| < 0.05.
    
    We construct a scenario where parameters are tuned so ice depletes near the 
    detected endpoint time. The key is balancing sublimation rate with ice volume.
    Using a very high mass_balance_weight forces the optimizer to match the endpoint.
    """
    # Use a longer endpoint time and moderate conditions
    endpoint_time_s = 7200.0  # 120 minutes - long enough for realistic depletion
    
    seg = create_synthetic_segment(
        duration_s=endpoint_time_s,
        n_points=60,
        Ts_k=253.15,  # -20C (typical shelf temp)
        Pc_pa=10.0,   # Typical pressure
        Tp_measured_k=243.15,  # -30C product temp
        fill_volume_m3=5e-9,  # 5 uL (moderate volume)
        label="balanced_depletion"
    )
    
    # Parameters that produce moderate sublimation matching this scenario
    # These values were empirically determined to produce ~1:1 match
    initial_guess = dict(Kv=20.0, Rp0=5e4, A1=5e4, A2=30.0)
    
    fitted = fit_parameters_joint(
        segments=[seg],
        initial_guess=initial_guess,
        use_transient=True,
        endpoint_time_s=endpoint_time_s,
        mass_balance_weight=500.0  # Very high weight to strongly enforce constraint
    )
    
    # Check that mass_balance_residual exists
    assert "mass_balance_residual" in fitted, "mass_balance_residual not in fitted result"
    assert "simulated_endpoint_time_s" in fitted, "simulated_endpoint_time_s not in fitted result"
    
    mass_residual = fitted["mass_balance_residual"]
    print(f"[TEST] Tuned depletion case:")
    print(f"  Detected endpoint: {endpoint_time_s:.1f} s")
    print(f"  Simulated endpoint: {fitted['simulated_endpoint_time_s']:.1f} s")
    print(f"  Mass-balance residual: {mass_residual:.6f}")
    
    # Residual magnitude should be < 0.05 (within 5% of endpoint time)
    assert abs(mass_residual) < 0.05, f"Expected |mass residual| < 0.05, got {abs(mass_residual)}"
    print("  [PASS] Mass residual within tolerance\n")


def test_no_endpoint_time():
    """
    Test that when endpoint_time_s is None, no mass-balance constraint is applied.
    """
    seg = create_synthetic_segment(
        duration_s=600.0,
        n_points=30,
        label="no_endpoint"
    )
    
    fitted = fit_parameters_joint(
        segments=[seg],
        use_transient=True,
        endpoint_time_s=None  # No endpoint provided
    )
    
    # Should not have mass-balance fields when endpoint not provided
    assert "mass_balance_residual" not in fitted or fitted.get("mass_balance_residual") is None
    assert "simulated_endpoint_time_s" not in fitted or fitted.get("simulated_endpoint_time_s") is None
    print("[TEST] No endpoint time case: [PASS] No mass-balance fields added\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing mass-balance endpoint closure constraint")
    print("=" * 60 + "\n")
    
    test_mass_balance_positive_residual()
    test_mass_balance_zero_residual()
    test_no_endpoint_time()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
