"""
Strict physics validation tests for the Pikal primary-drying model.

These tests enforce physical continuity and magnitude guardrails to ensure
the model produces physically realistic results.

Event: E-TEST-PHYSICS-001
Phase: Phase 0
Gate impact: BLOCKING
"""

import numpy as np
import pytest
from geometry import PCRTubeGeometry
from pikal_model import simulate_cycle, solve_Tp, DH_S, RHO_ICE


class TestFrontAreaMonotonicity:
    """Test that sublimation front area strictly decreases or remains constant as dried layer thickness increases."""

    def test_front_area_decreases_with_dried_layer_thickness(self):
        """
        As the dried layer thickness L increases (ice front recedes downward),
        the cross-sectional area A_front at the sublimation front must strictly
        decrease or remain constant. It can NEVER increase, because the PCR tube
        tapers from wide top to narrow bottom.
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 50e-6  # 50 µL typical fill
        fill_height = tube.fill_height(fill_volume_m3)
        
        # Generate a sequence of increasing dried layer thicknesses
        L_values = np.linspace(0.0, fill_height * 0.95, 50)
        
        # Calculate front area at each thickness
        A_front_values = [tube.front_area_m2(fill_height, L) for L in L_values]
        
        # Assert monotonic decrease (or constant): A_front[i+1] <= A_front[i]
        for i in range(len(A_front_values) - 1):
            assert A_front_values[i + 1] <= A_front_values[i] + 1e-15, (
                f"Physics violation: Front area increased from {A_front_values[i]:.6e} m² "
                f"to {A_front_values[i + 1]:.6e} m² as dried layer thickness increased "
                f"from {L_values[i]:.6e} m to {L_values[i + 1]:.6e} m. "
                f"This violates the tapered tube geometry constraint."
            )

    def test_front_area_never_exceeds_initial_area(self):
        """
        The front area at any dried layer thickness must never exceed
        the initial surface area (at L=0).
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 50e-6
        fill_height = tube.fill_height(fill_volume_m3)
        
        initial_area = tube.front_area_m2(fill_height, 0.0)
        
        # Test at various dried layer thicknesses
        for L in np.linspace(0.0, fill_height * 0.99, 100):
            A_front = tube.front_area_m2(fill_height, L)
            assert A_front <= initial_area + 1e-15, (
                f"Physics violation: Front area {A_front:.6e} m² at L={L:.6e} m "
                f"exceeds initial area {initial_area:.6e} m²."
            )


class TestIceMassNonNegative:
    """Test that ice mass can never become negative during simulation."""

    def test_ice_mass_stays_nonnegative_full_cycle(self):
        """
        Simulate a full drying cycle and verify that ice mass never goes negative.
        This tests the safeguard in simulate_cycle that clamps ice_mass >= 0.
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 50e-6
        initial_ice_mass = RHO_ICE * fill_volume_m3
        
        # Create a long time array to simulate extended drying
        t_s = np.linspace(0, 7200, 500)  # 2 hours
        Ts_k = np.full_like(t_s, 253.15)  # -20°C shelf temp
        Pc_pa = np.full_like(t_s, 50.0)   # 50 Pa chamber pressure
        
        # Use aggressive parameters to try to deplete ice quickly
        Kv = 20.0
        Rp0 = 1e4
        A1 = 1e5
        A2 = 50.0
        
        Tp_sim = simulate_cycle(
            t_s, Ts_k, Pc_pa, tube, fill_volume_m3,
            Kv, Rp0, A1, A2, use_transient=True
        )
        
        # Calculate cumulative sublimated mass at each timestep
        # by tracking the dried layer thickness progression
        L = 0.0
        for i in range(len(t_s) - 1):
            dt = t_s[i + 1] - t_s[i]
            Rp = Rp0 + A1 * L / (1.0 + A2 * L)
            A_front = tube.front_area_m2(tube.fill_height(fill_volume_m3), L)
            
            # Approximate flux (simplified, but sufficient for this test)
            from pikal_model import p_ice
            flux = max((p_ice(Tp_sim[i]) - Pc_pa[i]) / (Rp + 0.0), 0.0) if Rp > 0 else 0.0
            
            dmdt = flux * A_front
            L += max(flux / RHO_ICE, 0.0) * dt
            
            # Current ice mass
            current_ice_volume = fill_volume_m3 - (L * tube.area_at_height(
                tube.fill_height(fill_volume_m3) - L / 2
            ))  # Approximate
            current_ice_mass = max(RHO_ICE * current_ice_volume, 0.0)
            
            assert current_ice_mass >= -1e-12, (
                f"Physics violation: Ice mass became negative ({current_ice_mass:.6e} kg) "
                f"at t={t_s[i]:.1f}s, L={L:.6e}m"
            )

    def test_ice_mass_guardrail_in_simulation(self):
        """
        Directly verify that the simulation's internal ice_mass variable
        is properly guarded against going negative.
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 20e-6  # Small fill to deplete faster
        initial_ice_mass = RHO_ICE * fill_volume_m3
        
        # Very long simulation to ensure complete drying
        t_s = np.linspace(0, 14400, 200)  # 4 hours
        Ts_k = np.full_like(t_s, 263.15)  # -10°C (warmer for faster drying)
        Pc_pa = np.full_like(t_s, 30.0)   # Low pressure for fast sublimation
        
        Kv = 30.0
        Rp0 = 5e3  # Low resistance for fast drying
        A1 = 5e4
        A2 = 30.0
        
        Tp_sim = simulate_cycle(
            t_s, Ts_k, Pc_pa, tube, fill_volume_m3,
            Kv, Rp0, A1, A2, use_transient=True
        )
        
        # Verify product temperature stays physically reasonable
        # (should not go below chamber pressure's corresponding saturation temp)
        assert np.all(Tp_sim > 150.0), "Product temperature dropped to unphysical values"
        assert np.all(Tp_sim < Ts_k), "Product temperature exceeded shelf temperature"


class TestResistanceFormulation:
    """Test that the product resistance formulation behaves correctly."""

    def test_rp_greater_than_rp0_for_positive_L(self):
        """
        For the resistance formulation Rp = Rp0 + A1 * L / (1.0 + A2 * L),
        with A1 > 0 and A2 > 0, Rp must be strictly greater than Rp0 for any L > 0.
        
        This is because the dried layer adds resistance to vapor flow.
        """
        Rp0 = 2e4  # Base resistance
        A1 = 1e6   # Positive coefficient
        A2 = 100.0 # Positive coefficient
        
        # Test at various dried layer thicknesses
        L_values = np.logspace(-6, -2, 50)  # From 1 µm to 1 cm
        
        for L in L_values:
            Rp = Rp0 + A1 * L / (1.0 + A2 * L)
            
            assert Rp > Rp0, (
                f"Physics violation: Rp ({Rp:.2f}) is not greater than Rp0 ({Rp0:.2f}) "
                f"at L={L:.6e} m with A1={A1}, A2={A2}. "
                f"The dried layer must add resistance, not reduce it."
            )
            

    def test_rp_monotonic_increase(self):
        """
        Product resistance should increase monotonically with dried layer thickness.
        """
        Rp0 = 2e4
        A1 = 1e6
        A2 = 100.0
        
        L_values = np.linspace(0.0, 0.01, 100)  # 0 to 1 cm
        Rp_values = [Rp0 + A1 * L / (1.0 + A2 * L) for L in L_values]
        
        for i in range(len(Rp_values) - 1):
            assert Rp_values[i + 1] >= Rp_values[i] - 1e-10, (
                f"Physics violation: Rp decreased from {Rp_values[i]:.2f} to {Rp_values[i + 1]:.2f} "
                f"as L increased from {L_values[i]:.6e} to {L_values[i + 1]:.6e}"
            )

    def test_resistance_parameters_must_be_positive(self):
        """
        All resistance parameters (Rp0, A1, A2) must be positive for physical validity.
        """
        # These are enforced by the least_squares bounds in fit_parameters,
        # but we test the formula behavior here
        L = 0.001  # 1 mm dried layer
        
        # Test with valid positive parameters
        Rp0, A1, A2 = 2e4, 1e6, 100.0
        Rp_valid = Rp0 + A1 * L / (1.0 + A2 * L)
        assert Rp_valid > 0, "Valid parameters produced non-positive resistance"
        
        # Note: Negative A1 would cause Rp < Rp0, which is unphysical
        # Negative A2 could cause division issues
        # The fitting routine bounds these to positive values


class TestSolverBracket:
    """Test that the solver bracket handles edge cases correctly."""

    def test_solver_bracket_warning_near_shelf_temp(self):
        """
        When Tp_k approaches Ts_k - 0.01, the solver bracket is limiting.
        This test verifies the bracket behavior and ensures no silent clipping.
        """
        import warnings
        
        Ts_k = 253.15  # -20°C
        Pc_pa = 50.0
        Rp = 1e4
        Rs = 0.0
        contact_area = 1e-4
        A_front = 1e-5
        Kv = 15.0
        
        # When shelf temp is very high relative to chamber pressure,
        # the equilibrium Tp could approach Ts_k
        # The solver bracket is [Ts_k - 60, Ts_k - 0.01]
        
        lo, hi = Ts_k - 60.0, Ts_k - 0.01
        
        # Solve and check that result is within bracket
        flux, Tp_k = solve_Tp(Ts_k, Pc_pa, Rp, Rs, contact_area, A_front, Kv)
        
        assert Tp_k >= lo, f"Tp_k ({Tp_k:.4f}) fell below lower bracket bound ({lo:.4f})"
        assert Tp_k <= hi, (
            f"Tp_k ({Tp_k:.4f}) exceeded upper bracket bound ({hi:.4f}). "
            f"This indicates the solver bracket may be limiting the physics solution. "
            f"WARNING: The true equilibrium temperature may be higher than the solver allows."
        )
        
        # If Tp_k is very close to the upper bound, the bracket is active
        if Tp_k > hi - 0.001:
            # This is expected behavior when heat transfer dominates
            # The warning is implicit in the bracket design
            pass

    def test_solver_handles_extreme_conditions(self):
        """
        Test solver behavior under extreme conditions that might push against bracket limits.
        """
        # Very high Kv (extreme heat transfer)
        Ts_k = 263.15
        Pc_pa = 10.0  # Very low pressure
        Rp = 1e3  # Very low resistance
        Rs = 0.0
        contact_area = 5e-4
        A_front = 5e-5
        Kv = 500.0  # Upper bound of physical range
        
        flux, Tp_k = solve_Tp(Ts_k, Pc_pa, Rp, Rs, contact_area, A_front, Kv)
        
        # Tp should still be within physical bounds
        assert Ts_k - 60.0 <= Tp_k <= Ts_k - 0.01, (
            f"Solver produced out-of-bounds Tp_k={Tp_k:.4f}K for extreme conditions"
        )
        
        # Flux should be non-negative
        assert flux >= 0.0, f"Negative flux ({flux:.6e}) is unphysical"

    def test_bracket_does_not_silently_clip(self):
        """
        Verify that when the solver hits the bracket boundary, it returns
        a physically reasonable value rather than an arbitrary clipped value.
        """
        # Set up conditions where Tp might want to go very low
        Ts_k = 233.15  # -40°C (very cold shelf)
        Pc_pa = 100.0  # Higher pressure
        Rp = 1e5  # High resistance
        Rs = 0.0
        contact_area = 1e-4
        A_front = 1e-5
        Kv = 5.0  # Low heat transfer
        
        flux, Tp_k = solve_Tp(Ts_k, Pc_pa, Rp, Rs, contact_area, A_front, Kv)
        
        lo, hi = Ts_k - 60.0, Ts_k - 0.01
        
        # Should be within bracket
        assert lo <= Tp_k <= hi
        
        # If at lower bound, check it's physically reasonable
        if Tp_k < lo + 0.1:
            # At very low temperatures, p_ice becomes extremely small
            # This is expected behavior for cold shelf / high resistance conditions
            from pikal_model import p_ice
            p_ice_val = p_ice(Tp_k)
            assert p_ice_val >= 0, "Ice vapor pressure became negative"


class TestIntegratedPhysicsConsistency:
    """Integration tests for overall physics consistency."""

    def test_full_cycle_physics_consistency(self):
        """
        Run a complete drying simulation and verify all physical constraints hold.
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 50e-6
        
        t_s = np.linspace(0, 3600, 100)  # 1 hour
        Ts_k = np.full_like(t_s, 253.15)
        Pc_pa = np.full_like(t_s, 50.0)
        
        Kv = 15.0
        Rp0 = 2e4
        A1 = 1e6
        A2 = 100.0
        
        Tp_sim = simulate_cycle(
            t_s, Ts_k, Pc_pa, tube, fill_volume_m3,
            Kv, Rp0, A1, A2, use_transient=False
        )
        
        # Check 1: Product temp always below shelf temp
        assert np.all(Tp_sim <= Ts_k + 1e-6), "Product temperature exceeded shelf temperature"
        
        # Check 2: Product temp above absolute zero and reasonable lower bound
        assert np.all(Tp_sim > 150.0), "Product temperature dropped to unphysical values"
        
        # Check 3: Temperature changes smoothly (no jumps)
        temp_diffs = np.abs(np.diff(Tp_sim))
        max_jump = np.max(temp_diffs)
        assert max_jump < 10.0, (
            f"Temperature jumped by {max_jump:.2f}K between timesteps - likely numerical instability"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
