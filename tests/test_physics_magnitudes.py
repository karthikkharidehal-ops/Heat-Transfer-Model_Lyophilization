"""
Unit tests for physics magnitude validation.

These tests ensure that fitted parameters remain within physically realistic bounds.
If any parameter falls outside its valid range, the test fails with a physics violation error.
"""

import unittest
import numpy as np
from geometry import PCRTubeGeometry
from pikal_model import fit_parameters, simulate_cycle


# Physics-based bounds for Kv (heat transfer coefficient)
# Typical values for lyophilization: 1-500 W/m²/K
# Values outside this range indicate unphysical model behavior
KV_MIN = 1.0  # W/m²/K
KV_MAX = 500.0  # W/m²/K


def generate_synthetic_data(
    tube: PCRTubeGeometry,
    fill_volume_m3: float,
    true_Kv: float,
    true_Rp0: float,
    true_A1: float,
    true_A2: float,
    n_points: int = 100,
    noise_std_k: float = 0.5,
    use_transient: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic primary drying data for testing.
    
    Returns
    -------
    t_s, Ts_k, Pc_pa, Tp_measured_k : arrays
        Time, shelf temp, chamber pressure, and product temp data
    """
    t_s = np.linspace(0, 3600, n_points)  # 1 hour
    Ts_k = np.full(n_points, 253.15)  # -20°C shelf temp
    Pc_pa = np.full(n_points, 10.0)  # 10 Pa chamber pressure
    
    # Simulate with known parameters
    Tp_sim = simulate_cycle(
        t_s, Ts_k, Pc_pa, tube, fill_volume_m3,
        true_Kv, true_Rp0, true_A1, true_A2,
        use_transient=use_transient
    )
    
    # Add measurement noise
    noise = np.random.normal(0, noise_std_k, n_points)
    Tp_measured_k = Tp_sim + noise
    
    return t_s, Ts_k, Pc_pa, Tp_measured_k


class TestPhysicsMagnitudes(unittest.TestCase):
    """Tests to validate that fitted parameters are physically realistic."""
    
    def test_kv_within_physical_bounds_typical(self):
        """
        Test that Kv fitted from typical lyophilization data falls within [1, 500] W/m²/K.
        
        This is the primary test case using realistic parameters for PCR tube lyophilization.
        """
        tube = PCRTubeGeometry()
        fill_volume_m3 = 6e-9  # 6 µL
        
        # True parameters within physical ranges
        true_Kv = 15.0  # W/m²/K - typical value
        true_Rp0 = 2e4  # m²·s·Pa/kg
        true_A1 = 1e6  # m·s·Pa/kg
        true_A2 = 100.0  # 1/m
        
        np.random.seed(42)  # For reproducibility
        t_s, Ts_k, Pc_pa, Tp_measured_k = generate_synthetic_data(
            tube, fill_volume_m3, true_Kv, true_Rp0, true_A1, true_A2
        )
        
        # Fit parameters
        initial_guess = dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
        fitted = fit_parameters(
            t_s, Ts_k, Pc_pa, Tp_measured_k,
            tube, fill_volume_m3, initial_guess
        )
        
        fitted_Kv = fitted["Kv"]
        
        # PHYSICS VIOLATION CHECK: Kv MUST be in [1, 500] W/m²/K
        self.assertGreaterEqual(fitted_Kv, KV_MIN,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is below "
            f"the minimum physically valid value {KV_MIN} W/m²/K."
        )
        self.assertLessEqual(fitted_Kv, KV_MAX,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is above "
            f"the maximum physically valid value {KV_MAX} W/m²/K. "
            f"This indicates unphysical heat transfer modeling."
        )
    
    def test_kv_within_physical_bounds_low(self):
        """Test Kv fitting when true Kv is at the low end of physical range."""
        tube = PCRTubeGeometry()
        fill_volume_m3 = 6e-9
        
        true_Kv = 2.0  # Low but physical
        true_Rp0 = 2e4
        true_A1 = 1e6
        true_A2 = 100.0
        
        np.random.seed(42)
        t_s, Ts_k, Pc_pa, Tp_measured_k = generate_synthetic_data(
            tube, fill_volume_m3, true_Kv, true_Rp0, true_A1, true_A2
        )
        
        initial_guess = dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
        fitted = fit_parameters(
            t_s, Ts_k, Pc_pa, Tp_measured_k,
            tube, fill_volume_m3, initial_guess
        )
        
        fitted_Kv = fitted["Kv"]
        
        self.assertGreaterEqual(fitted_Kv, KV_MIN,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is below "
            f"the minimum physically valid value {KV_MIN} W/m²/K."
        )
        self.assertLessEqual(fitted_Kv, KV_MAX,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is above "
            f"the maximum physically valid value {KV_MAX} W/m²/K."
        )
    
    def test_kv_within_physical_bounds_high(self):
        """Test Kv fitting when true Kv is at the high end of physical range."""
        tube = PCRTubeGeometry()
        fill_volume_m3 = 6e-9
        
        true_Kv = 200.0  # High but physical
        true_Rp0 = 2e4
        true_A1 = 1e6
        true_A2 = 100.0
        
        np.random.seed(42)
        t_s, Ts_k, Pc_pa, Tp_measured_k = generate_synthetic_data(
            tube, fill_volume_m3, true_Kv, true_Rp0, true_A1, true_A2
        )
        
        initial_guess = dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
        fitted = fit_parameters(
            t_s, Ts_k, Pc_pa, Tp_measured_k,
            tube, fill_volume_m3, initial_guess
        )
        
        fitted_Kv = fitted["Kv"]
        
        self.assertGreaterEqual(fitted_Kv, KV_MIN,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is below "
            f"the minimum physically valid value {KV_MIN} W/m²/K."
        )
        self.assertLessEqual(fitted_Kv, KV_MAX,
            f"PHYSICS VIOLATION: Fitted Kv = {fitted_Kv:.2f} W/m²/K is above "
            f"the maximum physically valid value {KV_MAX} W/m²/K."
        )
    
    def test_contact_area_larger_than_base_area(self):
        """
        Verify that contact_area_m2 returns a larger value than base_area_m2
        for non-zero fill heights, confirming the lateral surface area is included.
        """
        tube = PCRTubeGeometry()
        fill_height = 10e-3  # 10 mm fill height
        
        base_area = tube.base_area_m2()
        contact_area = tube.contact_area_m2(fill_height)
        
        self.assertGreater(contact_area, base_area,
            f"PHYSICS VIOLATION: Contact area ({contact_area:.6e} m²) should be "
            f"larger than base area ({base_area:.6e} m²) for fill_height={fill_height:.3e} m. "
            f"The lateral surface area must be included in heat transfer calculations."
        )
        
        # The contact area should be significantly larger (order of magnitude)
        # because it includes the lateral surface area of the frustum
        ratio = contact_area / base_area
        self.assertGreater(ratio, 5.0,
            f"Contact area ratio = {ratio:.2f} is too small. "
            f"Expected ratio > 5 for typical PCR tube fill heights."
        )


if __name__ == "__main__":
    unittest.main()
