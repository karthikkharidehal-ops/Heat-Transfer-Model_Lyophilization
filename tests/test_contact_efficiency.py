"""
Test contact efficiency scaling in PCRTubeGeometry.

This test verifies that the contact_area_m2 method correctly scales
the calculated area by the contact_efficiency factor, as established
by Graberg's thesis (~22% effective contact area for PCR tubes in well plates).

Event: E-PHYS-CONTACT-001
Phase: Phase 0 / Phase 1 blocked
"""

import pytest
import numpy as np

from geometry import PCRTubeGeometry


class TestContactEfficiency:
    """Tests for contact_efficiency parameter and scaling."""
    
    def test_default_contact_efficiency_is_22_percent(self):
        """Verify default contact_efficiency is 0.22 (Graberg finding)."""
        tube = PCRTubeGeometry()
        assert tube.contact_efficiency == 0.22
    
    def test_contact_efficiency_can_be_customized(self):
        """Verify contact_efficiency can be set to custom values."""
        tube = PCRTubeGeometry(contact_efficiency=0.5)
        assert tube.contact_efficiency == 0.5
        
        tube = PCRTubeGeometry(contact_efficiency=1.0)
        assert tube.contact_efficiency == 1.0
    
    def test_contact_area_scales_by_efficiency_factor(self):
        """
        Verify contact_area_m2 returns area scaled by contact_efficiency.
        
        This test compares areas computed with different efficiency factors
        to ensure proper scaling.
        """
        fill_height = 10e-3  # 10 mm fill height
        
        # Create tubes with different efficiency factors
        tube_full = PCRTubeGeometry(contact_efficiency=1.0)
        tube_partial = PCRTubeGeometry(contact_efficiency=0.22)
        tube_half = PCRTubeGeometry(contact_efficiency=0.5)
        
        # Calculate contact areas
        area_full = tube_full.contact_area_m2(fill_height)
        area_partial = tube_partial.contact_area_m2(fill_height)
        area_half = tube_half.contact_area_m2(fill_height)
        
        # Verify scaling relationships
        # area_partial should be 22% of area_full
        expected_partial = area_full * 0.22
        assert np.isclose(area_partial, expected_partial, rtol=1e-10), \
            f"Expected {expected_partial}, got {area_partial}"
        
        # area_half should be 50% of area_full
        expected_half = area_full * 0.5
        assert np.isclose(area_half, expected_half, rtol=1e-10), \
            f"Expected {expected_half}, got {area_half}"
    
    def test_base_area_also_scaled_by_efficiency(self):
        """
        Verify that even at zero fill height, base area is scaled.
        
        When fill_height_m <= 0, only the base area should be returned,
        and it should still be scaled by contact_efficiency.
        """
        tube_full = PCRTubeGeometry(contact_efficiency=1.0)
        tube_partial = PCRTubeGeometry(contact_efficiency=0.22)
        
        # At zero fill height, only base area is considered
        base_area_full = tube_full.contact_area_m2(0.0)
        base_area_partial = tube_partial.contact_area_m2(0.0)
        
        # Base area should also scale by efficiency
        expected_base_partial = base_area_full * 0.22
        assert np.isclose(base_area_partial, expected_base_partial, rtol=1e-10), \
            f"Expected {expected_base_partial}, got {base_area_partial}"
    
    def test_contact_area_at_various_fill_heights(self):
        """
        Verify scaling is consistent across different fill heights.
        """
        tube_full = PCRTubeGeometry(contact_efficiency=1.0)
        tube_efficiency = PCRTubeGeometry(contact_efficiency=0.22)
        
        # Test at multiple fill heights
        fill_heights = [0.0, 5e-3, 10e-3, 15e-3, 19.5e-3]
        
        for h in fill_heights:
            area_full = tube_full.contact_area_m2(h)
            area_eff = tube_efficiency.contact_area_m2(h)
            
            expected = area_full * 0.22
            assert np.isclose(area_eff, expected, rtol=1e-10), \
                f"At height {h}: expected {expected}, got {area_eff}"
    
    def test_graberg_default_reflects_thesis_finding(self):
        """
        Document that the default 0.22 value comes from Graberg's thesis.
        
        This is a regression test to ensure the default doesn't accidentally
        change from the physics-based finding.
        """
        tube = PCRTubeGeometry()
        # Graberg thesis establishes ~22% effective contact area
        assert tube.contact_efficiency == 0.22, \
            "Default contact_efficiency must remain 0.22 per Graberg thesis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
