"""
PCR tube (frustum) geometry for the Pikal model.

0.2 mL PCR tubes taper from a wide top opening to a narrow/rounded
point at the bottom — NOT a straight-walled vial. As the sublimation
front recedes from the top surface downward, the cross-sectional area
it acts through shrinks. This module computes that geometry so
pikal_model.py can use area-vs-depth instead of a constant area.

Default dimensions below are typical for standard 0.2 mL PCR tubes —
CONFIRM against your actual tube spec / caliper measurement before
trusting fitted parameters. Fill volume varies by product, so it is
passed in per-fit, not hardcoded.
"""

from dataclasses import dataclass
import numpy as np

# Typical 0.2 mL PCR tube internal dimensions — VERIFY against your tubes.
DEFAULT_TOP_RADIUS_M = 2.85e-3
DEFAULT_BOTTOM_RADIUS_M = 0.75e-3   # rounded tip approximated as small flat
DEFAULT_TUBE_HEIGHT_M = 19.5e-3


@dataclass
class PCRTubeGeometry:
    top_radius_m: float = DEFAULT_TOP_RADIUS_M
    bottom_radius_m: float = DEFAULT_BOTTOM_RADIUS_M
    tube_height_m: float = DEFAULT_TUBE_HEIGHT_M
    contact_efficiency: float = 0.22  # Graberg thesis: ~22% effective contact area for PCR tubes in well plates

    def radius_at_height(self, h_from_bottom_m: float) -> float:
        """Linear taper: radius at height h above the tube's bottom point."""
        h = np.clip(h_from_bottom_m, 0.0, self.tube_height_m)
        frac = h / self.tube_height_m
        return self.bottom_radius_m + (self.top_radius_m - self.bottom_radius_m) * frac

    def area_at_height(self, h_from_bottom_m: float) -> float:
        r = self.radius_at_height(h_from_bottom_m)
        return float(np.pi * r ** 2)

    def fill_height(self, fill_volume_m3: float) -> float:
        """
        Invert the frustum volume formula to find how high the frozen
        product column sits, given a fill volume. Solved numerically.
        """
        from scipy.optimize import brentq

        def vol_up_to(h: float) -> float:
            # Frustum volume from tube bottom (h=0) to height h
            r_bot = self.bottom_radius_m
            r_top = self.radius_at_height(h)
            return float((np.pi * h / 3.0) * (r_bot**2 + r_bot * r_top + r_top**2))

        full_vol = vol_up_to(self.tube_height_m)
        if fill_volume_m3 >= full_vol:
            return float(self.tube_height_m)
            
        if fill_volume_m3 <= 0.0:
            return 0.0

        # SciPy's brentq type stubs are loose (can return float or tuple). 
        # We cast to float and ignore the strict Pylance arg-type warning.
        root = brentq(lambda h: vol_up_to(h) - fill_volume_m3, 1e-9, self.tube_height_m)
        return float(root)  # type: ignore[arg-type]

    @staticmethod
    def fill_height_from_calibration(
        fill_volume_m3: float, 
        calibration_points: list[tuple[float, float]]
    ) -> float:
        """
        Empirical alternative to fill_height(): interpolates from
        directly measured (volume_m3, height_m) pairs.
        """
        if not calibration_points:
            return 0.0
            
        pts = sorted(calibration_points)
        vols = np.array([p[0] for p in pts])
        heights = np.array([p[1] for p in pts])
        return float(np.interp(fill_volume_m3, vols, heights))

    def base_area_m2(self) -> float:
        """
        Contact area at the tube bottom — where heat enters from the block.
        """
        return self.area_at_height(0.0)

    def contact_area_m2(self, fill_height_m: float) -> float:
        """
        Calculate the total heat transfer contact area between the shelf/block
        and the tube up to the fill height. This includes:
        1. The lateral surface area of the frustum from h=0 to h=fill_height_m
        2. The base area at the bottom
        
        The result is scaled by self.contact_efficiency to account for non-conformal
        thermal contact between the PCR tube and aluminum block/shelf (Graberg thesis).
        
        Formula for lateral area of a frustum:
            pi * (r1 + r2) * sqrt((r1 - r2)^2 + h^2)
        
        Parameters
        ----------
        fill_height_m : float
            Height of the frozen product column from the tube bottom (m)
            
        Returns
        -------
        float
            Total contact area in m² (lateral + base), scaled by contact_efficiency
        """
        h = max(fill_height_m, 0.0)
        if h <= 0.0:
            return self.base_area_m2() * self.contact_efficiency
        
        r1 = self.bottom_radius_m  # radius at bottom (h=0)
        r2 = self.radius_at_height(h)  # radius at fill height
        
        # Lateral surface area of frustum
        slant_height = np.sqrt((r1 - r2) ** 2 + h ** 2)
        lateral_area = np.pi * (r1 + r2) * slant_height
        
        # Add base area
        base_area = self.base_area_m2()
        
        # Apply contact efficiency factor (Graberg thesis: ~22% effective contact)
        return float((lateral_area + base_area) * self.contact_efficiency)

    def front_area_m2(self, fill_height_m: float, dried_thickness_m: float) -> float:
        """
        Cross-sectional area of the sublimation front.
        """
        front_height = max(fill_height_m - dried_thickness_m, 0.0)
        return self.area_at_height(front_height)