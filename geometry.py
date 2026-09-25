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

    @property
    def outer_radius(self) -> float:
        """Return the top (outer) radius of the PCR tube."""
        return self.top_radius_m

    def front_area_m2(self, fill_height_m: float, dried_thickness_m: float) -> float:
        """
        Cross-sectional area of the sublimation front.
        """
        front_height = max(fill_height_m - dried_thickness_m, 0.0)
        return self.area_at_height(front_height)


# ---------------------------------------------------------------------------
# Single measured anchor (event E-GEOM-ANCHOR-001)
# ---------------------------------------------------------------------------
# In production PCR tubes, a 16.0 uL fill sits at a 4.0 mm fill height.
# This is the ONLY calibration point currently trusted. Multi-point
# calibration is DEFERRED and tracked as an open assumption
# (A-GEOM-004 in ASSUMPTIONS_REGISTER.md); it is MANDATORY before any
# recipe optimization or advisory MPC use.
ANCHOR_FILL_VOLUME_UL = 16.0
ANCHOR_FILL_VOLUME_M3 = ANCHOR_FILL_VOLUME_UL * 1e-9   # 1.6e-8 m^3
ANCHOR_FILL_HEIGHT_MM = 4.0
ANCHOR_FILL_HEIGHT_M = ANCHOR_FILL_HEIGHT_MM * 1e-3    # 4.0e-3 m

GEOMETRY_PROFILES = ("cone_tip", "scaled_frustum")


class AnchoredTubeGeometry(PCRTubeGeometry):
    """PCR-tube geometry pinned to the single measured anchor.

    MEASURED ANCHOR (E-GEOM-ANCHOR-001): 16.0 uL corresponds to a 4.0 mm
    fill height in production PCR tubes. The area profile A(h) is an
    explicit, selectable ASSUMPTION:

    - ``cone_tip``: r(h) = k*h on [0, H] with k chosen so that
      V(H) = (pi/3)*k^2*H^3 equals the anchor volume exactly.
    - ``scaled_frustum``: A(h) = c * A_ideal_frustum(h), where the ideal
      frustum is the legacy PCRTubeGeometry taper, and c is chosen so
      that V(H) = c * V_ideal(H) equals the anchor volume exactly.

    Self-consistency contract (enforced by tests/test_anchored_geometry.py):
      V(h)  = integral of A(h') dh' from 0 to h  (closed-form, exact)
      A(h)  = dV/dh (analytic)
      r(h)  = sqrt(A(h)/pi)
      ice mass is DERIVED from L only:
          ice_mass(L) = RHO_ICE * (V(H) - V(H - L))
      — never accumulated/integrated over time elsewhere.

    Multi-point calibration is deferred; this class ignores any
    calibration points passed to it.
    """

    def __init__(self, fill_volume_m3: float, fill_height_m: float,
                 profile: str = "cone_tip", **kwargs):
        if profile not in GEOMETRY_PROFILES:
            raise ValueError(
                f"Unknown geometry profile {profile!r}; "
                f"must be one of {GEOMETRY_PROFILES}"
            )
        if fill_volume_m3 <= 0 or fill_height_m <= 0:
            raise ValueError(
                "AnchoredTubeGeometry requires positive fill volume and height"
            )
        kwargs.pop("calibration_points", None)  # deferred: never interpolated
        super().__init__(**kwargs)
        self.profile = profile
        self.anchor_volume_m3 = float(fill_volume_m3)
        self.H = float(fill_height_m)  # fixed anchor height for this batch
        if profile == "cone_tip":
            # V(H) = (pi/3) k^2 H^3  =>  k = sqrt(3 V / (pi H^3))
            self.k = float(np.sqrt(3.0 * self.anchor_volume_m3
                                   / (np.pi * self.H ** 3)))
        else:
            # Ideal-frustum volume up to the anchor height:
            # V_ideal(H) = (pi H / 3)(rb^2 + rb*rH + rH^2)
            r_ideal_H = super().radius_at_height(self.H)
            rb = self.bottom_radius_m
            v_ideal_H = (np.pi * self.H / 3.0) * (rb ** 2 + rb * r_ideal_H + r_ideal_H ** 2)
            self.c = float(self.anchor_volume_m3 / v_ideal_H)

    # ---- core self-consistent primitives -------------------------------
    def A(self, h: float) -> float:
        """Cross-sectional area (m^2) at height h above the tube tip. A(h) = dV/dh."""
        h = min(max(float(h), 0.0), self.H)
        if self.profile == "cone_tip":
            return float(np.pi * self.k ** 2 * h ** 2)
        r = super().radius_at_height(h)
        return float(self.c * np.pi * r ** 2)

    def V(self, h: float) -> float:
        """Cumulative volume (m^3) from the tip up to height h. V = integral of A."""
        h = min(max(float(h), 0.0), self.H)
        if self.profile == "cone_tip":
            return float((np.pi / 3.0) * self.k ** 2 * h ** 3)
        # scaled ideal-frustum volume from bottom (0) to h
        r_bot = self.bottom_radius_m
        r_h = super().radius_at_height(h)
        v_ideal = (np.pi * h / 3.0) * (r_bot ** 2 + r_bot * r_h + r_h ** 2)
        return float(self.c * v_ideal)

    def r(self, h: float) -> float:
        """Effective radius (m) at height h, consistent with A(h): r = sqrt(A/pi)."""
        return float(np.sqrt(self.A(h) / np.pi))

    # ---- PCRTubeGeometry interface --------------------------------------
    def fill_height(self, fill_volume_m3: float) -> float:
        """Fill height is FIXED at the measured anchor for this batch.

        No interpolation from --calib strings in anchored mode.
        """
        return self.H

    def radius_at_height(self, h_from_bottom_m: float) -> float:
        """Radius consistent with the selected area profile: r = sqrt(A/pi)."""
        return self.r(h_from_bottom_m)

    def area_at_height(self, h_from_bottom_m: float) -> float:
        return self.A(h_from_bottom_m)

    def derived_ice_mass_kg(self, L_m: float, rho_ice: float) -> float:
        """Ice mass (kg) remaining when the dried layer has depth L from the top.

        DERIVED, never integrated:  ice_mass = rho_ice * (V(H) - V(H - L)).
        """
        L = min(max(float(L_m), 0.0), self.H)
        return float(rho_ice * (self.V(self.H) - self.V(self.H - L)))

    def lateral_area_m2(self, h: float) -> float:
        """True lateral (wall) contact area (m^2) between the tube wall and the
        block from the tip up to height h, consistent with r(h) = sqrt(A/pi).

        cone_tip: analytic cone slant pi*k*h*sqrt(1+k^2).
        scaled_frustum: piecewise integration of 2*pi*r(h')*sqrt(1+(dr/dh')^2)
        using the trapezoid rule on a fine grid (r is piecewise linear).
        """
        h = min(max(float(h), 0.0), self.H)
        if h <= 0.0:
            return 0.0
        if self.profile == "cone_tip":
            return float(np.pi * self.k * h * np.hypot(1.0, self.k))
        dr_dh = (super().top_radius_m - self.bottom_radius_m) / self.tube_height_m
        n = 2000
        hs = np.linspace(0.0, h, n + 1)
        rs = np.array([self.r(x) for x in hs])
        integrand = 2.0 * np.pi * rs * np.sqrt(1.0 + dr_dh ** 2)
        return float(np.trapezoid(integrand, hs))

    def base_area_m2(self) -> float:
        """Contact area at the tube bottom under the ACTIVE profile."""
        return self.A(0.0)

    def contact_area_m2(self, fill_height_m: float) -> float:
        """Effective thermal contact area (lateral wall + base), scaled by
        contact_efficiency, computed consistently with the active r(h)."""
        h = min(max(float(fill_height_m), 0.0), self.H)
        total = self.lateral_area_m2(h) + self.base_area_m2()
        return float(total * self.contact_efficiency)

    def front_area_m2(self, fill_height_m: float, dried_thickness_m: float) -> float:
        """Sublimation-front area: A at the front height (H - L)."""
        front_height = max(min(float(fill_height_m), self.H) - max(float(dried_thickness_m), 0.0), 0.0)
        return self.A(front_height)


def make_anchored_tube(profile: str = "cone_tip",
                       fill_volume_m3: float = ANCHOR_FILL_VOLUME_M3,
                       contact_efficiency: float = 0.22) -> AnchoredTubeGeometry:
    """Build an AnchoredTubeGeometry at the fixed measured anchor:
    fill height pinned to 4.0 mm; anchor volume defaults to 16.0 uL."""
    return AnchoredTubeGeometry(
        fill_volume_m3=fill_volume_m3,
        fill_height_m=ANCHOR_FILL_HEIGHT_M,
        profile=profile,
        contact_efficiency=contact_efficiency,
    )


def build_geometry_table_rows(tube, h_step_m: float = 0.5e-3) -> list[tuple[float, float, float]]:
    """Rows (h_m, V_m3, A_m2) from 0 to H inclusive in h_step_m steps."""
    H = tube.H
    n = int(round(H / h_step_m))
    rows = []
    for i in range(n + 1):
        h = min(i * h_step_m, H)
        rows.append((h, tube.V(h), tube.A(h)))
    return rows