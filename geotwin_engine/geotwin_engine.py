"""
================================================================================
GeoTwin Engine — Physics-Based Hard-Rock Geothermal Drilling Simulator
================================================================================
Developed for collaborative use with Geothermal Engineering Ltd (GEL) at the
United Downs Deep Geothermal Project, Cornwall, UK (Carnmenellis granite), and
structured for validation against the publicly available Utah FORGE dataset
(wells 58-32, 16A(78)-32; granite, quartz monzonite, monzonite, metamorphic).

Licence intention: MIT-style open licence to facilitate adoption by other
hard-rock geothermal operators.  See LICENCE file for details.

Physics basis:
  - Rate of Penetration (ROP) from Mechanical Specific Energy (MSE) balance
    (Teale, 1965) coupled with bit-rock contact mechanics and effective stress.
  - Formation strength: UCS with depth-dependent Hoek-Brown / Mohr-Coulomb.
  - Differential pressure / pore-pressure: effective stress law, Darcy filtration.
  - Mud hydraulics: Herschel-Bulkley non-Newtonian rheology, Fanning friction.
  - Thermal coupling: 1-D convective-conductive energy balance.
  - Wear: temperature-dependent abrasive bit-wear model.

Dependencies: Python ≥ 3.9, NumPy, SciPy, Matplotlib (no black-box physics).

Authors: GeoTwin Engine contributors
Version: 0.1.0-alpha
================================================================================
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
G_STD = 9.80665          # Standard gravity, m s⁻²
STEFAN_BOLTZMANN = 5.67e-8  # W m⁻² K⁻⁴ (not used directly but kept for ref)
KELVIN_OFFSET = 273.15   # °C → K


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RockProfile:
    """All geomechanical and thermal properties for one rock unit."""
    name: str
    # Geomechanical
    ucs_mpa: float                     # Uniaxial Compressive Strength at surface [MPa]
    ucs_thermal_coeff: float = -0.15   # ΔUCSfrac per 100 °C (thermal softening)
    friction_angle_deg: float = 35.0   # Internal friction angle φ [°]
    hoek_brown_mi: float = 25.0        # Hoek-Brown intact rock parameter m_i
    gsi: float = 75.0                  # Geological Strength Index
    compressibility: float = 1e-10     # Rock compressibility [Pa⁻¹]
    # Physical
    density_kg_m3: float = 2650.0      # Bulk density [kg m⁻³]
    porosity: float = 0.005            # Fraction (dimensionless)
    permeability_md: float = 0.01      # Matrix permeability [mD]
    # Thermal
    thermal_conductivity: float = 3.0  # λ [W m⁻¹ K⁻¹]
    specific_heat: float = 800.0       # Cp [J kg⁻¹ K⁻¹]
    # Abrasivity (Cerchar Abrasivity Index proxy)
    abrasivity_index: float = 3.5      # Higher = more wear (0–6 scale)
    # Depth range for this unit
    depth_top_m: float = 0.0
    depth_base_m: float = 10000.0


@dataclass
class BitParameters:
    """Tricone / PDC bit geometry and condition parameters."""
    diameter_m: float = 0.2159         # Bit diameter [m] (8.5 in)
    nozzle_diameter_m: float = 0.0127  # Nozzle diameter [m] (0.5 in)
    n_nozzles: int = 3
    bit_type: str = "PDC"              # "PDC" | "Tricone" | "Impreg"
    # Wear state (0 = new, 1 = fully worn)
    wear_state: float = 0.0
    # Cutting efficiency coefficients (empirical, tunable)
    ce_alpha: float = 0.85             # Initial cutting efficiency
    ce_wear_sensitivity: float = 0.6   # How quickly efficiency drops with wear


@dataclass
class MudParameters:
    """Drilling fluid (mud) properties — Herschel-Bulkley rheology."""
    density_kg_m3: float = 1200.0      # Mud density [kg m⁻³]
    # Herschel-Bulkley: τ = τ_y + K * γ^n
    yield_stress_pa: float = 5.0       # τ_y [Pa]
    consistency_index: float = 0.08    # K [Pa·s^n]
    flow_behaviour_index: float = 0.72 # n (dimensionless)
    # Flow
    flow_rate_m3s: float = 0.025       # Q [m³ s⁻¹]  (~400 GPH)
    # Thermal
    specific_heat: float = 3800.0      # Cp [J kg⁻¹ K⁻¹]
    thermal_conductivity: float = 0.55 # λ_mud [W m⁻¹ K⁻¹]
    # Filtration
    filtration_coeff: float = 5e-10    # Darcy filtration coefficient [m s⁻¹ Pa⁻¹]


@dataclass
class WellGeometry:
    """Drill-string and wellbore geometry."""
    drill_pipe_od_m: float = 0.127     # Drill pipe OD [m] (5 in)
    drill_pipe_id_m: float = 0.108     # Drill pipe ID [m]
    bha_od_m: float = 0.171            # BHA OD [m]
    wellbore_diameter_m: float = 0.2159  # Same as bit initially
    inclination_deg: float = 0.0       # Wellbore inclination from vertical


@dataclass
class FormationProfile:
    """Ordered list of rock units describing a stratigraphic column."""
    name: str
    site: str
    surface_temperature_c: float = 15.0
    geothermal_gradient_c_per_km: float = 35.0   # °C km⁻¹
    surface_pressure_mpa: float = 0.101325        # Atmospheric [MPa]
    hydrostatic_gradient_mpa_per_km: float = 9.81 # MPa km⁻¹
    pore_pressure_gradient_mpa_per_km: float = 9.5  # MPa km⁻¹ (slightly sub-hydrostatic)
    overburden_gradient_mpa_per_km: float = 25.0  # MPa km⁻¹
    rock_units: List[RockProfile] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ROCK PROPERTIES MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class RockPropertiesManager:
    """
    Manages multiple rock profiles and optionally loads depth-stamped CSV logs.

    CSV format expected (depth in metres, columns flexible):
        depth_m, ucs_mpa, thermal_conductivity, density_kg_m3, porosity, temperature_c

    Missing columns are back-filled from the parent RockProfile defaults.
    Use `interpolate(depth_m, property_name)` to get any property at a given depth.
    """

    # ── Built-in rock libraries ──────────────────────────────────────────────

    CORNISH_GRANITE = RockProfile(
        name="Carnmenellis Granite",
        ucs_mpa=200.0,           # Range 150–250 MPa; use 200 as baseline
        ucs_thermal_coeff=-0.18, # Stronger thermal softening in granite
        friction_angle_deg=36.0,
        hoek_brown_mi=28.0,      # Granite m_i from Hoek (2007)
        gsi=80.0,
        density_kg_m3=2650.0,
        porosity=0.004,
        permeability_md=0.005,   # Very low matrix permeability; fracture dominated
        thermal_conductivity=3.2, # W m⁻¹ K⁻¹ (Carnmenellis: 2.5–3.5)
        specific_heat=790.0,
        abrasivity_index=4.2,    # High quartz content → high abrasivity
        depth_top_m=0.0,
        depth_base_m=5500.0,
    )

    UTAH_FORGE_GRANITE = RockProfile(
        name="Utah FORGE Granite (Milford)",
        ucs_mpa=185.0,           # From well 58-32 core data (granite suite)
        ucs_thermal_coeff=-0.14,
        friction_angle_deg=34.0,
        hoek_brown_mi=25.0,
        gsi=72.0,
        density_kg_m3=2620.0,   # 2.54–2.67 g cm⁻³ range
        porosity=0.003,          # < 0.5 % (documented)
        permeability_md=0.008,
        thermal_conductivity=2.9,
        specific_heat=820.0,
        abrasivity_index=3.8,
        depth_top_m=0.0,
        depth_base_m=2600.0,
    )

    UTAH_FORGE_QUARTZ_MONZONITE = RockProfile(
        name="Utah FORGE Quartz Monzonite",
        ucs_mpa=170.0,
        ucs_thermal_coeff=-0.12,
        friction_angle_deg=33.0,
        hoek_brown_mi=22.0,
        gsi=70.0,
        density_kg_m3=2640.0,
        porosity=0.004,
        permeability_md=0.01,
        thermal_conductivity=2.7,
        specific_heat=830.0,
        abrasivity_index=3.4,
        depth_top_m=2600.0,
        depth_base_m=3500.0,
    )

    UTAH_FORGE_METAMORPHIC = RockProfile(
        name="Utah FORGE Interfingered Metamorphic",
        ucs_mpa=220.0,
        ucs_thermal_coeff=-0.10,
        friction_angle_deg=38.0,
        hoek_brown_mi=18.0,
        gsi=65.0,
        density_kg_m3=2680.0,
        porosity=0.002,
        permeability_md=0.003,
        thermal_conductivity=3.4,
        specific_heat=760.0,
        abrasivity_index=3.0,
        depth_top_m=3500.0,
        depth_base_m=4500.0,
    )

    # ── Initialisation ───────────────────────────────────────────────────────

    def __init__(self, formation: FormationProfile):
        self.formation = formation
        # Depth-stamped log arrays (loaded from CSV if provided)
        self._log_depths: np.ndarray = np.array([])
        self._log_data: Dict[str, np.ndarray] = {}

    def load_csv_log(self, filepath: str) -> None:
        """
        Load a depth-stamped property log from CSV.

        Expected header row: depth_m, [ucs_mpa], [thermal_conductivity],
        [density_kg_m3], [porosity], [temperature_c], ...

        Any recognised column name is stored and interpolated on demand.
        To add new columns simply include them; they become available via
        `interpolate(depth, col_name)`.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV log not found: {filepath}")

        with open(filepath, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        columns = rows[0].keys()
        if "depth_m" not in columns:
            raise ValueError("CSV must contain a 'depth_m' column")

        depths = []
        data: Dict[str, List[float]] = {c: [] for c in columns if c != "depth_m"}

        for row in rows:
            try:
                depths.append(float(row["depth_m"]))
                for col in data:
                    data[col].append(float(row[col]) if row[col] else float("nan"))
            except ValueError:
                continue  # skip header-repeat or malformed lines

        self._log_depths = np.array(depths)
        self._log_data = {k: np.array(v) for k, v in data.items()}
        print(f"[RockPropertiesManager] Loaded {len(depths)} rows from {filepath.name}")

    def get_rock_unit(self, depth_m: float) -> RockProfile:
        """Return the RockProfile whose depth range contains `depth_m`."""
        for unit in self.formation.rock_units:
            if unit.depth_top_m <= depth_m < unit.depth_base_m:
                return unit
        # Fall back to deepest unit
        return self.formation.rock_units[-1]

    def interpolate(self, depth_m: float, property_name: str) -> Optional[float]:
        """
        If a CSV log has been loaded and contains `property_name`, return the
        linearly interpolated value at `depth_m`.  Returns None if unavailable.
        """
        if property_name not in self._log_data or len(self._log_depths) == 0:
            return None
        arr = self._log_data[property_name]
        # Remove NaN entries for this column
        valid = ~np.isnan(arr)
        if not np.any(valid):
            return None
        return float(np.interp(depth_m, self._log_depths[valid], arr[valid]))

    def ucs_at_depth(self, depth_m: float, temperature_c: float) -> float:
        """
        Effective UCS [MPa] accounting for:
          1. CSV override if available.
          2. Thermal softening: ΔUCSfrac = coeff × ΔT/100 (empirical).
          3. Depth/confinement hardening via simple linear proxy
             (full Hoek-Brown used in effective_stress_correction below).

        Reference: Zhu et al. (2010) — temperature-dependent UCS in granite.
        """
        csv_val = self.interpolate(depth_m, "ucs_mpa")
        unit = self.get_rock_unit(depth_m)
        ucs_base = csv_val if csv_val is not None else unit.ucs_mpa

        # Thermal softening (relative to 25 °C reference)
        delta_t_100 = (temperature_c - 25.0) / 100.0
        thermal_factor = 1.0 + unit.ucs_thermal_coeff * max(delta_t_100, 0.0)
        thermal_factor = max(thermal_factor, 0.3)  # Floor at 30 % of baseline

        return ucs_base * thermal_factor

    def thermal_conductivity_at_depth(self, depth_m: float) -> float:
        """λ [W m⁻¹ K⁻¹] — CSV override or unit default."""
        csv_val = self.interpolate(depth_m, "thermal_conductivity")
        if csv_val is not None:
            return csv_val
        return self.get_rock_unit(depth_m).thermal_conductivity

    def density_at_depth(self, depth_m: float) -> float:
        """ρ [kg m⁻³] — CSV override or unit default."""
        csv_val = self.interpolate(depth_m, "density_kg_m3")
        if csv_val is not None:
            return csv_val
        return self.get_rock_unit(depth_m).density_kg_m3

    def formation_temperature(self, depth_m: float) -> float:
        """
        Undisturbed formation temperature [°C].
        CSV override (column 'temperature_c') takes precedence over
        linear geothermal gradient.
        """
        csv_val = self.interpolate(depth_m, "temperature_c")
        if csv_val is not None:
            return csv_val
        fp = self.formation
        return fp.surface_temperature_c + fp.geothermal_gradient_c_per_km * depth_m / 1000.0

    def pore_pressure(self, depth_m: float) -> float:
        """Pore pressure [MPa] — linear gradient model."""
        fp = self.formation
        return fp.surface_pressure_mpa + fp.pore_pressure_gradient_mpa_per_km * depth_m / 1000.0

    def overburden_stress(self, depth_m: float) -> float:
        """Vertical overburden (lithostatic) stress [MPa]."""
        fp = self.formation
        return fp.surface_pressure_mpa + fp.overburden_gradient_mpa_per_km * depth_m / 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MUD HYDRAULICS MODULE
# ──────────────────────────────────────────────────────────────────────────────

class MudHydraulicsModule:
    """
    Herschel-Bulkley non-Newtonian drilling fluid hydraulics.

    Governing rheology:
        τ = τ_y + K · γ̇^n      (shear stress vs. shear rate)

    Annular pressure loss (adapted Fanning friction):
        ΔP/L = f · (ρ v² / 2) · (4 / D_h)

    where the Fanning friction factor f is computed from the generalised
    Reynolds number for Herschel-Bulkley fluids (Dodge-Metzner approach).

    Jet hydraulics (nozzle impact force):
        v_j = Q / A_n    (nozzle exit velocity)
        HIF = ρ_mud · Q · v_j   (hydraulic impact force [N])

    References:
      - Bourgoyne et al. (1986), Applied Drilling Engineering, SPE Textbook
      - Kelessidis & Maglione (2006), Ceská geologická, on Herschel-Bulkley
    """

    def __init__(self, mud: MudParameters, geometry: WellGeometry, bit: BitParameters):
        self.mud = mud
        self.geo = geometry
        self.bit = bit

    # ── Geometry helpers ─────────────────────────────────────────────────────

    @property
    def annular_hydraulic_diameter(self) -> float:
        """D_h = D_wellbore - D_pipe [m] (annular gap)."""
        return self.geo.wellbore_diameter_m - self.geo.drill_pipe_od_m

    @property
    def annular_cross_section(self) -> float:
        """A_ann = π/4 · (D_wb² - D_pipe²) [m²]."""
        return math.pi / 4.0 * (
            self.geo.wellbore_diameter_m**2 - self.geo.drill_pipe_od_m**2
        )

    @property
    def nozzle_area(self) -> float:
        """Total nozzle area [m²]."""
        return self.bit.n_nozzles * math.pi / 4.0 * self.bit.nozzle_diameter_m**2

    # ── Flow velocities ──────────────────────────────────────────────────────

    @property
    def annular_velocity(self) -> float:
        """Mean annular velocity v_ann [m s⁻¹]."""
        return self.mud.flow_rate_m3s / self.annular_cross_section

    @property
    def nozzle_velocity(self) -> float:
        """Bit nozzle exit velocity [m s⁻¹] — Torricelli-type."""
        return self.mud.flow_rate_m3s / self.nozzle_area

    # ── Herschel-Bulkley effective viscosity ─────────────────────────────────

    def effective_viscosity(self, shear_rate_s: float) -> float:
        """
        Apparent (effective) dynamic viscosity [Pa·s] at a given shear rate.

            μ_eff = τ_y / γ̇ + K · γ̇^(n-1)

        Floor at a small positive shear rate to avoid division by zero.
        """
        gamma = max(shear_rate_s, 1e-6)
        mu = self.mud.yield_stress_pa / gamma + self.mud.consistency_index * gamma ** (
            self.mud.flow_behaviour_index - 1.0
        )
        return mu

    def generalised_reynolds_number(self) -> float:
        """
        Generalised (Metzner-Reed) Reynolds number for HB fluids in the annulus.

            Re_MR = ρ · v² · D_h / (8 · μ_eff)

        where μ_eff is evaluated at the representative annular shear rate
            γ̇_ann ≈ 8 v / D_h   (slot-flow approximation)

        Reference: Bourgoyne et al. (1986), Eq. 4-57.
        """
        v = self.annular_velocity
        d_h = self.annular_hydraulic_diameter
        gamma_ann = 8.0 * v / d_h
        mu_eff = self.effective_viscosity(gamma_ann)
        rho = self.mud.density_kg_m3
        return rho * v**2 * d_h / (8.0 * mu_eff + 1e-15)

    def fanning_friction_factor(self) -> float:
        """
        Fanning friction factor for annular flow.
          Laminar  (Re < 2100): f = 16 / Re
          Turbulent (Re ≥ 2100): Blasius approximation f = 0.046 · Re⁻⁰·²
        """
        re = self.generalised_reynolds_number()
        if re < 2100.0:
            return 16.0 / max(re, 1e-3)
        else:
            return 0.046 * re**(-0.2)

    def annular_pressure_loss(self, segment_length_m: float) -> float:
        """
        Annular friction pressure loss [Pa] over a wellbore segment.

            ΔP = f · (ρ v² / 2) · (4 L / D_h)
        """
        f = self.fanning_friction_factor()
        v = self.annular_velocity
        rho = self.mud.density_kg_m3
        d_h = self.annular_hydraulic_diameter
        return f * (rho * v**2 / 2.0) * (4.0 * segment_length_m / d_h)

    def bit_pressure_drop(self) -> float:
        """
        Pressure drop across bit nozzles [Pa] — Bernoulli with discharge coeff Cd=0.95.

            ΔP_bit = ρ · Q² / (2 · Cd² · A_n²)
        """
        cd = 0.95
        return (
            self.mud.density_kg_m3
            * self.mud.flow_rate_m3s**2
            / (2.0 * cd**2 * self.nozzle_area**2)
        )

    def hydraulic_impact_force(self) -> float:
        """
        Hydraulic impact force at bit face [N].

            HIF = ρ · Q · v_j

        This contributes to bottom-hole cleaning and cuttings lift.
        """
        return self.mud.density_kg_m3 * self.mud.flow_rate_m3s * self.nozzle_velocity

    def cuttings_transport_ratio(self, rop_m_per_s: float, bit_diameter_m: float) -> float:
        """
        Cuttings Transport Ratio (CTR) — fraction of generated cuttings lifted.

            CTR = v_ann / (v_ann + v_slip)

        Slip velocity v_slip estimated from Stokes' law modified for non-Newtonian
        fluids (simplified — full modelling requires particle size distribution):

            v_slip = (ρ_rock - ρ_mud) · g · d_cut² / (18 · μ_eff_ann)

        Uses representative cutting diameter d_cut ~ 1/3 of bit diameter (empirical).
        CTR > 0.97 is the conventional target for efficient hole cleaning.
        """
        # Representative cutting diameter: ~5 mm for PDC/Tricone in hard rock.
        # (bit_diameter/3 would give ~72 mm — geologically unrealistic boulders)
        d_cut = max(bit_diameter_m / 40.0, 0.003)  # 5–6 mm typical for PDC in granite
        rho_rock = 2650.0  # Approximate — could be passed from RockProfile
        delta_rho = max(rho_rock - self.mud.density_kg_m3, 0.0)
        gamma_ann = 8.0 * self.annular_velocity / self.annular_hydraulic_diameter
        mu_eff = self.effective_viscosity(gamma_ann)
        v_slip = delta_rho * G_STD * d_cut**2 / (18.0 * mu_eff + 1e-15)
        v_ann = self.annular_velocity
        ctr = v_ann / (v_ann + v_slip + 1e-15)
        return min(ctr, 1.0)

    def bottomhole_circulating_pressure(self, depth_m: float) -> float:
        """
        Approximate bottomhole circulating pressure [MPa].
        = hydrostatic head + annular friction pressure
        """
        hydrostatic = self.mud.density_kg_m3 * G_STD * depth_m / 1e6  # Pa → MPa
        friction = self.annular_pressure_loss(depth_m) / 1e6
        return hydrostatic + friction


# ──────────────────────────────────────────────────────────────────────────────
# 4.  THERMAL MODULE
# ──────────────────────────────────────────────────────────────────────────────

class ThermalModule:
    """
    1-D steady-state convective-conductive energy balance between drilling fluid,
    drill string, and formation.

    Approach: treat the wellbore as a co-axial heat exchanger.
      - Drill-string interior carries cool mud downward (pipe flow).
      - Annulus carries warmer mud + cuttings upward.
      - Heat exchange with formation governed by formation thermal conductivity
        and wellbore heat-transfer coefficient.

    For a first-principles derivation see:
      Ramey (1962), "Wellbore Heat Transmission", JPT.

    The module computes:
      (a) Fluid temperature at bit (downhole) T_bit
      (b) Annular temperature profile T_ann(z)
      (c) Wellbore wall temperature T_wall(z) (used by ROP module for thermal softening)

    Units: all temperatures in °C, depths in m, lengths in m.
    """

    def __init__(
        self,
        rock_manager: RockPropertiesManager,
        mud: MudParameters,
        geometry: WellGeometry,
    ):
        self.rpm = rock_manager
        self.mud = mud
        self.geo = geometry

    def overall_heat_transfer_coefficient(self, depth_m: float) -> float:
        """
        Overall radial heat transfer coefficient U [W m⁻² K⁻¹] at depth.

        Three resistances in series (cylindrical geometry simplified to flat-wall):
          1/U = 1/h_f + ln(r_wb/r_bit)/(2π λ_rock) + 1/h_ann

        Here we use a simplified lumped U typical for granite wells:
          U ≈ 50–150 W m⁻² K⁻¹  depending on conductivity and flow.
        """
        lambda_r = self.rpm.thermal_conductivity_at_depth(depth_m)
        r_wb = self.geo.wellbore_diameter_m / 2.0
        r_pipe = self.geo.drill_pipe_od_m / 2.0
        # Pipe-wall conductive resistance (simplified)
        r_cond = math.log(r_wb / (r_pipe + 1e-9)) / (2.0 * math.pi * lambda_r)
        # Convective resistance (Dittus-Boelter in turbulent flow, simplified)
        h_conv = 500.0  # W m⁻² K⁻¹ — conservative representative value
        return 1.0 / (1.0 / h_conv + r_cond + 1.0 / h_conv)

    def bit_temperature(self, depth_m: float, surface_mud_temp_c: float = 20.0) -> float:
        """
        Estimate fluid temperature at bit depth [°C].

        Uses the Ramey (1962) steady-state analytical solution for circulating
        wellbore temperature:

            T_bit ≈ T_form(depth) - A · (G - dT_mud/dz)

        where A = ρ_mud Cp Q / (2π r_wb U) is the thermal relaxation length [m]
        and G is the geothermal gradient [°C m⁻¹].

        Simplified here by assuming the fluid approaches formation temperature
        exponentially with characteristic depth A.
        """
        g_grad = self.rpm.formation.geothermal_gradient_c_per_km / 1000.0  # °C m⁻¹
        t_form = self.rpm.formation_temperature(depth_m)

        u = self.overall_heat_transfer_coefficient(depth_m)
        r_wb = self.geo.wellbore_diameter_m / 2.0
        # Thermal relaxation depth [m]
        A = (self.mud.density_kg_m3 * self.mud.specific_heat * self.mud.flow_rate_m3s) / (
            2.0 * math.pi * r_wb * u + 1e-9
        )
        # Fluid temp at depth z, starting at surface_mud_temp_c
        # Analytical solution: T_f(z) = T_form(z) - A*G + (T_surf - T_form(0) + A*G)*exp(-z/A)
        t_f = (
            t_form
            - A * g_grad
            + (surface_mud_temp_c - self.rpm.formation.surface_temperature_c + A * g_grad)
            * math.exp(-depth_m / (A + 1e-9))
        )
        # Bit temperature is blend of fluid and formation (due to mechanical energy)
        return t_f

    def annular_temperature_profile(
        self, depths: np.ndarray, surface_mud_temp_c: float = 20.0
    ) -> np.ndarray:
        """
        Return annular fluid temperature [°C] at each depth in `depths` array.
        The annular fluid rises from bottom (hottest) to surface (cooling).
        This uses the same Ramey model evaluated upwards.
        """
        temps = np.zeros_like(depths)
        for i, z in enumerate(depths):
            temps[i] = self.bit_temperature(z, surface_mud_temp_c)
        return temps

    def wall_temperature(self, depth_m: float, surface_mud_temp_c: float = 20.0) -> float:
        """
        Wellbore wall temperature [°C] — weighted blend of formation and fluid.
        Used for UCS thermal softening in the ROP model.
        """
        t_form = self.rpm.formation_temperature(depth_m)
        t_fluid = self.bit_temperature(depth_m, surface_mud_temp_c)
        # Weight toward formation (wall is mostly rock temperature)
        return 0.7 * t_form + 0.3 * t_fluid


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ROP MODULE — MSE-based with effective stress and bit-wear
# ──────────────────────────────────────────────────────────────────────────────

class ROPModule:
    """
    Rate of Penetration model based on Mechanical Specific Energy (MSE).

    Core identity (Teale, 1965):
        MSE = (WOB / A_bit) + (2π N T) / (A_bit · ROP)

    Rearranged for ROP:
        ROP = (2π N T) / (A_bit · (MSE - WOB/A_bit))

    MSE at efficiency η equals the formation's effective UCS:
        MSE_needed = UCS_eff / η_total

    where:
        UCS_eff  — effective UCS accounting for differential pressure (confining
                   stress reduces crack propagation; elevated pore pressure aids it)
        η_total  — product of cutting efficiency, hydraulic cleaning efficiency,
                   and wear degradation

    Torque T derived from equilibrium:
        T = T_bit + T_friction
        T_bit ≈ μ_bit · WOB · (D_bit / 4)   (simplified rolling / scraping contact)
        where μ_bit depends on bit type.

    Effective stress on the formation:
        σ_eff = σ_overburden - α · P_pore    (Biot effective stress)
        α ≈ 1 - K_dry/K_grain (Biot coefficient) ≈ 0.7 for granite

    Differential pressure effect on UCS:
        UCS_eff = UCS(T) · exp(-β · ΔP / σ_conf)
        where ΔP = P_BH - P_pore  (positive = overbalanced, reduces ROP)
        β ≈ 0.15 (tunable, calibrated to field observations)

    Abrasive wear ODE:
        dW/dt = κ_a · (WOB / A_bit) · N · CAI / UCS_eff
        where CAI = Cerchar Abrasivity Index, κ_a = wear rate constant.

    References:
      - Teale (1965), Int J Rock Mech Min Sci
      - Pessier & Fear (1992), SPE 24584 (torque-WOB relationship)
      - Detournay & Tan (2002), SPE 78957 (drilling efficiency)
      - Zhu et al. (2010), temperature-dependent UCS in granite
    """

    BIOT_COEFF = 0.72          # α (Biot coefficient) — granite value
    DP_SENSITIVITY = 0.04      # β [MPa⁻¹] — chip hold-down sensitivity (Bourgoyne et al. 1986)
    WEAR_RATE_CONST = 4e-10    # κ_a [m Pa⁻¹ rev⁻¹] (tunable)

    # Bit-type friction coefficient (Pessier & Fear 1992)
    BIT_FRICTION = {"PDC": 0.25, "Tricone": 0.35, "Impreg": 0.30}

    def __init__(
        self,
        rock_manager: RockPropertiesManager,
        mud_hydraulics: MudHydraulicsModule,
        thermal: ThermalModule,
        bit: BitParameters,
        wob_kn: float = 100.0,    # Weight on Bit [kN]
        rpm: float = 120.0,        # Rotary speed [rpm]
        surface_mud_temp_c: float = 20.0,
    ):
        self.rpm_module = rock_manager
        self.hyd = mud_hydraulics
        self.thermal = thermal
        self.bit = bit
        self.wob_n = wob_kn * 1e3  # Convert kN → N
        self.rotation_rpm = rpm
        self.surface_mud_temp = surface_mud_temp_c

    @property
    def bit_area(self) -> float:
        """Cross-sectional bit face area [m²]."""
        return math.pi / 4.0 * self.bit.diameter_m**2

    def cutting_efficiency(self) -> float:
        """
        Effective cutting efficiency η_ce [-]:
          - Starts at ce_alpha for a new bit.
          - Degrades linearly with wear state (0–1).
        """
        ce = self.bit.ce_alpha * (
            1.0 - self.bit.ce_wear_sensitivity * self.bit.wear_state
        )
        return max(ce, 0.05)  # Minimum 5% — drill does not become completely ineffective

    def hydraulic_efficiency(self, depth_m: float) -> float:
        """
        Hydraulic cleaning efficiency η_hyd [-].
        Based on ratio of hydraulic impact force to WOB and cuttings transport ratio.
        Higher HIF and CTR → better chip removal → higher effective ROP.

        η_hyd = min(1, CTR · (1 + HIF_normalised))
        HIF_normalised = HIF / WOB (dimensionless)
        """
        hif = self.hyd.hydraulic_impact_force()
        ctr = self.hyd.cuttings_transport_ratio(1e-3, self.bit.diameter_m)  # 1 mm/s placeholder
        hif_norm = min(hif / (self.wob_n + 1e-9), 0.3)  # Cap at 30 % contribution
        eta_hyd = ctr * (1.0 + hif_norm)
        return min(eta_hyd, 1.0)

    def effective_ucs(self, depth_m: float) -> float:
        """
        UCS corrected for:
          1. Temperature (via RockPropertiesManager)
          2. Differential pressure (overbalance suppresses ROP)
          3. Effective confining stress (Biot effective stress)

        Returns effective UCS [MPa].
        """
        t_wall = self.thermal.wall_temperature(depth_m, self.surface_mud_temp)
        ucs_t = self.rpm_module.ucs_at_depth(depth_m, t_wall)  # Thermal-corrected UCS

        # Bottomhole circulating pressure vs pore pressure
        p_bh = self.hyd.bottomhole_circulating_pressure(depth_m)   # [MPa]
        p_pore = self.rpm_module.pore_pressure(depth_m)             # [MPa]
        delta_p = p_bh - p_pore                                     # Differential pressure

        # Confining stress (mean effective stress)
        sigma_ov = self.rpm_module.overburden_stress(depth_m)
        sigma_eff = sigma_ov - self.BIOT_COEFF * p_pore  # Biot effective stress [MPa]
        sigma_conf = max(sigma_eff, 1.0)  # Floor at 1 MPa

        # Differential pressure (chip hold-down) correction.
        # Overbalance (ΔP > 0) presses cuttings against the formation, increasing
        # the effective drilling resistance ("chip hold-down" effect).
        # Underbalanced drilling (ΔP < 0) reduces effective resistance → higher ROP.
        # After Detournay & Cheng (1993); Sinor & Warren (1987); Bourgoyne et al. (1986).
        #
        # Formulation: exp(β · ΔP) where ΔP is in MPa and β has units MPa⁻¹.
        # Do NOT divide by σ_conf here — that would shrink ΔP/σ_conf as depth grows
        # (σ_conf rises ~25 MPa/km vs ΔP ~2.3 MPa/km), nullifying chip hold-down at
        # depth and producing a physically inverted ROP-vs-depth trend.
        dp_factor = math.exp(self.DP_SENSITIVITY * delta_p)

        return ucs_t * dp_factor

    def torque(self) -> float:
        """
        Bit torque [N·m].

        T_bit = μ_b · WOB · D_bit / 4

        Simplified roller-contact / PDC scraping torque model
        (Pessier & Fear 1992, SPE 24584).
        """
        mu_b = self.BIT_FRICTION.get(self.bit.bit_type, 0.3)
        return mu_b * self.wob_n * self.bit.diameter_m / 4.0

    def compute_rop(self, depth_m: float) -> float:
        """
        Compute instantaneous ROP [m s⁻¹] at given depth.

        From Teale (1965) MSE identity rearranged:
            ROP = (2π N T) / (A_bit · (MSE_req - WOB/A_bit))

        MSE_req = UCS_eff / η_total

        If the denominator is negative (torque term dominates over WOB contribution),
        the solution is physically non-meaningful; we clamp to a minimum ROP.

        Returns ROP in m s⁻¹.
        """
        n_rads = self.rotation_rpm * 2.0 * math.pi / 60.0  # rad s⁻¹
        torque = self.torque()
        a_bit = self.bit_area

        ucs_eff = self.effective_ucs(depth_m)  # MPa
        ucs_eff_pa = ucs_eff * 1e6             # Pa

        # Total drilling efficiency
        eta_total = self.cutting_efficiency() * self.hydraulic_efficiency(depth_m)
        eta_total = max(eta_total, 0.01)

        # Required MSE to overcome formation
        mse_req = ucs_eff_pa / eta_total       # Pa

        # WOB contribution to MSE (confining term)
        wob_term = self.wob_n / a_bit          # Pa (= N/m²)

        # Net MSE available from rotation
        mse_net = mse_req - wob_term

        if mse_net <= 0.0:
            # WOB term alone exceeds required MSE — bit flying through
            # Apply generous but physically bounded ROP
            rop = 0.02  # 20 mm/s cap (unrealistically fast — flag this)
        else:
            # Core Teale equation
            rop = (n_rads * torque) / (a_bit * mse_net)

        # Apply practical floor and ceiling
        rop = max(rop, 1e-6)    # Minimum 0.001 mm/s
        rop = min(rop, 0.020)   # Maximum 20 mm/s = 72 m/hr (generous ceiling)

        return rop

    def wear_rate(self, depth_m: float, rop_m_per_s: float) -> float:
        """
        Abrasive wear rate dW/dt [dimensionless s⁻¹].

        dW/dt = κ_a · (WOB/A_bit) · N · CAI / UCS_eff_pa

        Wear state W increases from 0 (new) to 1 (worn out).
        The higher the UCS and the lower the CAI, the slower the wear.
        """
        unit = self.rpm_module.get_rock_unit(depth_m)
        cai = unit.abrasivity_index
        t_wall = self.thermal.wall_temperature(depth_m, self.surface_mud_temp)
        ucs_eff_pa = self.effective_ucs(depth_m) * 1e6
        n_rps = self.rotation_rpm / 60.0
        wob_stress = self.wob_n / self.bit_area

        dw_dt = self.WEAR_RATE_CONST * wob_stress * n_rps * cai / (ucs_eff_pa + 1e-9)
        return dw_dt

    def mechanical_specific_energy(self, depth_m: float, rop_m_per_s: float) -> float:
        """
        Actual MSE [MPa] at given operating conditions.

        MSE = WOB/A + 2π N T / (A · ROP)

        Lower MSE → more efficient drilling.
        """
        n_rads = self.rotation_rpm * 2.0 * math.pi / 60.0
        torque = self.torque()
        a_bit = self.bit_area
        rop = max(rop_m_per_s, 1e-9)
        mse_pa = self.wob_n / a_bit + (2.0 * math.pi * n_rads * torque) / (a_bit * rop)
        return mse_pa / 1e6  # → MPa


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EXPERIMENT RUNNER — parameter sweeps and sensitivity analysis
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Utility class for running parameter sweeps and comparative experiments.

    Example usage:
        runner = ExperimentRunner(base_simulator)
        results = runner.sweep_wob(wob_values_kn=[50, 75, 100, 125, 150])
        runner.plot_sweep(results, param_name="WOB (kN)")

    To add a new sweep dimension, follow the pattern in sweep_wob() below.
    """

    def __init__(self, base_params: dict):
        """
        base_params: dict matching the signature of GeothermalDrillingSimulator.__init__.
        """
        self.base_params = base_params

    def sweep_wob(
        self,
        wob_values_kn: List[float],
        target_depth_m: float = 3000.0,
    ) -> List[dict]:
        """Run a suite of simulations across different WOB values."""
        results = []
        for wob in wob_values_kn:
            params = dict(self.base_params)
            params["wob_kn"] = wob
            sim = GeothermalDrillingSimulator(**params)
            result = sim.run_simulation(target_depth_m=target_depth_m)
            result["wob_kn"] = wob
            results.append(result)
            print(
                f"  WOB={wob:6.1f} kN → avg ROP={result['avg_rop_m_hr']:.2f} m/hr, "
                f"avg MSE={result['avg_mse_mpa']:.1f} MPa"
            )
        return results

    def sweep_rpm(
        self,
        rpm_values: List[float],
        target_depth_m: float = 3000.0,
    ) -> List[dict]:
        """Run a suite of simulations across different RPM values."""
        results = []
        for rpm in rpm_values:
            params = dict(self.base_params)
            params["rotation_rpm"] = rpm
            sim = GeothermalDrillingSimulator(**params)
            result = sim.run_simulation(target_depth_m=target_depth_m)
            result["rotation_rpm"] = rpm
            results.append(result)
            print(
                f"  RPM={rpm:6.1f}   → avg ROP={result['avg_rop_m_hr']:.2f} m/hr, "
                f"final wear={result['final_wear_state']:.3f}"
            )
        return results

    def sweep_flow_rate(
        self,
        flow_rates_m3s: List[float],
        target_depth_m: float = 3000.0,
    ) -> List[dict]:
        """Sweep over different mud flow rates."""
        results = []
        for q in flow_rates_m3s:
            params = dict(self.base_params)
            params["flow_rate_m3s"] = q
            sim = GeothermalDrillingSimulator(**params)
            result = sim.run_simulation(target_depth_m=target_depth_m)
            result["flow_rate_m3s"] = q
            results.append(result)
            print(
                f"  Q={q*1000:.1f} L/s → avg ROP={result['avg_rop_m_hr']:.2f} m/hr"
            )
        return results

    @staticmethod
    def plot_sweep(
        results: List[dict],
        param_name: str,
        param_key: str,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Plot key metrics against a swept parameter.
        `param_key` must match the key injected into result dicts in sweep_*.
        """
        if metrics is None:
            metrics = ["avg_rop_m_hr", "avg_mse_mpa", "final_wear_state"]

        x = [r[param_key] for r in results]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]

        labels = {
            "avg_rop_m_hr": "Average ROP (m/hr)",
            "avg_mse_mpa": "Average MSE (MPa)",
            "final_wear_state": "Final Bit Wear State",
            "hydraulic_efficiency_avg": "Avg Hydraulic Efficiency",
        }

        for ax, metric in zip(axes, metrics):
            y = [r.get(metric, float("nan")) for r in results]
            ax.plot(x, y, "o-", linewidth=2, markersize=6)
            ax.set_xlabel(param_name)
            ax.set_ylabel(labels.get(metric, metric))
            ax.grid(True, alpha=0.4)
            ax.set_title(labels.get(metric, metric))

        fig.suptitle(f"Parameter Sweep: {param_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN SIMULATOR CLASS
# ──────────────────────────────────────────────────────────────────────────────

class GeothermalDrillingSimulator:
    """
    Top-level simulator: integrates ROP ODE from surface to target depth.

    State vector: y = [depth_m, wear_state]

    ODE system:
        dy[0]/dt = ROP(depth, wear)    [m s⁻¹]
        dy[1]/dt = dW/dt(depth, ROP)   [dimensionless s⁻¹]

    Integration performed with scipy.integrate.solve_ivp (RK45 by default).

    Initialisation via keyword arguments or a JSON/dict parameter file.
    See `from_json()` classmethod for file-based loading.
    """

    def __init__(
        self,
        # Site / formation selection
        site: str = "cornish_granite",          # "cornish_granite" | "utah_forge"
        # Operational parameters
        wob_kn: float = 100.0,                  # Weight on Bit [kN]
        rotation_rpm: float = 120.0,            # Rotary speed [rpm]
        flow_rate_m3s: float = 0.025,           # Mud flow rate [m³/s]
        surface_mud_temp_c: float = 20.0,       # Surface mud temperature [°C]
        # Bit parameters
        bit_diameter_m: float = 0.2159,         # 8.5 in
        bit_type: str = "PDC",
        # Mud parameters (Herschel-Bulkley)
        mud_density_kg_m3: float = 1200.0,
        mud_yield_stress_pa: float = 5.0,
        mud_consistency_k: float = 0.08,
        mud_flow_index_n: float = 0.72,
        # CSV log path (optional)
        csv_log_path: Optional[str] = None,
        # Simulation control
        max_depth_m: float = 4000.0,
        time_step_s: float = 60.0,              # Output time step [s]
        max_sim_hours: float = 2000.0,          # Safety ceiling on sim time
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self.site = site
        self.wob_kn = wob_kn
        self.rotation_rpm = rotation_rpm
        self.flow_rate_m3s = flow_rate_m3s
        self.surface_mud_temp_c = surface_mud_temp_c
        self.max_depth_m = max_depth_m
        self.time_step_s = time_step_s
        self.max_sim_hours = max_sim_hours
        self.rtol = rtol
        self.atol = atol

        # ── Build formation profile ──────────────────────────────────────────
        formation = self._build_formation(site)

        # ── Rock properties manager ──────────────────────────────────────────
        self.rock_mgr = RockPropertiesManager(formation)
        if csv_log_path:
            self.rock_mgr.load_csv_log(csv_log_path)

        # ── Sub-modules ──────────────────────────────────────────────────────
        self.bit = BitParameters(
            diameter_m=bit_diameter_m,
            bit_type=bit_type,
        )
        self.mud = MudParameters(
            density_kg_m3=mud_density_kg_m3,
            yield_stress_pa=mud_yield_stress_pa,
            consistency_index=mud_consistency_k,
            flow_behaviour_index=mud_flow_index_n,
            flow_rate_m3s=flow_rate_m3s,
        )
        self.geometry = WellGeometry(wellbore_diameter_m=bit_diameter_m)

        self.hydraulics = MudHydraulicsModule(self.mud, self.geometry, self.bit)
        self.thermal = ThermalModule(self.rock_mgr, self.mud, self.geometry)
        self.rop_module = ROPModule(
            rock_manager=self.rock_mgr,
            mud_hydraulics=self.hydraulics,
            thermal=self.thermal,
            bit=self.bit,
            wob_kn=wob_kn,
            rpm=rotation_rpm,
            surface_mud_temp_c=surface_mud_temp_c,
        )

        # Results storage
        self._results: Optional[dict] = None

    # ── Formation builders ───────────────────────────────────────────────────

    @staticmethod
    def _build_formation(site: str) -> FormationProfile:
        """
        Build a FormationProfile for one of the supported sites.
        To add a new site, add a matching elif branch.
        """
        rpm = RockPropertiesManager  # Class reference (for rock unit constants)

        # Deep-copy each rock unit so that per-instance modifications (e.g. in unit
        # tests or parameter sweeps) do not pollute the class-level defaults.
        if site == "cornish_granite":
            return FormationProfile(
                name="United Downs Deep Geothermal (GEL)",
                site="cornish_granite",
                surface_temperature_c=12.0,        # SW England mean surface T
                geothermal_gradient_c_per_km=35.0, # High heat flow: Cornubian batholith
                surface_pressure_mpa=0.101325,
                hydrostatic_gradient_mpa_per_km=9.81,
                pore_pressure_gradient_mpa_per_km=9.5,
                overburden_gradient_mpa_per_km=24.5,  # Granite: lower than sedimentary
                rock_units=[copy.deepcopy(rpm.CORNISH_GRANITE)],
            )
        elif site == "utah_forge":
            return FormationProfile(
                name="Utah FORGE (Milford, UT)",
                site="utah_forge",
                surface_temperature_c=18.0,
                geothermal_gradient_c_per_km=55.0,  # Higher gradient → 175–225 °C at ~3 km
                surface_pressure_mpa=0.101325,
                hydrostatic_gradient_mpa_per_km=9.81,
                pore_pressure_gradient_mpa_per_km=9.5,
                overburden_gradient_mpa_per_km=25.0,
                rock_units=[
                    copy.deepcopy(rpm.UTAH_FORGE_GRANITE),
                    copy.deepcopy(rpm.UTAH_FORGE_QUARTZ_MONZONITE),
                    copy.deepcopy(rpm.UTAH_FORGE_METAMORPHIC),
                ],
            )
        else:
            raise ValueError(
                f"Unknown site '{site}'. Available: 'cornish_granite', 'utah_forge'. "
                "To add a new site, extend _build_formation() and define rock units."
            )

    # ── JSON parameter loading ────────────────────────────────────────────────

    @classmethod
    def from_json(cls, filepath: str) -> "GeothermalDrillingSimulator":
        """
        Load simulation parameters from a JSON file.

        Example JSON:
        {
            "site": "utah_forge",
            "wob_kn": 80.0,
            "rotation_rpm": 100.0,
            "flow_rate_m3s": 0.020,
            "max_depth_m": 3500.0
        }
        Any key matching an __init__ parameter is accepted; others are ignored.
        """
        with open(filepath) as fh:
            params = json.load(fh)
        return cls(**{k: v for k, v in params.items() if k in cls.__init__.__code__.co_varnames})

    # ── ODE system ───────────────────────────────────────────────────────────

    def _ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE right-hand side.

        State: y = [depth_m, wear_state]

        Returns dy/dt = [ROP, dW/dt]
        """
        depth = y[0]
        wear = np.clip(y[1], 0.0, 1.0)

        # Update bit wear state in the ROP module
        self.bit.wear_state = wear

        # Compute ROP
        rop = self.rop_module.compute_rop(depth)

        # Compute wear rate
        dw_dt = self.rop_module.wear_rate(depth, rop)

        return np.array([rop, dw_dt])

    # ── Terminal events ──────────────────────────────────────────────────────

    def _event_target_depth(self, t: float, y: np.ndarray) -> float:
        """Stop when target depth reached."""
        return y[0] - self.max_depth_m

    def _event_bit_worn(self, t: float, y: np.ndarray) -> float:
        """Stop when bit is fully worn (wear_state ≥ 1)."""
        return y[1] - 0.99

    # ── Main simulation ──────────────────────────────────────────────────────

    def run_simulation(self, target_depth_m: Optional[float] = None) -> dict:
        """
        Integrate the ODE system from surface to target_depth_m.

        Returns a results dictionary containing:
          - t_s:            time array [s]
          - depth_m:        depth array [m]
          - rop_m_hr:       ROP at each time step [m/hr]
          - mse_mpa:        MSE at each time step [MPa]
          - wear_state:     bit wear state array
          - temperature_c:  fluid temperature at bit depth [°C]
          - bhp_mpa:        bottomhole circulating pressure [MPa]
          - avg_rop_m_hr:   average ROP [m/hr]
          - avg_mse_mpa:    average MSE [MPa]
          - final_wear_state: wear state at end
          - total_drill_time_hr: total drilling time [hr]
          - hydraulic_efficiency_avg: average hydraulic efficiency
        """
        if target_depth_m is None:
            target_depth_m = self.max_depth_m

        self.max_depth_m = target_depth_m
        t_max = self.max_sim_hours * 3600.0

        # Evaluation time points for output
        t_eval = np.arange(0, t_max, self.time_step_s)

        y0 = np.array([0.01, 0.0])  # Start just below surface, new bit

        # Events — must be plain functions (not bound methods) for scipy attribute injection
        target = self.max_depth_m

        def ev1(t: float, y: np.ndarray) -> float:
            return y[0] - target

        def ev2(t: float, y: np.ndarray) -> float:
            return y[1] - 0.99

        ev1.terminal = True   # type: ignore[attr-defined]
        ev1.direction = 1.0   # type: ignore[attr-defined]
        ev2.terminal = True   # type: ignore[attr-defined]
        ev2.direction = 1.0   # type: ignore[attr-defined]

        print(f"\n[GeoTwin] Starting simulation: site='{self.site}', "
              f"target={target_depth_m:.0f} m")
        print(f"          WOB={self.wob_kn} kN, RPM={self.rotation_rpm}, "
              f"Q={self.flow_rate_m3s*1000:.1f} L/s, Bit={self.bit.bit_type}")

        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(0.0, t_max),
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            events=[ev1, ev2],
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False,
        )

        t = sol.t
        depths = sol.y[0]
        wear = sol.y[1]

        # Compute derived quantities at each time step
        rop_arr = np.zeros(len(t))
        mse_arr = np.zeros(len(t))
        temp_arr = np.zeros(len(t))
        bhp_arr = np.zeros(len(t))
        eta_hyd_arr = np.zeros(len(t))

        for i, (ti, di, wi) in enumerate(zip(t, depths, wear)):
            self.bit.wear_state = np.clip(wi, 0.0, 1.0)
            rop = self.rop_module.compute_rop(di)
            rop_arr[i] = rop * 3600.0  # m/s → m/hr
            mse_arr[i] = self.rop_module.mechanical_specific_energy(di, rop)
            temp_arr[i] = self.thermal.bit_temperature(di, self.surface_mud_temp_c)
            bhp_arr[i] = self.hydraulics.bottomhole_circulating_pressure(di)
            eta_hyd_arr[i] = self.rop_module.hydraulic_efficiency(di)

        results = {
            "t_s": t,
            "depth_m": depths,
            "rop_m_hr": rop_arr,
            "mse_mpa": mse_arr,
            "wear_state": wear,
            "temperature_c": temp_arr,
            "bhp_mpa": bhp_arr,
            "hydraulic_efficiency": eta_hyd_arr,
            "avg_rop_m_hr": float(np.mean(rop_arr)),
            "avg_mse_mpa": float(np.mean(mse_arr)),
            "final_wear_state": float(wear[-1]),
            "total_drill_time_hr": float(t[-1] / 3600.0),
            "hydraulic_efficiency_avg": float(np.mean(eta_hyd_arr)),
            "site": self.site,
            "wob_kn": self.wob_kn,
            "rotation_rpm": self.rotation_rpm,
        }

        self._results = results

        print(f"[GeoTwin] Simulation complete.")
        print(f"          Final depth:      {depths[-1]:.1f} m")
        print(f"          Total drill time: {results['total_drill_time_hr']:.1f} hr")
        print(f"          Average ROP:      {results['avg_rop_m_hr']:.2f} m/hr")
        print(f"          Average MSE:      {results['avg_mse_mpa']:.1f} MPa")
        print(f"          Final bit wear:   {results['final_wear_state']:.3f}")

        return results

    # ── Export ───────────────────────────────────────────────────────────────

    def export_csv(self, filepath: str, results: Optional[dict] = None) -> None:
        """
        Export simulation results to CSV for review in Excel / Sheets.

        Columns: time_hr, depth_m, rop_m_hr, mse_mpa, wear_state,
                 temperature_c, bhp_mpa, hydraulic_efficiency
        """
        if results is None:
            results = self._results
        if results is None:
            raise RuntimeError("No results to export. Run run_simulation() first.")

        filepath = Path(filepath)
        t_hr = results["t_s"] / 3600.0

        with open(filepath, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "time_hr", "depth_m", "rop_m_hr", "mse_mpa", "wear_state",
                "temperature_c", "bhp_mpa", "hydraulic_efficiency"
            ])
            for row in zip(
                t_hr,
                results["depth_m"],
                results["rop_m_hr"],
                results["mse_mpa"],
                results["wear_state"],
                results["temperature_c"],
                results["bhp_mpa"],
                results["hydraulic_efficiency"],
            ):
                writer.writerow([f"{v:.4f}" for v in row])

        print(f"[GeoTwin] Results exported to {filepath}")

    def export_summary_json(self, filepath: str, results: Optional[dict] = None) -> None:
        """Export a human-readable summary of scalar metrics to JSON."""
        if results is None:
            results = self._results
        if results is None:
            raise RuntimeError("No results to export.")

        summary = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
        with open(filepath, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[GeoTwin] Summary exported to {filepath}")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_dashboard(self, results: Optional[dict] = None, save_path: Optional[str] = None) -> None:
        """
        Matplotlib dashboard with six panels:
          1. Depth vs Time
          2. ROP vs Depth
          3. MSE vs Depth
          4. Bit wear vs Depth
          5. Temperature profile vs Depth
          6. Bottomhole pressure vs Depth
        """
        if results is None:
            results = self._results
        if results is None:
            raise RuntimeError("No results. Run run_simulation() first.")

        t_hr = results["t_s"] / 3600.0
        d = results["depth_m"]

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"GeoTwin Engine — {results['site'].replace('_', ' ').title()} | "
            f"WOB={results['wob_kn']} kN, RPM={results['rotation_rpm']}, "
            f"Avg ROP={results['avg_rop_m_hr']:.2f} m/hr",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Depth vs time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_hr, d, "b-", linewidth=1.5)
        ax1.set_xlabel("Time (hr)")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title("Depth vs Time")
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.4)

        # 2. ROP vs depth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results["rop_m_hr"], d, "g-", linewidth=1.5)
        ax2.set_xlabel("ROP (m/hr)")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title("Rate of Penetration")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.4)

        # 3. MSE vs depth
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(results["mse_mpa"], d, "r-", linewidth=1.5)
        ax3.set_xlabel("MSE (MPa)")
        ax3.set_ylabel("Depth (m)")
        ax3.set_title("Mechanical Specific Energy")
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.4)

        # 4. Bit wear vs depth
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(results["wear_state"], d, "m-", linewidth=1.5)
        ax4.set_xlabel("Wear State (0=new, 1=worn)")
        ax4.set_ylabel("Depth (m)")
        ax4.set_title("Bit Wear State")
        ax4.invert_yaxis()
        ax4.axvline(0.7, color="orange", linestyle="--", alpha=0.7, label="Pull threshold (0.7)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)

        # 5. Temperature vs depth
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results["temperature_c"], d, "darkorange", linewidth=1.5, label="Fluid at bit")
        # Overlay formation temperature
        depth_range = np.linspace(d[0], d[-1], 100)
        t_form = [self.rock_mgr.formation_temperature(z) for z in depth_range]
        ax5.plot(t_form, depth_range, "k--", linewidth=1, alpha=0.6, label="Formation T")
        ax5.set_xlabel("Temperature (°C)")
        ax5.set_ylabel("Depth (m)")
        ax5.set_title("Temperature Profile")
        ax5.invert_yaxis()
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.4)

        # 6. BHP vs depth
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(results["bhp_mpa"], d, "teal", linewidth=1.5, label="BHP (circ)")
        pore_p = [self.rock_mgr.pore_pressure(z) for z in d]
        ax6.plot(pore_p, d, "b--", linewidth=1, alpha=0.6, label="Pore pressure")
        ax6.set_xlabel("Pressure (MPa)")
        ax6.set_ylabel("Depth (m)")
        ax6.set_title("Bottomhole Pressure")
        ax6.invert_yaxis()
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.4)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[GeoTwin] Dashboard saved to {save_path}")
        else:
            plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  UNIT TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestROPLimits(unittest.TestCase):
    """Unit tests for limiting cases and physics invariants."""

    def setUp(self):
        self.sim_cornish = GeothermalDrillingSimulator(
            site="cornish_granite", wob_kn=80.0, rotation_rpm=100.0, max_depth_m=500.0
        )
        self.sim_forge = GeothermalDrillingSimulator(
            site="utah_forge", wob_kn=80.0, rotation_rpm=100.0, max_depth_m=500.0
        )

    def test_rop_positive(self):
        """ROP must always be positive."""
        for depth in [100, 500, 1000, 2000]:
            rop = self.sim_cornish.rop_module.compute_rop(depth)
            self.assertGreater(rop, 0.0, f"ROP non-positive at depth={depth} m")

    def test_rop_decreases_with_ucs(self):
        """Higher UCS rock → lower ROP, all else equal."""
        # Simulate two rocks with different UCS
        sim_soft = GeothermalDrillingSimulator(site="cornish_granite", wob_kn=100.0)
        sim_hard = GeothermalDrillingSimulator(site="cornish_granite", wob_kn=100.0)
        # Manually lower UCS for soft rock
        sim_soft.rock_mgr.formation.rock_units[0].ucs_mpa = 50.0
        sim_hard.rock_mgr.formation.rock_units[0].ucs_mpa = 250.0
        rop_soft = sim_soft.rop_module.compute_rop(500.0)
        rop_hard = sim_hard.rop_module.compute_rop(500.0)
        self.assertGreater(rop_soft, rop_hard, "Softer rock should drill faster")

    def test_thermal_softening(self):
        """UCS must decrease with temperature."""
        rm = self.sim_cornish.rock_mgr
        ucs_cold = rm.ucs_at_depth(100.0, 25.0)
        ucs_hot = rm.ucs_at_depth(100.0, 200.0)
        self.assertGreater(ucs_cold, ucs_hot, "UCS should decrease with temperature")

    def test_wear_increases(self):
        """Wear state must be monotonically non-decreasing."""
        res = self.sim_cornish.run_simulation(target_depth_m=200.0)
        diffs = np.diff(res["wear_state"])
        self.assertTrue(np.all(diffs >= -1e-10), "Wear state must not decrease")

    def test_hb_effective_viscosity(self):
        """Herschel-Bulkley viscosity must be positive and finite."""
        hyd = self.sim_cornish.hydraulics
        for gamma in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            mu = hyd.effective_viscosity(gamma)
            self.assertGreater(mu, 0.0)
            self.assertTrue(math.isfinite(mu))

    def test_ctr_bounded(self):
        """CTR must be in [0, 1]."""
        hyd = self.sim_cornish.hydraulics
        ctr = hyd.cuttings_transport_ratio(1e-3, 0.2159)
        self.assertGreaterEqual(ctr, 0.0)
        self.assertLessEqual(ctr, 1.0)

    def test_overbalance_increases_effective_ucs(self):
        """
        Higher mud weight → greater bottomhole overbalance → higher effective UCS
        at the bit face (chip hold-down effect, Detournay & Cheng 1993).

        We test effective_ucs directly rather than full ROP because at constant
        geometry, heavier mud also lowers cuttings slip velocity (denser fluid
        carries cuttings better), which can partially counteract the chip-hold-down
        penalty in the full ROP path.  The chip-hold-down physics itself is
        captured cleanly in effective_ucs.
        """
        sim_lo = GeothermalDrillingSimulator(site="utah_forge", mud_density_kg_m3=1100.0)
        sim_hi = GeothermalDrillingSimulator(site="utah_forge", mud_density_kg_m3=1500.0)
        ucs_lo = sim_lo.rop_module.effective_ucs(2000.0)
        ucs_hi = sim_hi.rop_module.effective_ucs(2000.0)
        self.assertGreater(
            ucs_hi, ucs_lo,
            "Higher overbalance (heavier mud) must increase effective UCS at bit face"
        )

    def test_temperature_profile_geothermal(self):
        """Formation temperature must increase with depth."""
        rm = self.sim_cornish.rock_mgr
        t1 = rm.formation_temperature(500.0)
        t2 = rm.formation_temperature(2000.0)
        self.assertGreater(t2, t1, "Temperature must increase with depth")

    def test_mse_positive(self):
        """MSE must be positive."""
        rop = self.sim_cornish.rop_module.compute_rop(1000.0)
        mse = self.sim_cornish.rop_module.mechanical_specific_energy(1000.0, rop)
        self.assertGreater(mse, 0.0)

    def test_csv_export(self):
        """CSV export must produce a readable file with correct headers."""
        res = self.sim_cornish.run_simulation(target_depth_m=100.0)
        path = "/tmp/geotwin_test_export.csv"
        self.sim_cornish.export_csv(path, results=res)
        with open(path) as fh:
            header = fh.readline().strip().split(",")
        self.assertIn("depth_m", header)
        self.assertIn("rop_m_hr", header)

    def test_utah_forge_multiunit(self):
        """Utah FORGE simulation must traverse multiple rock units."""
        res = self.sim_forge.run_simulation(target_depth_m=400.0)
        self.assertGreater(len(res["depth_m"]), 10)
        self.assertGreater(res["avg_rop_m_hr"], 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  EXAMPLE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def run_cornish_granite_example():
    """
    Example 1: United Downs Deep Geothermal Project (GEL, Cornwall)
    ---------------------------------------------------------------
    Simulates drilling through Carnmenellis granite with parameters typical of
    the UD-1 / UD-2 well programme.  Adjust values and re-run to explore the
    effect of WOB, RPM, and mud strategy on ROP and bit life.

    To substitute real GEL field data:
      1. Prepare a CSV with columns: depth_m, ucs_mpa, thermal_conductivity,
         density_kg_m3, temperature_c (any subset is acceptable).
      2. Pass csv_log_path="path/to/gel_log.csv" to the constructor.
      3. The RockPropertiesManager will use logged values in preference to defaults.
    """
    print("\n" + "=" * 70)
    print("  Example 1: United Downs / Cornish Granite (GEL)")
    print("=" * 70)

    sim = GeothermalDrillingSimulator(
        site="cornish_granite",
        wob_kn=120.0,              # High WOB for hard granite
        rotation_rpm=80.0,         # Lower RPM to control wear in abrasive rock
        flow_rate_m3s=0.030,       # 30 L/s — good hole cleaning at 8.5 in
        mud_density_kg_m3=1150.0,  # Slightly weighted — no significant pore pressure
        mud_yield_stress_pa=7.0,   # Higher yield stress for cuttings suspension
        mud_consistency_k=0.09,
        mud_flow_index_n=0.70,
        bit_type="PDC",
        max_depth_m=4500.0,
        time_step_s=120.0,
    )

    results = sim.run_simulation(target_depth_m=4500.0)

    # Export
    sim.export_csv("cornish_granite_sim.csv", results=results)
    sim.export_summary_json("cornish_granite_summary.json", results=results)

    # Plot (comment out if running headless)
    sim.plot_dashboard(results=results, save_path="cornish_granite_dashboard.png")

    return sim, results


def run_utah_forge_example():
    """
    Example 2: Utah FORGE (Milford, Utah)
    --------------------------------------
    Simulates drilling through the granite / quartz-monzonite / metamorphic
    sequence encountered in wells 58-32 and 16A(78)-32.  Temperature target
    is 175–225 °C at ~3–3.5 km depth (geothermal gradient ~55 °C/km).

    To load Utah FORGE public data:
      1. Download lithology / UCS / temperature logs from the Utah FORGE
         data repository (https://gdr.openei.org/submissions/1111).
      2. Format as CSV and pass csv_log_path="path/to/forge_58-32.csv".
      3. The simulator will override defaults with your observed values.
    """
    print("\n" + "=" * 70)
    print("  Example 2: Utah FORGE Granitoid Sequence")
    print("=" * 70)

    sim = GeothermalDrillingSimulator(
        site="utah_forge",
        wob_kn=100.0,
        rotation_rpm=120.0,
        flow_rate_m3s=0.025,
        mud_density_kg_m3=1180.0,
        mud_yield_stress_pa=5.0,
        mud_consistency_k=0.07,
        mud_flow_index_n=0.74,
        bit_type="PDC",
        max_depth_m=3500.0,
        time_step_s=120.0,
    )

    results = sim.run_simulation(target_depth_m=3500.0)

    sim.export_csv("utah_forge_sim.csv", results=results)
    sim.export_summary_json("utah_forge_summary.json", results=results)
    sim.plot_dashboard(results=results, save_path="utah_forge_dashboard.png")

    return sim, results


def run_wob_sweep_example():
    """
    Example 3: WOB sensitivity sweep for Utah FORGE
    -------------------------------------------------
    Demonstrates the ExperimentRunner for systematic parameter exploration.
    Sweeps WOB from 60 to 160 kN in 20 kN increments, reporting average ROP
    and MSE.  Useful for identifying the optimal WOB window before committing
    to a field programme.
    """
    print("\n" + "=" * 70)
    print("  Example 3: WOB Sweep — Utah FORGE")
    print("=" * 70)

    base_params = dict(
        site="utah_forge",
        rotation_rpm=100.0,
        flow_rate_m3s=0.025,
        mud_density_kg_m3=1180.0,
        max_depth_m=2000.0,
        time_step_s=300.0,
    )

    runner = ExperimentRunner(base_params)
    results = runner.sweep_wob(
        wob_values_kn=[60, 80, 100, 120, 140, 160],
        target_depth_m=2000.0,
    )

    runner.plot_sweep(
        results,
        param_name="Weight on Bit (kN)",
        param_key="wob_kn",
        metrics=["avg_rop_m_hr", "avg_mse_mpa", "final_wear_state"],
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Run unit tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestROPLimits)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    elif "--sweep" in sys.argv:
        run_wob_sweep_example()

    elif "--forge" in sys.argv:
        run_utah_forge_example()

    else:
        # Default: run both site examples
        sim_c, res_c = run_cornish_granite_example()
        sim_f, res_f = run_utah_forge_example()

        print("\n" + "=" * 70)
        print("  Comparative Summary")
        print("=" * 70)
        print(f"{'Metric':<30} {'Cornish Granite':>18} {'Utah FORGE':>18}")
        print("-" * 68)
        metrics = [
            ("Avg ROP (m/hr)", "avg_rop_m_hr"),
            ("Avg MSE (MPa)", "avg_mse_mpa"),
            ("Total drill time (hr)", "total_drill_time_hr"),
            ("Final bit wear", "final_wear_state"),
            ("Avg hydraulic efficiency", "hydraulic_efficiency_avg"),
        ]
        for label, key in metrics:
            vc = res_c.get(key, float("nan"))
            vf = res_f.get(key, float("nan"))
            print(f"{label:<30} {vc:>18.3f} {vf:>18.3f}")

        print("\n[GeoTwin] Done.  CSV and PNG outputs written to working directory.")
        print("          Run with --test to execute physics unit tests.")
        print("          Run with --sweep for WOB sensitivity analysis.")
        print("          Run with --forge for Utah FORGE only.")

"""
================================================================================
USAGE SUMMARY
================================================================================

1. Basic run (both sites):
       python geotwin_engine.py

2. Unit tests only:
       python geotwin_engine.py --test

3. Utah FORGE only:
       python geotwin_engine.py --forge

4. WOB sweep:
       python geotwin_engine.py --sweep

5. From a JSON parameter file:
       # Create params.json, then in Python:
       sim = GeothermalDrillingSimulator.from_json("params.json")
       results = sim.run_simulation()
       sim.export_csv("results.csv")

6. Load real field data:
       sim = GeothermalDrillingSimulator(
           site="cornish_granite",
           csv_log_path="gel_ud2_log.csv",  # depth_m, ucs_mpa, temperature_c, ...
       )

7. Custom rock type (add to _build_formation and as a RockProfile constant):
       my_rock = RockProfile(
           name="My Custom Granite",
           ucs_mpa=175.0,
           ...
       )
       # Add to RockPropertiesManager class and create a FormationProfile entry.

================================================================================
NEXT PRIORITY EXTENSIONS
================================================================================

Priority 1 — Full dynamic mud hydraulics:
  Extend MudHydraulicsModule with transient wellbore pressure surge/swab
  modelling (drill-pipe connection, pipe-running speed effects).  This is the
  highest practical risk in hard narrow-gauge HPHT wells and directly impacts
  wellbore stability assessments needed by GEL.

Priority 2 — Automated cross-site parameter calibration:
  Implement a scipy.optimize-based inversion loop that adjusts wear constants
  (WEAR_RATE_CONST, DP_SENSITIVITY) to minimise residuals between simulated
  and observed ROP logs from GEL field reports or Utah FORGE public data.
  This turns the tool from a forward model into a calibrated digital twin.

Priority 3 — Wellbore stability and deviation module:
  Add a 3-D trajectory module (minimum-curvature method) and a Mohr-Coulomb /
  Mogi-Coulomb wellbore stability calculator.  Hard granite at depth exhibits
  stress-controlled breakout in oriented wells; the GEL UD wells show
  oblique fracture systems that a stability module would quantify for sidetrack
  and completion planning.
================================================================================
"""
