#!/usr/bin/env python3
"""
Coarse-grained dissolution simulator for a single tablet formulation using OpenMM.

This script ignores any existing project code and builds a minimal, self-contained
coarse-grained (CG) molecular dynamics model that approximates dissolution behavior
for a metformin tablet with excipients (filler, binder, lubricant, others). Optionally,
explicit CG water beads can be included to model solvation, concentration gradients,
and wetting/penetration more realistically than with implicit solvent alone.

Core ideas (simplified for speed and robustness):
- Represent each component (metformin, filler, binder, lubricant, other) as CG beads.
- Beads interact via Lennard-Jones (LJ) potentials with type-specific epsilon/sigma.
- No permanent bonds are used; cohesion is via LJ attraction (binder has stronger
  cohesion with others). This allows the cluster to disperse under thermal motion.
- Default uses implicit solvent (no explicit water) with Langevin dynamics to approximate
  hydrodynamic/solvent effects while keeping runtime small. Optionally, enable explicit
  water to simulate solvation, diffusion, and wetting.
- Track dissolution as metformin beads that have left the tablet region and have no
  close contacts within a cutoff (i.e., solubilized). Output a CSV time series.

This is not a substitute for detailed all-atom dissolution modeling, but it provides
an extensible and fast baseline for formulation screening.

Usage:
  python formulation_dissolution_openmm.py \
    --config formulations/example_metformin_formulation.json \
    --output outputs/metformin_run

Outputs:
- <output>_metrics.csv: time_ps, num_dissolved, fraction_dissolved, tablet_radius_nm
  (tablet metrics computed using non-water beads; dissolution classification uses water
   neighbors if explicit water is present)
- <output>_final_positions.pdb: final bead positions for visualization (CG atoms)

Config JSON schema (minimal):
{
  "temperature_K": 310,
  "friction_per_ps": 1.0,
  "timestep_fs": 5,
  "simulation_time_ps": 2000,
  "box_nm": 12.0,
  "random_seed": 2025,
  "formulation": {
    "metformin": {"beads": 200},
    "excipients": [
      {"name": "microcrystalline_cellulose", "role": "filler", "beads": 200},
      {"name": "povidone", "role": "binder", "beads": 120},
      {"name": "magnesium_stearate", "role": "lubricant", "beads": 50}
    ],
    "water": {"beads": 30000}
  },
  "mixing_rules": {
    "water_metformin": {"epsilon": 3.0, "sigma": 0.39},
    "water_magnesium_stearate": {"epsilon": 0.3, "sigma": 0.47},
    "water_povidone": {"epsilon": 2.2, "sigma": 0.43}
  },
  "analysis": {
    "solvation_cutoff_nm": 0.6,
    "concentration_bins": 20,
    "output_frequency": 10
  },
  "dissolution_physics": {
    "enable_noyes_whitney": true,
    "rate_constant_nm_per_ps": 0.001,
    "saturation_concentration_beads_per_nm3": 0.1,
    "min_bead_mass_fraction": 0.05,
    "concentration_shell_nm": 1.0,
    "apply_to_components": ["metformin"]
  },
  "ph_effects": {
    "enable_ph_effects": true,
    "temperature_k": 310,
    "use_davies_equation": true,
    "gi_phases": [
      {"name": "gastric", "ph": 1.5, "ionic_strength_m": 0.1, "duration_ps": 3600000, "buffer_capacity": 0.01},
      {"name": "duodenal", "ph": 6.0, "ionic_strength_m": 0.15, "duration_ps": 1800000, "buffer_capacity": 0.02},
      {"name": "intestinal", "ph": 7.4, "ionic_strength_m": 0.12, "duration_ps": 10800000, "buffer_capacity": 0.015}
    ],
    "species_data": {
      "metformin": {
        "pka_values": [2.8, 11.5],
        "base_charge": 0,
        "solubility_factors": [1.0, 15.0, 100.0],
        "rate_factors": [0.5, 2.0, 8.0]
      }
    }
  }
}

You can override per-ingredient parameters in the config via an optional
"parameters" map under each component (epsilon_kj_mol, sigma_nm, mass_amu).
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import openmm
from openmm import unit
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # fallback if tqdm is unavailable


# ------------------------------ Data Structures ------------------------------


@dataclass
class BeadType:
    name: str
    epsilon_kj_mol: float
    sigma_nm: float
    mass_amu: float


@dataclass
class ComponentSpec:
    label: str
    bead_type: BeadType
    count: int


@dataclass
class DissolvingBead:
    """Tracks dynamic properties of a dissolving bead."""
    component_label: str
    original_mass_amu: float
    current_mass_amu: float
    original_sigma_nm: float
    current_sigma_nm: float
    current_epsilon_kj_mol: float
    is_dissolved: bool = False
    cumulative_dissolved_mass_amu: float = 0.0
    
    @property
    def mass_fraction_remaining(self) -> float:
        """Fraction of original mass remaining."""
        return self.current_mass_amu / self.original_mass_amu if self.original_mass_amu > 0 else 0.0
    
    @property
    def surface_area_nm2(self) -> float:
        """Current surface area assuming spherical bead."""
        radius_nm = self.current_sigma_nm / 2.0
        return 4.0 * np.pi * radius_nm * radius_nm


@dataclass
class DissolutionParams:
    """Noyes-Whitney dissolution parameters."""
    rate_constant_nm_per_ps: float  # k in Noyes-Whitney equation
    saturation_concentration_beads_per_nm3: float  # Cs - max solubility
    min_bead_mass_fraction: float  # Remove bead when mass falls below this fraction
    concentration_shell_nm: float  # Radius to calculate local concentration
    apply_to_components: List[str]  # Which components can dissolve (e.g., ["metformin"])


@dataclass
class IonicSpecies:
    """Properties of an ionic species at different pH values."""
    name: str
    pka_values: List[float]  # pKa values for polyprotic species
    base_charge: int  # Charge at fully deprotonated state
    solubility_factors: List[float]  # Solubility multiplier for each ionization state
    rate_factors: List[float]  # Rate constant multiplier for each ionization state


@dataclass
class GIPhase:
    """Gastrointestinal tract phase parameters."""
    name: str
    ph: float
    ionic_strength_m: float  # Ionic strength in molarity
    duration_ps: float  # Duration of this phase
    buffer_capacity: float  # Resistance to pH change


@dataclass
class PHParams:
    """pH and ionic strength parameters for dissolution."""
    enable_ph_effects: bool
    gi_phases: List[GIPhase]  # Sequential GI phases (stomach → intestine)
    species_data: Dict[str, IonicSpecies]  # Component-specific ionization data
    temperature_k: float  # For activity coefficient calculations
    use_davies_equation: bool  # Use Davies equation for activity coefficients


# ------------------------------ Defaults / Library ---------------------------


def get_default_bead_library() -> Dict[str, BeadType]:
    """Return a default library of CG bead parameters for ingredients.

    Epsilon/sigma are loosely tuned to encode relative cohesion/hydrophilicity.
    Values are in kJ/mol and nm. Mass is in atomic mass units (Daltons).
    """
    return {
        # Metformin: small, hydrophilic API
        "metformin": BeadType(name="metformin", epsilon_kj_mol=1.10, sigma_nm=0.47, mass_amu=129.0),
        # Water (CG one-bead): small sigma, stronger epsilon for realistic bulk behavior
        "water": BeadType(name="water", epsilon_kj_mol=2.0, sigma_nm=0.32, mass_amu=18.0),
        # Filler examples
        "microcrystalline_cellulose": BeadType(
            name="microcrystalline_cellulose", epsilon_kj_mol=0.85, sigma_nm=0.52, mass_amu=162.0
        ),
        "lactose_monohydrate": BeadType(
            name="lactose_monohydrate", epsilon_kj_mol=0.90, sigma_nm=0.50, mass_amu=198.0
        ),
        # Binder examples (stronger cohesion)
        "povidone": BeadType(name="povidone", epsilon_kj_mol=1.55, sigma_nm=0.55, mass_amu=111.0),
        "hypromellose": BeadType(name="hypromellose", epsilon_kj_mol=1.45, sigma_nm=0.56, mass_amu=126.0),
        # Lubricant examples (hydrophobic, weaker cohesion with hydrophilic)
        "magnesium_stearate": BeadType(
            name="magnesium_stearate", epsilon_kj_mol=0.65, sigma_nm=0.62, mass_amu=591.0
        ),
        # Disintegrant example (promotes breakup slightly lower cohesion with binder)
        "croscarmellose_sodium": BeadType(
            name="croscarmellose_sodium", epsilon_kj_mol=0.95, sigma_nm=0.58, mass_amu=241.0
        ),
        # Generic fallback
        "generic": BeadType(name="generic", epsilon_kj_mol=1.00, sigma_nm=0.50, mass_amu=100.0),
    }


def resolve_bead_type(name: str, overrides: Dict[str, float]) -> BeadType:
    lib = get_default_bead_library()
    base = lib.get(name, lib["generic"])
    return BeadType(
        name=name,
        epsilon_kj_mol=float(overrides.get("epsilon_kj_mol", base.epsilon_kj_mol)),
        sigma_nm=float(overrides.get("sigma_nm", base.sigma_nm)),
        mass_amu=float(overrides.get("mass_amu", base.mass_amu)),
    )


# ------------------------------ Geometry Helpers -----------------------------


def generate_cluster_positions(
    counts: Dict[str, int],
    spacing_nm: float,
    box_size_nm: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Generate initial positions for each component as a compact cluster centered in the box.

    Places all beads on a cubic lattice within a sphere. Returns positions per label.
    """
    rng = np.random.default_rng(seed)
    total = sum(counts.values())
    # Estimate the number of lattice sites needed (cube of side sites_per_axis)
    sites_per_axis = math.ceil((total) ** (1.0 / 3.0))
    # Build lattice
    coords = []
    for i in range(sites_per_axis):
        for j in range(sites_per_axis):
            for k in range(sites_per_axis):
                coords.append((i, j, k))
    coords = np.array(coords[:total], dtype=float)

    # Center lattice and apply small random jitter to avoid perfect symmetry
    coords -= np.mean(coords, axis=0)
    coords *= spacing_nm
    jitter = rng.normal(0.0, 0.05 * spacing_nm, size=coords.shape)
    coords += jitter

    # Shift to center of box
    center = 0.5 * box_size_nm
    coords += center

    # Partition positions by label in provided order
    positions_by_label: Dict[str, np.ndarray] = {}
    start = 0
    for label, count in counts.items():
        positions_by_label[label] = coords[start : start + count]
        start += count
    return positions_by_label


def compute_centroid(positions_nm: np.ndarray) -> np.ndarray:
    return np.mean(positions_nm, axis=0)


def compute_tablet_radius_nm(positions_nm: np.ndarray, centroid_nm: np.ndarray) -> float:
    # Use RMS radius as characteristic radius
    disp = positions_nm - centroid_nm
    rms = np.sqrt(np.mean(np.sum(disp * disp, axis=1)))
    return float(rms)


def count_dissolved_metformin(
    met_positions_nm: np.ndarray,
    all_positions_nm: np.ndarray,
    centroid_nm: np.ndarray,
    tablet_radius_nm: float,
    radial_margin_nm: float,
    contact_cutoff_nm: float,
) -> int:
    """Count metformin beads considered dissolved.

    A bead is dissolved if:
    - Its distance from the tablet centroid exceeds (tablet_radius + radial_margin)
    - AND it has no neighbors within contact_cutoff_nm (loosely, not aggregated)
    """
    if met_positions_nm.size == 0:
        return 0
    disp = met_positions_nm - centroid_nm
    r = np.sqrt(np.sum(disp * disp, axis=1))
    radial_mask = r > (tablet_radius_nm + radial_margin_nm)
    if not np.any(radial_mask):
        return 0
    # Brute-force neighbor check (N small for CG). For speed, use a simple grid if needed.
    candidates = met_positions_nm[radial_mask]
    count = 0
    cutoff2 = contact_cutoff_nm * contact_cutoff_nm
    for pos in candidates:
        d2 = np.sum((all_positions_nm - pos) ** 2, axis=1)
        # Subtract self if it is included in all_positions (safe thresholding)
        close = np.sum(d2 < cutoff2) - 1  # exclude self
        if close <= 0:
            count += 1
    return count


def count_dissolved_metformin_explicit(
    met_positions_nm: np.ndarray,
    all_positions_nm: np.ndarray,
    nonwater_positions_nm: np.ndarray,
    water_positions_nm: np.ndarray,
    centroid_nm: np.ndarray,
    tablet_radius_nm: float,
    radial_margin_nm: float,
    contact_cutoff_nm: float,
    water_shell_nm: float,
    min_water_neighbors: int = 1,
) -> int:
    """Count metformin beads considered dissolved when explicit water is present.

    Criteria:
    - Radially outside tablet: r > tablet_radius + radial_margin
    - Has >= min_water_neighbors within water_shell_nm
    - Has no non-water neighbors within contact_cutoff_nm
    """
    if met_positions_nm.size == 0:
        return 0
    disp = met_positions_nm - centroid_nm
    r = np.sqrt(np.sum(disp * disp, axis=1))
    radial_mask = r > (tablet_radius_nm + radial_margin_nm)
    if not np.any(radial_mask):
        return 0
    candidates = met_positions_nm[radial_mask]
    count = 0
    contact2 = contact_cutoff_nm * contact_cutoff_nm
    shell2 = water_shell_nm * water_shell_nm
    for pos in candidates:
        # Non-water close contacts (including metformin & excipients)
        if nonwater_positions_nm.size:
            d2_nonwater = np.sum((nonwater_positions_nm - pos) ** 2, axis=1)
            # Subtract self later by requiring > 0 neighbors strictly
            close_nonwater = np.sum(d2_nonwater < contact2) - 1  # exclude self if present
            if close_nonwater > 0:
                continue
        # Water neighbors in solvation shell
        if water_positions_nm.size:
            d2_water = np.sum((water_positions_nm - pos) ** 2, axis=1)
            n_water = int(np.sum(d2_water < shell2))
            if n_water >= min_water_neighbors:
                count += 1
        else:
            # No water present; cannot consider dissolved under explicit criterion
            pass
    return count


def generate_water_positions_excluding_sphere(
    count: int,
    box_size_nm: float,
    exclude_center_nm: np.ndarray,
    exclude_radius_nm: float,
    seed: int,
) -> np.ndarray:
    """Generate uniformly random water positions in the box excluding a spherical region.

    Rejection sampling in batches for simplicity.
    """
    if count <= 0:
        return np.zeros((0, 3), dtype=float)
    rng = np.random.default_rng(seed + 1337)
    positions: List[np.ndarray] = []
    batch = max(1024, min(65536, count * 2))
    excl_center = exclude_center_nm.reshape(1, 3)
    excl_r2 = exclude_radius_nm * exclude_radius_nm
    accepted = 0
    while accepted < count:
        candidates = rng.random((batch, 3)) * box_size_nm
        d2 = np.sum((candidates - excl_center) ** 2, axis=1)
        keep_mask = d2 > excl_r2
        kept = candidates[keep_mask]
        if kept.size:
            positions.append(kept)
            accepted += kept.shape[0]
    positions_nm = np.vstack(positions)[:count]
    return positions_nm


# ------------------------------ Analysis Functions ---------------------------


def analyze_solvation_shells(
    met_positions_nm: np.ndarray,
    water_positions_nm: np.ndarray,
    solvation_cutoff_nm: float = 0.6,
) -> Dict[str, float]:
    """Analyze solvation shells around metformin beads.
    
    Returns:
        Dict with statistics: mean_solvation_number, std_solvation_number,
                            min_solvation_number, max_solvation_number
    """
    if met_positions_nm.size == 0 or water_positions_nm.size == 0:
        return {
            "mean_solvation_number": 0.0,
            "std_solvation_number": 0.0,
            "min_solvation_number": 0.0,
            "max_solvation_number": 0.0,
        }
    
    solvation_numbers = []
    cutoff2 = solvation_cutoff_nm * solvation_cutoff_nm
    
    for met_pos in met_positions_nm:
        # Calculate distances to all water molecules
        d2 = np.sum((water_positions_nm - met_pos) ** 2, axis=1)
        n_solvating = int(np.sum(d2 < cutoff2))
        solvation_numbers.append(n_solvating)
    
    solvation_numbers = np.array(solvation_numbers)
    
    return {
        "mean_solvation_number": float(np.mean(solvation_numbers)),
        "std_solvation_number": float(np.std(solvation_numbers)),
        "min_solvation_number": float(np.min(solvation_numbers)),
        "max_solvation_number": float(np.max(solvation_numbers)),
    }


def calculate_radial_concentration(
    met_positions_nm: np.ndarray,
    centroid_nm: np.ndarray,
    box_size_nm: float,
    bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate radial concentration profile of metformin from tablet center.
    
    Returns:
        bin_centers_nm: Radial distances (bin centers)
        concentrations: Metformin concentration in each bin (beads/nm³)
    """
    if met_positions_nm.size == 0:
        bin_centers = np.linspace(0, box_size_nm/2, bins)
        return bin_centers, np.zeros_like(bin_centers)
    
    # Calculate distances from centroid
    disp = met_positions_nm - centroid_nm
    r = np.sqrt(np.sum(disp * disp, axis=1))
    
    # Create radial bins
    max_radius = min(box_size_nm / 2, np.max(r) * 1.1)
    bin_edges = np.linspace(0, max_radius, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Count metformin beads in each bin
    counts, _ = np.histogram(r, bins=bin_edges)
    
    # Calculate bin volumes (spherical shells)
    bin_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    # Calculate concentrations (beads per nm³)
    concentrations = counts / bin_volumes
    
    return bin_centers, concentrations


def track_water_infiltration(
    water_positions_nm: np.ndarray,
    tablet_centroid_nm: np.ndarray,
    tablet_radius_nm: float,
) -> Dict[str, float]:
    """Track water infiltration into the tablet region.
    
    Returns:
        Dict with metrics: water_inside_tablet, max_penetration_depth_nm,
                         fraction_infiltrated
    """
    if water_positions_nm.size == 0:
        return {
            "water_inside_tablet": 0.0,
            "max_penetration_depth_nm": 0.0,
            "fraction_infiltrated": 0.0,
        }
    
    # Calculate distances from tablet center
    disp = water_positions_nm - tablet_centroid_nm
    r = np.sqrt(np.sum(disp * disp, axis=1))
    
    # Count water inside tablet region
    inside_mask = r <= tablet_radius_nm
    water_inside = int(np.sum(inside_mask))
    
    # Calculate maximum penetration depth
    max_penetration = tablet_radius_nm - np.min(r) if r.size > 0 else 0.0
    max_penetration = max(0.0, max_penetration)  # Can't be negative
    
    # Fraction of water that has infiltrated
    fraction_infiltrated = water_inside / len(water_positions_nm)
    
    return {
        "water_inside_tablet": float(water_inside),
        "max_penetration_depth_nm": float(max_penetration),
        "fraction_infiltrated": float(fraction_infiltrated),
    }


def calculate_local_concentration(
    bead_position_nm: np.ndarray,
    dissolved_positions_nm: np.ndarray,
    shell_radius_nm: float,
) -> float:
    """Calculate local concentration of dissolved material around a bead.
    
    Args:
        bead_position_nm: Position of the bead
        dissolved_positions_nm: Positions of dissolved/solubilized beads
        shell_radius_nm: Radius of concentration calculation shell
    
    Returns:
        Local concentration in beads/nm³
    """
    if dissolved_positions_nm.size == 0:
        return 0.0
    
    # Calculate distances to dissolved beads
    distances = np.sqrt(np.sum((dissolved_positions_nm - bead_position_nm) ** 2, axis=1))
    
    # Count dissolved beads within shell
    dissolved_in_shell = int(np.sum(distances <= shell_radius_nm))
    
    # Shell volume
    shell_volume_nm3 = (4.0/3.0) * np.pi * shell_radius_nm**3
    
    return dissolved_in_shell / shell_volume_nm3


def get_current_gi_phase(time_ps: float, gi_phases: List[GIPhase]) -> Tuple[GIPhase, float]:
    """Get current GI phase and progress within phase based on simulation time.
    
    Args:
        time_ps: Current simulation time
        gi_phases: List of sequential GI phases
    
    Returns:
        Tuple of (current_phase, phase_progress_fraction)
    """
    if not gi_phases:
        # Default gastric phase if none specified
        default_phase = GIPhase(name="gastric", ph=1.5, ionic_strength_m=0.1, 
                               duration_ps=float('inf'), buffer_capacity=0.01)
        return default_phase, 0.0
    
    cumulative_time = 0.0
    for phase in gi_phases:
        if time_ps <= cumulative_time + phase.duration_ps:
            progress = (time_ps - cumulative_time) / phase.duration_ps
            return phase, min(1.0, max(0.0, progress))
        cumulative_time += phase.duration_ps
    
    # Return last phase if time exceeds all phases
    return gi_phases[-1], 1.0


def calculate_ionization_fractions(ph: float, pka_values: List[float], base_charge: int) -> List[float]:
    """Calculate ionization state fractions using Henderson-Hasselbalch equation.
    
    For polyprotic species with multiple pKa values.
    
    Args:
        ph: Current pH
        pka_values: List of pKa values (sorted from lowest to highest)
        base_charge: Charge at fully deprotonated state
        
    Returns:
        List of fractions for each ionization state [fully protonated → fully deprotonated]
    """
    if not pka_values:
        # Neutral species
        return [1.0]
    
    n_states = len(pka_values) + 1  # Number of ionization states
    alpha = np.zeros(n_states)
    
    # Calculate alpha values using Henderson-Hasselbalch
    h_concentration = 10**(-ph)
    
    # Calculate denominators for each state
    denominator = 0.0
    for i in range(n_states):
        term = 1.0
        for j in range(i):
            term *= (h_concentration / (10**(-pka_values[j])))
        denominator += term
    
    # Calculate fractions
    for i in range(n_states):
        numerator = 1.0
        for j in range(i):
            numerator *= (h_concentration / (10**(-pka_values[j])))
        alpha[i] = numerator / denominator
    
    return alpha.tolist()


def calculate_average_charge(ionization_fractions: List[float], base_charge: int, pka_values: List[float]) -> float:
    """Calculate average charge based on ionization state fractions.
    
    Args:
        ionization_fractions: Fractions for each ionization state
        base_charge: Charge at fully deprotonated state
        pka_values: List of pKa values
        
    Returns:
        Average charge of the species
    """
    if not ionization_fractions:
        return 0.0
    
    avg_charge = 0.0
    for i, fraction in enumerate(ionization_fractions):
        # Each protonation increases charge by +1
        protons_added = len(pka_values) - i
        charge = base_charge + protons_added
        avg_charge += fraction * charge
    
    return avg_charge


def calculate_davies_activity_coefficient(charge: float, ionic_strength: float) -> float:
    """Calculate activity coefficient using Davies equation.
    
    γ = 10^(-A|z²|[(√I)/(1+√I) - 0.3I])
    where A = 0.5085 at 25°C in water
    
    Args:
        charge: Ionic charge
        ionic_strength: Ionic strength in molarity
        
    Returns:
        Activity coefficient
    """
    if abs(charge) < 0.1:  # Neutral species
        return 1.0
    
    A = 0.5085  # Davies constant at 25°C
    sqrt_I = np.sqrt(ionic_strength)
    
    exponent = -A * (charge**2) * (sqrt_I / (1 + sqrt_I) - 0.3 * ionic_strength)
    return 10**exponent


def get_ph_dependent_dissolution_params(
    base_params: DissolutionParams,
    species_name: str,
    ph: float,
    ionic_strength: float,
    species_data: Dict[str, IonicSpecies],
) -> DissolutionParams:
    """Calculate pH-dependent dissolution parameters.
    
    Args:
        base_params: Base dissolution parameters
        species_name: Name of the dissolving species
        ph: Current pH
        ionic_strength: Current ionic strength
        species_data: Ionic species database
        
    Returns:
        Modified dissolution parameters for current conditions
    """
    if species_name not in species_data:
        # No pH data available, return base parameters
        return base_params
    
    species = species_data[species_name]
    
    # Calculate ionization fractions
    fractions = calculate_ionization_fractions(ph, species.pka_values, species.base_charge)
    
    # Calculate weighted solubility and rate factors
    solubility_factor = sum(f * sf for f, sf in zip(fractions, species.solubility_factors))
    rate_factor = sum(f * rf for f, rf in zip(fractions, species.rate_factors))
    
    # Calculate average charge for activity coefficient
    avg_charge = calculate_average_charge(fractions, species.base_charge, species.pka_values)
    activity_coeff = calculate_davies_activity_coefficient(avg_charge, ionic_strength)
    
    # Apply pH-dependent modifications
    modified_params = DissolutionParams(
        rate_constant_nm_per_ps=base_params.rate_constant_nm_per_ps * rate_factor * activity_coeff,
        saturation_concentration_beads_per_nm3=base_params.saturation_concentration_beads_per_nm3 * solubility_factor * activity_coeff,
        min_bead_mass_fraction=base_params.min_bead_mass_fraction,
        concentration_shell_nm=base_params.concentration_shell_nm,
        apply_to_components=base_params.apply_to_components,
    )
    
    return modified_params


def apply_noyes_whitney_dissolution(
    dissolving_beads: List[DissolvingBead],
    bead_positions_nm: np.ndarray,
    dissolved_positions_nm: np.ndarray,
    dissolution_params: DissolutionParams,
    dt_ps: float,
) -> Tuple[List[DissolvingBead], List[int]]:
    """Apply Noyes-Whitney dissolution kinetics to beads.
    
    The Noyes-Whitney equation: dM/dt = k·A·(Cs - C)
    
    Args:
        dissolving_beads: List of bead objects to update
        bead_positions_nm: Current positions of all dissolving beads
        dissolved_positions_nm: Positions of fully dissolved beads for concentration calc
        dissolution_params: Dissolution physics parameters
        dt_ps: Time step in picoseconds
    
    Returns:
        Tuple of (updated_beads, indices_to_remove)
    """
    updated_beads = []
    indices_to_remove = []
    
    for i, bead in enumerate(dissolving_beads):
        if bead.is_dissolved:
            updated_beads.append(bead)
            continue
        
        # Check if this component should dissolve
        should_dissolve = any(comp in bead.component_label for comp in dissolution_params.apply_to_components)
        if not should_dissolve:
            updated_beads.append(bead)
            continue
            
        # Calculate local concentration around this bead
        local_concentration = calculate_local_concentration(
            bead_positions_nm[i],
            dissolved_positions_nm,
            dissolution_params.concentration_shell_nm
        )
        
        # Concentration driving force (Cs - C)
        concentration_gradient = max(0.0, 
            dissolution_params.saturation_concentration_beads_per_nm3 - local_concentration)
        
        if concentration_gradient <= 0.0:
            # No driving force - saturated local environment
            updated_beads.append(bead)
            continue
        
        # Apply Noyes-Whitney equation: dM/dt = k·A·(Cs - C)
        dissolution_rate_amu_per_ps = (
            dissolution_params.rate_constant_nm_per_ps * 
            bead.surface_area_nm2 * 
            concentration_gradient
        )
        
        # Calculate mass loss over time step
        mass_loss_amu = dissolution_rate_amu_per_ps * dt_ps
        mass_loss_amu = min(mass_loss_amu, bead.current_mass_amu)  # Can't lose more than current mass
        
        # Update bead properties
        new_bead = DissolvingBead(
            component_label=bead.component_label,
            original_mass_amu=bead.original_mass_amu,
            current_mass_amu=bead.current_mass_amu - mass_loss_amu,
            original_sigma_nm=bead.original_sigma_nm,
            current_sigma_nm=bead.current_sigma_nm,  # Will update based on mass
            current_epsilon_kj_mol=bead.current_epsilon_kj_mol,
            is_dissolved=bead.is_dissolved,
            cumulative_dissolved_mass_amu=bead.cumulative_dissolved_mass_amu + mass_loss_amu
        )
        
        # Update size based on mass (assuming constant density)
        # sigma scales as cube root of mass for spherical beads
        mass_ratio = new_bead.current_mass_amu / new_bead.original_mass_amu
        if mass_ratio > 0:
            new_bead.current_sigma_nm = bead.original_sigma_nm * (mass_ratio ** (1.0/3.0))
        else:
            new_bead.current_sigma_nm = 0.0
        
        # Check if bead should be considered fully dissolved
        if new_bead.mass_fraction_remaining <= dissolution_params.min_bead_mass_fraction:
            new_bead.is_dissolved = True
            new_bead.cumulative_dissolved_mass_amu = new_bead.original_mass_amu
            indices_to_remove.append(i)
        
        updated_beads.append(new_bead)
    
    return updated_beads, indices_to_remove


def update_nonbonded_parameters(
    nb_force: openmm.NonbondedForce,
    dissolving_beads: List[DissolvingBead],
    context: openmm.Context,
) -> None:
    """Update OpenMM NonbondedForce parameters for dissolving beads.
    
    Args:
        nb_force: OpenMM NonbondedForce to modify
        dissolving_beads: Current bead states
        context: OpenMM context to reinitialize after parameter updates
    """
    for i, bead in enumerate(dissolving_beads):
        if bead.is_dissolved:
            # Set dissolved beads to minimal interaction
            nb_force.setParticleParameters(
                i,
                0.0 * unit.elementary_charge,  # charge
                0.01 * unit.nanometer,  # very small sigma
                0.001 * unit.kilojoule_per_mole,  # minimal epsilon
            )
        else:
            # Update with current dissolution state
            nb_force.setParticleParameters(
                i,
                0.0 * unit.elementary_charge,  # charge
                bead.current_sigma_nm * unit.nanometer,  # current size
                bead.current_epsilon_kj_mol * unit.kilojoule_per_mole,  # current interaction
            )
    
    # Reinitialize context to apply parameter changes
    context.reinitialize(preserveState=True)


# ------------------------------ System Construction --------------------------


def build_system(
    components: List[ComponentSpec],
    cutoff_nm: float,
    use_pbc: bool,
    box_size_nm: float,
) -> Tuple[openmm.System, openmm.NonbondedForce]:
    system = openmm.System()

    # Create a single NonbondedForce using LJ only (charges=0). Use Lorentz-Berthelot combining.
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(
        openmm.NonbondedForce.CutoffPeriodic if use_pbc else openmm.NonbondedForce.CutoffNonPeriodic
    )
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    nb.setUseDispersionCorrection(True)

    # Particle bookkeeping
    particle_types: List[BeadType] = []
    for comp in components:
        particle_types.extend([comp.bead_type] * comp.count)

    for bead in particle_types:
        mass = bead.mass_amu * unit.amu
        idx = system.addParticle(mass)
        # Charge 0, LJ sigma/epsilon
        nb.addParticle(
            0.0 * unit.elementary_charge,
            bead.sigma_nm * unit.nanometer,
            bead.epsilon_kj_mol * unit.kilojoule_per_mole,
        )

    if use_pbc:
        a = openmm.Vec3(box_size_nm, 0.0, 0.0) * unit.nanometer
        b = openmm.Vec3(0.0, box_size_nm, 0.0) * unit.nanometer
        c = openmm.Vec3(0.0, 0.0, box_size_nm) * unit.nanometer
        system.setDefaultPeriodicBoxVectors(a, b, c)

    system.addForce(nb)
    return system, nb


def apply_custom_mixing_rules(
    nb_force: openmm.NonbondedForce,
    components: List[ComponentSpec],
    mixing_rules: Dict[str, Dict[str, float]] = None,
) -> None:
    """Apply custom mixing rules for specific component interactions using addException().
    
    This function overrides default Lorentz-Berthelot combining rules with realistic
    interaction parameters, particularly for water-component pairs.
    
    Args:
        nb_force: The NonbondedForce object to modify
        components: List of component specifications in particle order
        mixing_rules: Optional dict of custom interaction parameters
                     Format: {"water_metformin": {"epsilon": 3.0, "sigma": 0.39}}
    """
    if mixing_rules is None:
        # Default enhanced mixing rules for realistic solvation behavior
        mixing_rules = {
            "water_metformin": {"epsilon": 3.0, "sigma": 0.39},  # Strong hydrophilic interaction
            "water_magnesium_stearate": {"epsilon": 0.3, "sigma": 0.47},  # Weak hydrophobic interaction
            "water_povidone": {"epsilon": 2.2, "sigma": 0.43},  # Moderate binder interaction
            "water_hypromellose": {"epsilon": 2.0, "sigma": 0.44},  # Moderate binder interaction
        }
    
    # Build particle index mapping by component type
    particle_indices: Dict[str, List[int]] = {}
    start_idx = 0
    for comp in components:
        # Extract base component name (remove role prefix if present)
        comp_name = comp.label.split(":")[-1] if ":" in comp.label else comp.label
        if comp_name not in particle_indices:
            particle_indices[comp_name] = []
        particle_indices[comp_name].extend(range(start_idx, start_idx + comp.count))
        start_idx += comp.count
    
    # Apply custom mixing rules via exceptions
    for rule_key, params in mixing_rules.items():
        parts = rule_key.split("_", 1)
        if len(parts) != 2:
            continue
        comp1_name, comp2_name = parts
        
        if comp1_name not in particle_indices or comp2_name not in particle_indices:
            continue
            
        indices1 = particle_indices[comp1_name]
        indices2 = particle_indices[comp2_name]
        
        epsilon = params.get("epsilon", 1.0) * unit.kilojoule_per_mole
        sigma = params.get("sigma", 0.5) * unit.nanometer
        
        # Add exceptions for all pairs between these component types
        for i1 in indices1:
            for i2 in indices2:
                if i1 != i2:  # Don't add self-interactions
                    # addException(i, j, chargeProd, sigma, epsilon)
                    nb_force.addException(
                        i1, i2, 
                        0.0 * unit.elementary_charge**2,  # No charge interactions
                        sigma,
                        epsilon
                    )
        
        print(f"Applied custom mixing rule {rule_key}: ε={params['epsilon']:.2f} kJ/mol, σ={params['sigma']:.3f} nm")
        print(f"  Affected {len(indices1)} × {len(indices2)} particle pairs")


# ------------------------------ PDB Export (CG) -------------------------------


def write_cg_pdb(path: str, positions_nm: np.ndarray, labels: List[str], box_size_nm: float) -> None:
    """Write a minimal PDB for visualization. One atom per bead, element C.

    PDB here is generated manually for simplicity.
    """
    with open(path, "w") as fh:
        # CRYST1 line for box
        fh.write(
            f"CRYST1 {box_size_nm*10:9.3f} {box_size_nm*10:9.3f} {box_size_nm*10:9.3f}  90.00  90.00  90.00 P 1           1\n"
        )
        atom_index = 1
        residue_index = 1
        for pos, label in zip(positions_nm, labels):
            x, y, z = (pos * 10.0)  # nm -> Angstrom
            atom_name = "C"
            res_name = label[:3].upper()
            fh.write(
                f"ATOM  {atom_index:5d} {atom_name:>4s} {res_name:>3s} A{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_index += 1
            if atom_index % 9999 == 0:
                residue_index += 1
        fh.write("END\n")


# ------------------------------ Simulation Runner ----------------------------


def run_simulation(config_path: str, output_prefix: str, override_steps: int = None) -> None:
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    temperature_K = float(cfg.get("temperature_K", 310.0))
    friction_per_ps = float(cfg.get("friction_per_ps", 1.0))
    timestep_fs = float(cfg.get("timestep_fs", 5.0))
    simulation_time_ps = float(cfg.get("simulation_time_ps", 2000.0))
    box_nm = float(cfg.get("box_nm", 12.0))
    random_seed = int(cfg.get("random_seed", 2025))

    steps = (
        int(override_steps)
        if override_steps is not None
        else int(round(simulation_time_ps * 1000.0 / timestep_fs))
    )

    # Map components
    formulation = cfg.get("formulation", {})
    if not formulation:
        raise ValueError("Config missing 'formulation' block.")

    # Build components, separating tablet (non-water) and water
    tablet_components: List[ComponentSpec] = []
    water_component: ComponentSpec = None

    # Metformin
    met_spec = formulation.get("metformin", {})
    met_beads = int(met_spec.get("beads", met_spec.get("molecules", 200)))
    met_params = met_spec.get("parameters", {})
    met_type = resolve_bead_type("metformin", met_params)
    tablet_components.append(ComponentSpec(label="metformin", bead_type=met_type, count=met_beads))

    # Excipients
    for item in formulation.get("excipients", []):
        name = str(item.get("name", "generic"))
        role = str(item.get("role", "other"))
        label = f"{role}:{name}"
        count = int(item.get("beads", 100))
        params = item.get("parameters", {})
        bead_type = resolve_bead_type(name, params)
        tablet_components.append(ComponentSpec(label=label, bead_type=bead_type, count=count))

    # Optional explicit water
    water_spec = formulation.get("water", {})
    water_beads = int(water_spec.get("beads", 0))
    if water_beads > 0:
        water_params = water_spec.get("parameters", {})
        water_type = resolve_bead_type("water", water_params)
        water_component = ComponentSpec(label="water", bead_type=water_type, count=water_beads)

    # Create initial positions for tablet as a compact cluster
    counts_tablet = {comp.label: comp.count for comp in tablet_components}
    spacing_nm = 0.60  # lattice spacing roughly corresponding to sigma scale
    positions_by_label_tablet = generate_cluster_positions(counts_tablet, spacing_nm, box_nm, random_seed)
    labels_tablet: List[str] = []
    positions_list_tablet: List[np.ndarray] = []
    for comp in tablet_components:
        labels_tablet.extend([comp.label] * comp.count)
        positions_list_tablet.append(positions_by_label_tablet[comp.label])
    positions_tablet_nm = np.vstack(positions_list_tablet) if positions_list_tablet else np.zeros((0, 3))

    # Estimate tablet centroid/radius from tablet beads only
    centroid_tablet = compute_centroid(positions_tablet_nm) if positions_tablet_nm.size else np.array([
        0.5 * box_nm, 0.5 * box_nm, 0.5 * box_nm
    ])
    tablet_radius_est = compute_tablet_radius_nm(positions_tablet_nm, centroid_tablet) if positions_tablet_nm.size else 0.5

    # Generate water positions outside tablet region
    labels: List[str] = list(labels_tablet)
    positions_list: List[np.ndarray] = [positions_tablet_nm]
    if water_component is not None:
        water_exclusion_margin_nm = 0.8
        positions_water_nm = generate_water_positions_excluding_sphere(
            water_component.count,
            box_size_nm=box_nm,
            exclude_center_nm=centroid_tablet,
            exclude_radius_nm=tablet_radius_est + water_exclusion_margin_nm,
            seed=random_seed,
        )
        labels.extend(["water"] * water_component.count)
        positions_list.append(positions_water_nm)

    positions_nm = np.vstack(positions_list) if positions_list else np.zeros((0, 3))

    # Build system using all components
    components_all: List[ComponentSpec] = list(tablet_components)
    if water_component is not None:
        components_all.append(water_component)
    cutoff_nm = 1.2
    system, nb = build_system(components_all, cutoff_nm=cutoff_nm, use_pbc=True, box_size_nm=box_nm)
    
    # Apply custom mixing rules for realistic solvation behavior
    mixing_rules_config = cfg.get("mixing_rules", {})
    apply_custom_mixing_rules(nb, components_all, mixing_rules_config if mixing_rules_config else None)

    # Integrator & context
    integrator = openmm.LangevinIntegrator(
        temperature_K * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(random_seed)

    platform = None  # let OpenMM choose (CUDA/CPU)
    context = openmm.Context(system, integrator, platform) if platform else openmm.Context(system, integrator)

    # Set positions
    context.setPositions(positions_nm * unit.nanometer)

    # Minimize to remove overlaps
    openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.0 * unit.kilojoule_per_mole, maxIterations=200)

    # Simulation loop
    report_interval_steps = max(1, steps // 100)  # ~100 samples over the run
    time_ps = 0.0
    dt_ps = timestep_fs * 0.001

    # Index bookkeeping
    met_indices: List[int] = []
    water_indices: List[int] = []
    nonwater_indices: List[int] = []
    start = 0
    for comp in components_all:
        indices = list(range(start, start + comp.count))
        if comp.label == "metformin":
            met_indices.extend(indices)
        if comp.label == "water":
            water_indices.extend(indices)
        else:
            nonwater_indices.extend(indices)
        start += comp.count

    # Initial metrics (compute centroid/radius using non-water)
    state = context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if nonwater_indices:
        pos_nonwater = pos_nm[nonwater_indices]
    else:
        pos_nonwater = pos_nm
    centroid = compute_centroid(pos_nonwater) if pos_nonwater.size else np.array([
        0.5 * box_nm, 0.5 * box_nm, 0.5 * box_nm
    ])
    tablet_radius = compute_tablet_radius_nm(pos_nonwater, centroid) if pos_nonwater.size else 0.5

    # Analysis configuration
    analysis_cfg = cfg.get("analysis", {})
    solvation_cutoff_nm = float(analysis_cfg.get("solvation_cutoff_nm", 0.6))
    concentration_bins = int(analysis_cfg.get("concentration_bins", 20))
    output_frequency = int(analysis_cfg.get("output_frequency", 10))  # Steps between analysis outputs
    
    # Dissolution physics configuration
    dissolution_cfg = cfg.get("dissolution_physics", {})
    enable_noyes_whitney = bool(dissolution_cfg.get("enable_noyes_whitney", False))
    dissolution_params = None
    dissolving_beads = []
    dissolved_positions = np.zeros((0, 3))  # Track positions of fully dissolved beads
    
    if enable_noyes_whitney:
        dissolution_params = DissolutionParams(
            rate_constant_nm_per_ps=float(dissolution_cfg.get("rate_constant_nm_per_ps", 0.001)),
            saturation_concentration_beads_per_nm3=float(dissolution_cfg.get("saturation_concentration_beads_per_nm3", 0.1)),
            min_bead_mass_fraction=float(dissolution_cfg.get("min_bead_mass_fraction", 0.05)),
            concentration_shell_nm=float(dissolution_cfg.get("concentration_shell_nm", 1.0)),
            apply_to_components=dissolution_cfg.get("apply_to_components", ["metformin"])
        )
        
        # Initialize DissolvingBead objects for dissolving components
        for comp in components_all:
            if any(target in comp.label for target in dissolution_params.apply_to_components):
                for _ in range(comp.count):
                    dissolving_beads.append(DissolvingBead(
                        component_label=comp.label,
                        original_mass_amu=comp.bead_type.mass_amu,
                        current_mass_amu=comp.bead_type.mass_amu,
                        original_sigma_nm=comp.bead_type.sigma_nm,
                        current_sigma_nm=comp.bead_type.sigma_nm,
                        current_epsilon_kj_mol=comp.bead_type.epsilon_kj_mol,
                    ))
            else:
                # Non-dissolving components get placeholder beads
                for _ in range(comp.count):
                    dissolving_beads.append(DissolvingBead(
                        component_label=comp.label,
                        original_mass_amu=comp.bead_type.mass_amu,
                        current_mass_amu=comp.bead_type.mass_amu,
                        original_sigma_nm=comp.bead_type.sigma_nm,
                        current_sigma_nm=comp.bead_type.sigma_nm,
                        current_epsilon_kj_mol=comp.bead_type.epsilon_kj_mol,
                        is_dissolved=False  # Non-dissolving components never dissolve
                    ))
    
    # pH effects configuration
    ph_cfg = cfg.get("ph_effects", {})
    enable_ph_effects = bool(ph_cfg.get("enable_ph_effects", False))
    ph_params = None
    
    if enable_ph_effects:
        # Parse GI phases
        gi_phases = []
        for phase_cfg in ph_cfg.get("gi_phases", []):
            gi_phases.append(GIPhase(
                name=str(phase_cfg.get("name", "unknown")),
                ph=float(phase_cfg.get("ph", 7.0)),
                ionic_strength_m=float(phase_cfg.get("ionic_strength_m", 0.1)),
                duration_ps=float(phase_cfg.get("duration_ps", 3600000)),
                buffer_capacity=float(phase_cfg.get("buffer_capacity", 0.01))
            ))
        
        # Parse species data
        species_data = {}
        for species_name, species_cfg in ph_cfg.get("species_data", {}).items():
            species_data[species_name] = IonicSpecies(
                name=species_name,
                pka_values=species_cfg.get("pka_values", []),
                base_charge=int(species_cfg.get("base_charge", 0)),
                solubility_factors=species_cfg.get("solubility_factors", [1.0]),
                rate_factors=species_cfg.get("rate_factors", [1.0])
            )
        
        ph_params = PHParams(
            enable_ph_effects=True,
            gi_phases=gi_phases,
            species_data=species_data,
            temperature_k=float(ph_cfg.get("temperature_k", 310.0)),
            use_davies_equation=bool(ph_cfg.get("use_davies_equation", True))
        )
    
    # Output setup
    metrics_path = f"{output_prefix}_metrics.csv"
    solvation_path = f"{output_prefix}_solvation.csv"
    concentration_path = f"{output_prefix}_concentration.csv"
    penetration_path = f"{output_prefix}_penetration.csv"
    dissolution_path = f"{output_prefix}_dissolution.csv"
    ph_path = f"{output_prefix}_ph_effects.csv"
    pdb_path = f"{output_prefix}_final_positions.pdb"
    
    # Create analysis output files if water is present
    solvation_file = open(solvation_path, "w", newline="") if water_indices else None
    concentration_file = open(concentration_path, "w", newline="")
    penetration_file = open(penetration_path, "w", newline="") if water_indices else None
    dissolution_file = open(dissolution_path, "w", newline="") if enable_noyes_whitney else None
    ph_file = open(ph_path, "w", newline="") if enable_ph_effects else None
    
    try:
        with open(metrics_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time_ps", "num_dissolved", "fraction_dissolved", "tablet_radius_nm"])
            
            # Initialize analysis writers
            solvation_writer = None
            concentration_writer = None
            penetration_writer = None
            dissolution_writer = None
            ph_writer = None
            
            if solvation_file:
                solvation_writer = csv.writer(solvation_file)
                solvation_writer.writerow(["time_ps", "mean_solvation_number", "std_solvation_number", 
                                         "min_solvation_number", "max_solvation_number"])
            
            concentration_writer = csv.writer(concentration_file)
            # Write header with bin center positions
            bin_centers, _ = calculate_radial_concentration(
                np.zeros((0, 3)), np.array([0, 0, 0]), box_nm, concentration_bins
            )
            conc_header = ["time_ps"] + [f"concentration_r_{r:.3f}_nm" for r in bin_centers]
            concentration_writer.writerow(conc_header)
            
            if penetration_file:
                penetration_writer = csv.writer(penetration_file)
                penetration_writer.writerow(["time_ps", "water_inside_tablet", "max_penetration_depth_nm", 
                                           "fraction_infiltrated"])
            
            if dissolution_file:
                dissolution_writer = csv.writer(dissolution_file)
                dissolution_writer.writerow(["time_ps", "total_dissolved_mass_amu", "mean_mass_fraction_remaining", 
                                           "num_fully_dissolved", "dissolution_rate_amu_per_ps"])
            
            if ph_file:
                ph_writer = csv.writer(ph_file)
                ph_writer.writerow(["time_ps", "gi_phase", "ph", "ionic_strength_m", "metformin_charge", 
                                   "solubility_factor", "rate_factor", "activity_coefficient"])

            # Heuristics for dissolution classification
            radial_margin_nm = 1.5
            contact_cutoff_nm = 0.8
            water_shell_nm = 0.6

            # Write initial line
            met_pos = pos_nm[met_indices] if met_indices else np.zeros((0, 3))
            if water_indices:
                water_pos = pos_nm[water_indices]
                nonwater_pos = pos_nm[nonwater_indices] if nonwater_indices else np.zeros((0, 3))
                dissolved = count_dissolved_metformin_explicit(
                    met_pos,
                    pos_nm,
                    nonwater_pos,
                    water_pos,
                    centroid,
                    tablet_radius,
                    radial_margin_nm,
                    contact_cutoff_nm,
                    water_shell_nm,
                    min_water_neighbors=1,
                )
            else:
                dissolved = count_dissolved_metformin(
                    met_pos, pos_nm, centroid, tablet_radius, radial_margin_nm, contact_cutoff_nm
                )
            frac = (dissolved / len(met_indices)) if met_indices else 0.0
            writer.writerow([f"{time_ps:.3f}", dissolved, f"{frac:.6f}", f"{tablet_radius:.4f}"])
            
            # Initial analysis output
            def write_analysis_data(step_num: int):
                """Write analysis data for the current step."""
                if solvation_writer and water_indices:
                    water_pos = pos_nm[water_indices]
                    solvation_stats = analyze_solvation_shells(met_pos, water_pos, solvation_cutoff_nm)
                    solvation_writer.writerow([
                        f"{time_ps:.3f}",
                        f"{solvation_stats['mean_solvation_number']:.4f}",
                        f"{solvation_stats['std_solvation_number']:.4f}",
                        f"{solvation_stats['min_solvation_number']:.1f}",
                        f"{solvation_stats['max_solvation_number']:.1f}",
                    ])
                
                # Concentration profile
                bin_centers, concentrations = calculate_radial_concentration(
                    met_pos, centroid, box_nm, concentration_bins
                )
                conc_row = [f"{time_ps:.3f}"] + [f"{c:.6f}" for c in concentrations]
                concentration_writer.writerow(conc_row)
                
                # Water infiltration
                if penetration_writer and water_indices:
                    water_pos = pos_nm[water_indices]
                    infiltration_stats = track_water_infiltration(water_pos, centroid, tablet_radius)
                    penetration_writer.writerow([
                        f"{time_ps:.3f}",
                        f"{infiltration_stats['water_inside_tablet']:.1f}",
                        f"{infiltration_stats['max_penetration_depth_nm']:.4f}",
                        f"{infiltration_stats['fraction_infiltrated']:.6f}",
                    ])
                
                # Dissolution kinetics output
                if dissolution_writer and enable_noyes_whitney:
                    # Calculate dissolution metrics
                    total_dissolved_mass = sum(bead.cumulative_dissolved_mass_amu for bead in dissolving_beads)
                    active_beads = [bead for bead in dissolving_beads if any(comp in bead.component_label for comp in dissolution_params.apply_to_components)]
                    if active_beads:
                        mean_mass_fraction = sum(bead.mass_fraction_remaining for bead in active_beads) / len(active_beads)
                        num_fully_dissolved = sum(1 for bead in active_beads if bead.is_dissolved)
                        # Approximate dissolution rate from last time step
                        dissolution_rate = total_dissolved_mass / max(time_ps, dt_ps) if time_ps > 0 else 0.0
                    else:
                        mean_mass_fraction = 0.0
                        num_fully_dissolved = 0
                        dissolution_rate = 0.0
                    
                    dissolution_writer.writerow([
                        f"{time_ps:.3f}",
                        f"{total_dissolved_mass:.4f}",
                        f"{mean_mass_fraction:.6f}",
                        num_fully_dissolved,
                        f"{dissolution_rate:.6f}",
                    ])
                
                # pH effects output
                if ph_writer and enable_ph_effects:
                    current_phase, phase_progress = get_current_gi_phase(time_ps, ph_params.gi_phases)
                    
                    # Calculate metformin ionization if present
                    if "metformin" in ph_params.species_data:
                        metformin_species = ph_params.species_data["metformin"]
                        fractions = calculate_ionization_fractions(current_phase.ph, metformin_species.pka_values, metformin_species.base_charge)
                        avg_charge = calculate_average_charge(fractions, metformin_species.base_charge, metformin_species.pka_values)
                        
                        # Calculate factors
                        solubility_factor = sum(f * sf for f, sf in zip(fractions, metformin_species.solubility_factors))
                        rate_factor = sum(f * rf for f, rf in zip(fractions, metformin_species.rate_factors))
                        activity_coeff = calculate_davies_activity_coefficient(avg_charge, current_phase.ionic_strength_m)
                    else:
                        avg_charge = 0.0
                        solubility_factor = 1.0
                        rate_factor = 1.0
                        activity_coeff = 1.0
                    
                    ph_writer.writerow([
                        f"{time_ps:.3f}",
                        current_phase.name,
                        f"{current_phase.ph:.2f}",
                        f"{current_phase.ionic_strength_m:.4f}",
                        f"{avg_charge:.4f}",
                        f"{solubility_factor:.6f}",
                        f"{rate_factor:.6f}",
                        f"{activity_coeff:.6f}",
                    ])
            
            write_analysis_data(0)

            # Dynamics
            step_iter = range(1, steps + 1)
            if tqdm is not None:
                step_iter = tqdm(step_iter, total=steps, desc="Simulating", unit="step")
            for step in step_iter:
                integrator.step(1)
                time_ps += dt_ps
                
                # Apply Noyes-Whitney dissolution kinetics if enabled
                if enable_noyes_whitney and dissolution_params:
                    state = context.getState(getPositions=True)
                    current_pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    
                    # Get current pH-dependent dissolution parameters
                    current_dissolution_params = dissolution_params
                    if enable_ph_effects and ph_params:
                        current_phase, _ = get_current_gi_phase(time_ps, ph_params.gi_phases)
                        current_dissolution_params = get_ph_dependent_dissolution_params(
                            dissolution_params,
                            "metformin",  # Primary dissolving species
                            current_phase.ph,
                            current_phase.ionic_strength_m,
                            ph_params.species_data
                        )
                    
                    # Apply dissolution kinetics with pH-dependent parameters
                    dissolving_beads, indices_to_remove = apply_noyes_whitney_dissolution(
                        dissolving_beads,
                        current_pos_nm,
                        dissolved_positions,
                        current_dissolution_params,
                        dt_ps
                    )
                    
                    # Update positions of dissolved beads for concentration calculations
                    if indices_to_remove:
                        dissolved_pos_list = [current_pos_nm[i] for i in indices_to_remove]
                        if dissolved_pos_list:
                            new_dissolved = np.array(dissolved_pos_list)
                            dissolved_positions = np.vstack([dissolved_positions, new_dissolved]) if dissolved_positions.size > 0 else new_dissolved
                    
                    # Update OpenMM parameters for changed beads
                    update_nonbonded_parameters(nb, dissolving_beads, context)

                if step % report_interval_steps == 0 or step == steps:
                    state = context.getState(getPositions=True)
                    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    # Compute centroid/radius from non-water positions
                    if nonwater_indices:
                        pos_nonwater = pos_nm[nonwater_indices]
                    else:
                        pos_nonwater = pos_nm
                    centroid = compute_centroid(pos_nonwater) if pos_nonwater.size else np.array([
                        0.5 * box_nm, 0.5 * box_nm, 0.5 * box_nm
                    ])
                    tablet_radius = compute_tablet_radius_nm(pos_nonwater, centroid) if pos_nonwater.size else 0.5
                    met_pos = pos_nm[met_indices] if met_indices else np.zeros((0, 3))
                    if water_indices:
                        water_pos = pos_nm[water_indices]
                        nonwater_pos = pos_nm[nonwater_indices] if nonwater_indices else np.zeros((0, 3))
                        dissolved = count_dissolved_metformin_explicit(
                            met_pos,
                            pos_nm,
                            nonwater_pos,
                            water_pos,
                            centroid,
                            tablet_radius,
                            radial_margin_nm,
                            contact_cutoff_nm,
                            water_shell_nm,
                            min_water_neighbors=1,
                        )
                    else:
                        dissolved = count_dissolved_metformin(
                            met_pos, pos_nm, centroid, tablet_radius, radial_margin_nm, contact_cutoff_nm
                        )
                    frac = (dissolved / len(met_indices)) if met_indices else 0.0
                    writer.writerow([f"{time_ps:.3f}", dissolved, f"{frac:.6f}", f"{tablet_radius:.4f}"])
                    
                    # Write detailed analysis data at specified frequency
                    if step % (report_interval_steps * output_frequency) == 0 or step == steps:
                        write_analysis_data(step)
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise
    finally:
        # Close analysis files
        if solvation_file:
            solvation_file.close()
        if concentration_file:
            concentration_file.close()
        if penetration_file:
            penetration_file.close()
        if dissolution_file:
            dissolution_file.close()
        if ph_file:
            ph_file.close()

    # Final structure output (exclude water beads, include all other ingredients)
    final_state = context.getState(getPositions=True)
    final_positions_nm = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    nonwater_mask_list = [lab != "water" for lab in labels]
    if any(nonwater_mask_list):
        nonwater_mask = np.array(nonwater_mask_list, dtype=bool)
        filtered_positions_nm = final_positions_nm[nonwater_mask]
        filtered_labels = [lab for lab in labels if lab != "water"]
    else:
        filtered_positions_nm = final_positions_nm
        filtered_labels = labels
    write_cg_pdb(pdb_path, filtered_positions_nm, filtered_labels, box_nm)

    # Clean up
    del context
    del integrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse-grained dissolution simulator (OpenMM)")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument(
        "--output",
        required=True,
        help="Output prefix (directory must exist or will be created). Example: outputs/run1",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of integration steps (default derived from config time & timestep)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    run_simulation(args.config, args.output, override_steps=args.steps)


if __name__ == "__main__":
    main()


