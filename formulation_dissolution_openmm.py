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


# ------------------------------ Defaults / Library ---------------------------


def get_default_bead_library() -> Dict[str, BeadType]:
    """Return a default library of CG bead parameters for ingredients.

    Epsilon/sigma are loosely tuned to encode relative cohesion/hydrophilicity.
    Values are in kJ/mol and nm. Mass is in atomic mass units (Daltons).
    """
    return {
        # Metformin: small, hydrophilic API
        "metformin": BeadType(name="metformin", epsilon_kj_mol=1.10, sigma_nm=0.47, mass_amu=129.0),
        # Water (CG one-bead): small sigma, moderate epsilon
        "water": BeadType(name="water", epsilon_kj_mol=0.75, sigma_nm=0.32, mass_amu=18.0),
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

    # Output setup
    metrics_path = f"{output_prefix}_metrics.csv"
    pdb_path = f"{output_prefix}_final_positions.pdb"
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_ps", "num_dissolved", "fraction_dissolved", "tablet_radius_nm"])

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

        # Dynamics
        for step in range(1, steps + 1):
            integrator.step(1)
            time_ps += dt_ps

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

    # Final structure output
    final_state = context.getState(getPositions=True)
    final_positions_nm = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    write_cg_pdb(pdb_path, final_positions_nm, labels, box_nm)

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


