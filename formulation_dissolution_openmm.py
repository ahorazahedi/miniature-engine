#!/usr/bin/env python3
"""
Coarse-grained dissolution simulator for a single tablet formulation using OpenMM.

This script ignores any existing project code and builds a minimal, self-contained
coarse-grained (CG) molecular dynamics model that approximates dissolution behavior
for a metformin tablet with excipients (filler, binder, lubricant, others).

Core ideas (simplified for speed and robustness):
- Represent each component (metformin, filler, binder, lubricant, other) as CG beads.
- Beads interact via Lennard-Jones (LJ) potentials with type-specific epsilon/sigma.
- No permanent bonds are used; cohesion is via LJ attraction (binder has stronger
  cohesion with others). This allows the cluster to disperse under thermal motion.
- Use implicit solvent (no explicit water) with Langevin dynamics to approximate
  hydrodynamic/solvent effects while keeping runtime small and avoiding water models.
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
    ]
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

    components: List[ComponentSpec] = []

    # Metformin
    met_spec = formulation.get("metformin", {})
    met_beads = int(met_spec.get("beads", met_spec.get("molecules", 200)))
    met_params = met_spec.get("parameters", {})
    met_type = resolve_bead_type("metformin", met_params)
    components.append(ComponentSpec(label="metformin", bead_type=met_type, count=met_beads))

    # Excipients
    for item in formulation.get("excipients", []):
        name = str(item.get("name", "generic"))
        role = str(item.get("role", "other"))
        label = f"{role}:{name}"
        count = int(item.get("beads", 100))
        params = item.get("parameters", {})
        bead_type = resolve_bead_type(name, params)
        components.append(ComponentSpec(label=label, bead_type=bead_type, count=count))

    # Create initial positions (cluster) and labels
    counts = {comp.label: comp.count for comp in components}
    spacing_nm = 0.60  # lattice spacing roughly corresponding to sigma scale
    positions_by_label = generate_cluster_positions(counts, spacing_nm, box_nm, random_seed)
    labels: List[str] = []
    positions_list: List[np.ndarray] = []
    for comp in components:
        labels.extend([comp.label] * comp.count)
        positions_list.append(positions_by_label[comp.label])
    positions_nm = np.vstack(positions_list)

    # Build system
    cutoff_nm = 1.2
    system, nb = build_system(components, cutoff_nm=cutoff_nm, use_pbc=True, box_size_nm=box_nm)

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

    # Initial metrics
    state = context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    centroid = compute_centroid(pos_nm)
    tablet_radius = compute_tablet_radius_nm(pos_nm, centroid)

    met_indices = []
    start = 0
    for comp in components:
        if comp.label == "metformin":
            met_indices.extend(range(start, start + comp.count))
        start += comp.count

    # Output setup
    metrics_path = f"{output_prefix}_metrics.csv"
    pdb_path = f"{output_prefix}_final_positions.pdb"
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_ps", "num_dissolved", "fraction_dissolved", "tablet_radius_nm"])

        # Heuristics for dissolution classification
        radial_margin_nm = 1.5
        contact_cutoff_nm = 0.8

        # Write initial line
        met_pos = pos_nm[met_indices]
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
                centroid = compute_centroid(pos_nm)
                tablet_radius = compute_tablet_radius_nm(pos_nm, centroid)
                met_pos = pos_nm[met_indices]
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


