#!/usr/bin/env python3
"""
Simple PDB visualizer for coarse-grained beads.

Features:
- Colors by residue name (MET, FIL, BIN, LUB). Water typically excluded upstream.
- Interactive 3D scatter using matplotlib.
- Optional headless save to PNG.

Usage examples:
  python visualize_pdb.py --pdb outputs/metformin_run_tqdm_final_positions.pdb
  python visualize_pdb.py --pdb outputs/metformin_run_tqdm_final_positions.pdb \
    --save outputs/preview.png --no-show --size 5 --dpi 150
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CG PDB as 3D scatter (colored by residue)")
    parser.add_argument("--pdb", required=True, help="Path to PDB file (CG atoms)")
    parser.add_argument("--save", default=None, help="Optional path to save a PNG image")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    parser.add_argument("--size", type=float, default=4.0, help="Marker size")
    parser.add_argument("--alpha", type=float, default=0.9, help="Marker transparency")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI when saving")
    parser.add_argument("--downsample", type=int, default=0, help="Randomly keep only N beads (0 = keep all)")
    parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle for view_init")
    parser.add_argument("--azim", type=float, default=-60.0, help="Azimuth angle for view_init")
    return parser.parse_args()


def load_pdb_atoms(pdb_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load atom coordinates and residue names from a minimal ATOM-only PDB.

    Returns:
        positions (N,3) in Angstrom
        residue_names (len N) three-letter codes
    """
    coords: List[List[float]] = []
    resnames: List[str] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            # Our PDB writer produces lines like:
            # ATOM  idx  C RES Aresid   x y z  ...
            # Use split to be robust to spacing
            parts = line.split()
            if len(parts) < 9:
                continue
            res = parts[3].upper()
            try:
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
            except ValueError:
                continue
            coords.append([x, y, z])
            resnames.append(res)
    if not coords:
        return np.zeros((0, 3), dtype=float), []
    return np.array(coords, dtype=float), resnames


def get_color_map() -> Dict[str, str]:
    # Distinct colors for residues; fallback to gray for unknowns
    return {
        "MET": "#1f77b4",  # blue
        "FIL": "#ff7f0e",  # orange
        "BIN": "#2ca02c",  # green
        "LUB": "#d62728",  # red
    }


def visualize(points_ang: np.ndarray, resnames: List[str], save_path: str, no_show: bool,
              size: float, alpha: float, dpi: int, elev: float, azim: float) -> None:
    if no_show or save_path:
        # Use non-interactive backend if only saving or explicitly no_show
        import matplotlib
        if no_show:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

    cmap = get_color_map()
    unique_res = sorted(set(resnames))

    fig = plt.figure(figsize=(8, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Plot per residue for legend clarity
    for res in unique_res:
        mask = np.array([r == res for r in resnames], dtype=bool)
        color = cmap.get(res, "#7f7f7f")
        if np.any(mask):
            ax.scatter(points_ang[mask, 0], points_ang[mask, 1], points_ang[mask, 2],
                       s=size, c=color, alpha=alpha, depthshade=False, label=res)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right")
    ax.set_box_aspect((1, 1, 1))

    # Fit axes limits
    if points_ang.size > 0:
        mins = points_ang.min(axis=0)
        maxs = points_ang.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = (maxs - mins).max()
        mins_fit = center - extent / 2.0
        maxs_fit = center + extent / 2.0
        ax.set_xlim(mins_fit[0], maxs_fit[0])
        ax.set_ylim(mins_fit[1], maxs_fit[1])
        ax.set_zlim(mins_fit[2], maxs_fit[2])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if not no_show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pdb_path = os.path.abspath(args.pdb)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    coords_ang, resnames = load_pdb_atoms(pdb_path)
    if args.downsample and coords_ang.shape[0] > args.downsample:
        rng = np.random.default_rng(2025)
        idx = rng.choice(coords_ang.shape[0], size=args.downsample, replace=False)
        coords_ang = coords_ang[idx]
        resnames = [resnames[i] for i in idx]

    visualize(coords_ang, resnames, args.save, args.no_show, args.size, args.alpha, args.dpi, args.elev, args.azim)


if __name__ == "__main__":
    main()


