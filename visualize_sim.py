"""visualize_sim.py — save simulated movies as a tiled TIFF for inspection.

Layout: 2 rows × 3 columns, one panel per species in MW order.
  Row 0:  monomer | dimer   | trimer
  Row 1: tetramer | pentamer | hexamer

Each panel is contrast-scaled independently so the monomer (low contrast)
is as visible as the hexamer.  A raw (shared-scale) TIFF is also saved.

Output (written to CACHE_DIR):
    sim_tiled_scaled.tiff   -- per-panel contrast scaled  (recommended for inspection)
    sim_tiled_raw.tiff      -- all panels on the same contrast scale

Usage
-----
    python visualize_sim.py              # train split (default)
    python visualize_sim.py --split test
    python visualize_sim.py --frames 200 # save only first 200 frames (faster)
"""

import argparse
import os
import sys

import numpy as np
import tifffile

# Allow running from anywhere
sys.path.insert(0, os.path.dirname(__file__))
from prepare import CACHE_DIR, SIM_SPECIES

GRID_ROWS = 2
GRID_COLS = 3
GAP_PX    = 2   # separator between panels (filled with 0)

SPECIES_ORDER = [sp["label"] for sp in SIM_SPECIES]  # monomer → hexamer


def load_movies(split: str) -> dict[str, np.ndarray]:
    split_dir = os.path.join(CACHE_DIR, split)
    if not os.path.isdir(split_dir):
        sys.exit(f"Cache not found at {split_dir} — run: python prepare.py")
    movies = {}
    for label in SPECIES_ORDER:
        path = os.path.join(split_dir, f"{label}_movie.npy")
        if not os.path.exists(path):
            sys.exit(f"Missing: {path} — run: python prepare.py")
        movies[label] = np.load(path)
        T, H, W = movies[label].shape
        noise = np.load(os.path.join(split_dir, f"{label}_noise.npy"))[0]
        print(f"  {label:<12}  shape={T}×{H}×{W}  "
              f"contrast=[{movies[label].min():.4f}, {movies[label].max():.4f}]  "
              f"noise={noise:.5f}")
    return movies


def build_tiled(movies: dict, n_frames: int, normalize: bool) -> np.ndarray:
    """Stack all species into a (T, grid_H, grid_W) float32 array."""
    sample     = next(iter(movies.values()))
    T_src, H, W = sample.shape
    T          = min(n_frames, T_src)

    grid_H = GRID_ROWS * H + (GRID_ROWS - 1) * GAP_PX
    grid_W = GRID_COLS * W + (GRID_COLS - 1) * GAP_PX
    out    = np.zeros((T, grid_H, grid_W), dtype=np.float32)

    for idx, label in enumerate(SPECIES_ORDER):
        row = idx // GRID_COLS
        col = idx %  GRID_COLS
        y0  = row * (H + GAP_PX)
        x0  = col * (W + GAP_PX)

        panel = movies[label][:T].copy()

        if normalize:
            # Scale to [0, 1] using the 1st–99th percentile of the full panel
            lo = np.percentile(panel, 1)
            hi = np.percentile(panel, 99)
            if hi > lo:
                panel = np.clip((panel - lo) / (hi - lo), 0.0, 1.0)

        out[:, y0: y0 + H, x0: x0 + W] = panel

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize simulated MP movies as tiled TIFF")
    parser.add_argument("--split",  default="train", choices=["train", "test"])
    parser.add_argument("--frames", type=int, default=None,
                        help="Number of frames to include (default: all)")
    args = parser.parse_args()

    print(f"Loading {args.split} movies from {CACHE_DIR} …")
    movies   = load_movies(args.split)
    n_frames = args.frames or next(iter(movies.values())).shape[0]

    print(f"\nBuilding tiled movies ({n_frames} frames, "
          f"{GRID_ROWS}×{GRID_COLS} grid) …")

    # Per-panel scaled (best for visual inspection)
    tiled_scaled = build_tiled(movies, n_frames, normalize=True)
    out_scaled   = os.path.join(CACHE_DIR, f"sim_tiled_scaled_{args.split}.tiff")
    tifffile.imwrite(out_scaled, tiled_scaled, imagej=True)
    print(f"  Saved (per-panel scaled) → {out_scaled}")
    print(f"  Shape: {tiled_scaled.shape}  dtype: {tiled_scaled.dtype}")

    # Raw contrast (shared scale — shows true relative amplitudes)
    tiled_raw  = build_tiled(movies, n_frames, normalize=False)
    out_raw    = os.path.join(CACHE_DIR, f"sim_tiled_raw_{args.split}.tiff")
    tifffile.imwrite(out_raw, tiled_raw, imagej=True)
    print(f"  Saved (raw contrast)     → {out_raw}")

    print(f"""
Panel layout (row × col):
  [0,0] monomer  (55 kDa)    [0,1] dimer    (110 kDa)  [0,2] trimer   (165 kDa)
  [1,0] tetramer (220 kDa)   [1,1] pentamer (275 kDa)  [1,2] hexamer  (330 kDa)

Open in ImageJ/FIJI: File → Open, then Image → Adjust → Brightness/Contrast.
The scaled TIFF normalizes each panel independently so the monomer is visible.
The raw TIFF preserves true contrast ratios (monomer will appear dim vs hexamer).
""")


if __name__ == "__main__":
    main()
