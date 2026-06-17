"""prepare.py — one-time data generation and frozen evaluation harness.

Frozen: do not modify.

Usage
-----
    python prepare.py            # generate train + test data (~20 min first run)
    python prepare.py --check    # verify cache exists without regenerating
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tifffile

from sim import (
    BrownianSimulation, MPPSF, ExperimentalNoise,
    select_clean_noise_frames, generate_contrast_movie,
    trajectories_to_dataframe, load_calibration,
)

# ── frozen paths ──────────────────────────────────────────────────────────────
EMPTY_TIFF = os.environ.get(
    "MP_EMPTY_TIFF",
    "/Volumes/data/kimberlyc/MP_data/MA_MBP/5nM_MA_MBP/Trial_1/"
    "001_Empty_2_per_PIP2_5_min_1_new_PIP2.tiff",
)
MC_FILE = os.environ.get(
    "MP_MC_FILE",
    "/Volumes/data/kimberlyc/MP_data/MA_MBP/5nM_MA_MBP/Trial_1/MC_2_25_26.mc",
)
CACHE_DIR = os.environ.get(
    "MP_CACHE_DIR",
    os.path.expanduser("~/.cache/autoresearch-mp"),
)

# ── frozen simulation constants ───────────────────────────────────────────────
FOV_H         = 59
FOV_W         = 150
PIXEL_SIZE_NM = 84.4
FRAME_RATE    = 475.0
SIM_N_FRAMES  = 10_000
N_PARTICLES   = 80      # in simulation box; ~20 visible in FOV at any frame
ROI_HALF      = 6       # edge exclusion margin in pixels

# Diffusion coefficients (µm²/s) — TODO: replace with experimentally measured values
# Approximate Saffman-Delbrück scaling relative to monomer
SIM_SPECIES = [
    dict(label="monomer",  oligomer=1, mass_kda=55,  D_um2s=4.0),
    dict(label="dimer",    oligomer=2, mass_kda=110, D_um2s=2.8),
    dict(label="trimer",   oligomer=3, mass_kda=165, D_um2s=2.3),
    dict(label="tetramer", oligomer=4, mass_kda=220, D_um2s=2.0),
    dict(label="pentamer", oligomer=5, mass_kda=275, D_um2s=1.8),
    dict(label="hexamer",  oligomer=6, mass_kda=330, D_um2s=1.6),
]

CLASS_NAMES  = ["noise"] + [sp["label"] for sp in SIM_SPECIES]
N_CLASSES    = len(CLASS_NAMES)   # 7
LABEL_TO_NAME = {i: n for i, n in enumerate(CLASS_NAMES)}
NAME_TO_LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}

TRAIN_SEED_BASE = 42
TEST_SEED_BASE  = 990

# Evaluation constants — frozen
MATCH_RADIUS_PX = 1.0    # GT matching radius
FDR_TARGET      = 0.10   # informational target only; not enforced


# ── data generation ───────────────────────────────────────────────────────────

def compute_noise_level(movie: np.ndarray) -> float:
    """Robust MAD-based noise estimate (mean over frames)."""
    from scipy.stats import median_abs_deviation
    mads = [median_abs_deviation(frame.ravel()) for frame in movie]
    return 1.4826 * float(np.mean(mads))


def _generate_split(split: str, noise_model: ExperimentalNoise, calib) -> None:
    seed_base = TRAIN_SEED_BASE if split == "train" else TEST_SEED_BASE
    out_dir   = os.path.join(CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    psf = MPPSF()

    for i, sp in enumerate(SIM_SPECIES):
        print(f"\n  [{split}] Simulating {sp['label']} (seed={seed_base + i}) …")
        sim = BrownianSimulation(
            n_particles   = N_PARTICLES,
            n_frames      = SIM_N_FRAMES,
            pixel_size_nm = PIXEL_SIZE_NM,
            fov_shape     = (FOV_H, FOV_W),
            species       = {sp["label"]: (1.0, sp["D_um2s"], sp["mass_kda"])},
            frame_rate    = FRAME_RATE,
            seed          = seed_base + i,
        ).run()

        movie = generate_contrast_movie(
            sim, psf,
            noise_model = noise_model,
            calibration = calib,
            noise_seed  = seed_base + i + 1000,
        )

        gt_df = trajectories_to_dataframe(sim, calibration=calib)
        gt_df["oligomer_label"] = sp["oligomer"]   # integer class label 1–6

        noise_level = compute_noise_level(movie)

        stem = sp["label"]
        np.save(os.path.join(out_dir, f"{stem}_movie.npy"), movie)
        with open(os.path.join(out_dir, f"{stem}_gt.pkl"), "wb") as f:
            pickle.dump(gt_df, f)
        np.save(os.path.join(out_dir, f"{stem}_noise.npy"),
                np.array([noise_level], dtype=np.float32))

        print(f"    {len(gt_df):,} GT observations  noise={noise_level:.5f}  "
              f"saved → {out_dir}/{stem}_*")


def generate() -> None:
    """One-time: generate train and test data from simulation + empty bilayer."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load empty bilayer noise model
    print(f"Loading empty bilayer: {EMPTY_TIFF}")
    empty_frames = tifffile.imread(EMPTY_TIFF).astype(np.float32)
    if empty_frames.ndim == 2:
        empty_frames = empty_frames[np.newaxis]
    print(f"  Loaded {empty_frames.shape[0]} frames  "
          f"({empty_frames.shape[1]}×{empty_frames.shape[2]} px)")

    mask          = select_clean_noise_frames(empty_frames)
    noise_model   = ExperimentalNoise(empty_frames[mask])
    print(f"  Noise model: {noise_model}")

    # Load calibration
    calib = load_calibration(MC_FILE)

    # Generate train and test splits
    for split in ("train", "test"):
        print(f"\n{'='*60}")
        print(f"Generating {split} set …")
        _generate_split(split, noise_model, calib)

    print(f"\nDone. Data written to {CACHE_DIR}")


# ── dataset loader ────────────────────────────────────────────────────────────

def load_dataset(split: str) -> list[dict]:
    """Load pre-generated movies and ground-truth DataFrames.

    Returns list of dicts (one per species):
        {
            'species'  : str,
            'oligomer' : int,
            'movie'    : np.ndarray (T, H, W) float32,
            'gt'       : pd.DataFrame  [frame, x, y, species_label, oligomer_label, ...],
            'noise'    : float,
        }
    """
    split_dir = os.path.join(CACHE_DIR, split)
    if not os.path.isdir(split_dir):
        sys.exit(f"ERROR: cache not found at {split_dir}\nRun: python prepare.py")

    data = []
    for sp in SIM_SPECIES:
        stem = sp["label"]
        movie_path = os.path.join(split_dir, f"{stem}_movie.npy")
        gt_path    = os.path.join(split_dir, f"{stem}_gt.pkl")
        noise_path = os.path.join(split_dir, f"{stem}_noise.npy")
        for p in (movie_path, gt_path, noise_path):
            if not os.path.exists(p):
                sys.exit(f"ERROR: missing {p}\nRun: python prepare.py")

        movie       = np.load(movie_path)
        with open(gt_path, "rb") as f:
            gt_df = pickle.load(f)
        noise_level = float(np.load(noise_path)[0])

        data.append(dict(
            species  = sp["label"],
            oligomer = sp["oligomer"],
            movie    = movie,
            gt       = gt_df,
            noise    = noise_level,
        ))
    return data


# ── matching ──────────────────────────────────────────────────────────────────

def match_to_gt(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match detected peaks to ground truth within MATCH_RADIUS_PX, per frame.

    Parameters
    ----------
    preds_df:
        Columns required: frame (int), y (int/float), x (int/float).
    gt_df:
        Ground truth DataFrame from load_dataset(); columns include
        frame, y, x, oligomer_label.

    Returns
    -------
    pd.DataFrame
        preds_df with added columns:
        - matched     : bool — True if within radius of a GT peak
        - true_label  : int  — oligomer label (1–6) if matched, else 0 (noise)
    """
    preds = preds_df.copy()
    preds["matched"]    = False
    preds["true_label"] = 0   # default: noise

    for frame_id, frame_preds in preds.groupby("frame"):
        frame_gt = gt_df[gt_df["frame"] == frame_id]
        if frame_gt.empty or frame_preds.empty:
            continue

        p_yx = frame_preds[["y", "x"]].values.astype(float)
        g_yx = frame_gt[["y", "x"]].values.astype(float)

        # Greedy nearest-neighbor matching (sufficient for ~20 particles/frame)
        used_gt = set()
        for pi, p_idx in enumerate(frame_preds.index):
            dists = np.sqrt(np.sum((g_yx - p_yx[pi]) ** 2, axis=1))
            best  = int(np.argmin(dists))
            if dists[best] <= MATCH_RADIUS_PX and best not in used_gt:
                used_gt.add(best)
                gt_row = frame_gt.iloc[best]
                preds.at[p_idx, "matched"]    = True
                preds.at[p_idx, "true_label"] = int(gt_row["oligomer_label"])

    return preds


# ── evaluation harness (frozen) ───────────────────────────────────────────────

def evaluate(pipeline, verbose: bool = True) -> dict:
    """Run pipeline on all test movies and compute classification metrics.

    Parameters
    ----------
    pipeline:
        Object with method predict(movie, noise_level) -> pd.DataFrame
        Required columns in prediction: frame (int), y, x, predicted_label (int)
        predicted_label: 0=noise, 1=monomer, ..., 6=hexamer

    Returns
    -------
    dict with keys: monomer_recall, fdr, macro_f1, accuracy,
                    per_class_recall, per_class_precision, per_class_f1
    """
    test_data = load_dataset("test")

    # Accumulators: per class (0=noise, 1–6=species)
    tp = np.zeros(N_CLASSES, dtype=int)
    fp = np.zeros(N_CLASSES, dtype=int)
    fn = np.zeros(N_CLASSES, dtype=int)
    n_ghost = 0      # detections with no GT match (regardless of predicted label)
    n_total_det = 0  # total detections

    for entry in test_data:
        movie, gt_df, noise_level = entry["movie"], entry["gt"], entry["noise"]

        preds_df = pipeline.predict(movie, noise_level)
        if preds_df is None or len(preds_df) == 0:
            # All GT peaks are FN
            for label in gt_df["oligomer_label"].unique():
                fn[int(label)] += int((gt_df["oligomer_label"] == label).sum())
            continue

        matched = match_to_gt(preds_df, gt_df)
        n_total_det += len(matched)
        n_ghost     += int((~matched["matched"]).sum())

        for _, row in matched.iterrows():
            pred = int(row["predicted_label"])
            true = int(row["true_label"])   # 0 if ghost

            if not row["matched"]:
                # Ghost peak: FP for whatever class the model predicted
                fp[pred] += 1
            else:
                if pred == true:
                    tp[pred] += 1
                else:
                    fp[pred] += 1
                    fn[true] += 1

        # FN: GT peaks with no matching detection
        all_matched_gt_frames = set(zip(
            matched.loc[matched["matched"], "frame"].values,
            matched.loc[matched["matched"], "y"].values,
            matched.loc[matched["matched"], "x"].values,
        ))
        for _, gt_row in gt_df.iterrows():
            key = (int(gt_row["frame"]), int(gt_row["y"]), int(gt_row["x"]))
            if key not in all_matched_gt_frames:
                fn[int(gt_row["oligomer_label"])] += 1

    # Per-class metrics (skip class 0 = noise for recall/precision)
    per_class_recall    = {}
    per_class_precision = {}
    per_class_f1        = {}
    for c in range(1, N_CLASSES):
        name = CLASS_NAMES[c]
        denom_rec  = tp[c] + fn[c]
        denom_prec = tp[c] + fp[c]
        rec  = tp[c] / denom_rec  if denom_rec  > 0 else 0.0
        prec = tp[c] / denom_prec if denom_prec > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_recall[name]    = rec
        per_class_precision[name] = prec
        per_class_f1[name]        = f1

    monomer_recall = per_class_recall.get("monomer", 0.0)
    fdr            = n_ghost / n_total_det if n_total_det > 0 else 0.0
    macro_f1       = float(np.mean(list(per_class_f1.values())))
    total_tp       = int(sum(tp[1:]))
    total_matched  = total_tp + int(sum(fp[1:])) + int(sum(fn[1:]))
    accuracy       = total_tp / (n_total_det - n_ghost) if (n_total_det - n_ghost) > 0 else 0.0

    metrics = dict(
        monomer_recall      = monomer_recall,
        fdr                 = fdr,
        macro_f1            = macro_f1,
        accuracy            = accuracy,
        per_class_recall    = per_class_recall,
        per_class_precision = per_class_precision,
        per_class_f1        = per_class_f1,
        n_total_det         = n_total_det,
        n_ghost             = n_ghost,
    )

    if verbose:
        print("\n" + "="*50)
        print("Evaluation results")
        print("="*50)
        print(f"{'Class':<12} {'Recall':>8} {'Prec':>8} {'F1':>8}")
        print("-"*40)
        for c in range(1, N_CLASSES):
            name = CLASS_NAMES[c]
            marker = " ◄" if name == "monomer" else ""
            print(f"{name:<12} {per_class_recall[name]:8.4f} "
                  f"{per_class_precision[name]:8.4f} "
                  f"{per_class_f1[name]:8.4f}{marker}")
        print("-"*40)
        print(f"{'detections':<12} {n_total_det:>8d}")
        print(f"{'ghost peaks':<12} {n_ghost:>8d}")
        print()
        # Machine-parseable summary lines (read by the experiment loop)
        print(f"monomer_recall: {monomer_recall:.6f}")
        print(f"fdr:            {fdr:.6f}")
        print(f"macro_f1:       {macro_f1:.6f}")
        print(f"accuracy:       {accuracy:.6f}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MP autoresearch data generation")
    parser.add_argument("--check", action="store_true",
                        help="Verify cache exists without regenerating")
    args = parser.parse_args()

    if args.check:
        missing = []
        for split in ("train", "test"):
            for sp in SIM_SPECIES:
                for suffix in ("_movie.npy", "_gt.pkl", "_noise.npy"):
                    p = os.path.join(CACHE_DIR, split, sp["label"] + suffix)
                    if not os.path.exists(p):
                        missing.append(p)
        if missing:
            print(f"MISSING {len(missing)} files — run: python prepare.py")
            for p in missing[:5]:
                print(f"  {p}")
        else:
            print(f"Cache OK: {CACHE_DIR}")
        return

    generate()


if __name__ == "__main__":
    main()
