"""train.py — classification pipeline.

This is the file the agent modifies.  Everything is fair game:
  - Peak detection method and parameters
  - Feature engineering
  - Classifier architecture and hyperparameters

Interface contract (frozen — test.py depends on this):
  Pipeline.predict(movie, noise_level) -> pd.DataFrame
    Required output columns: frame (int), y (int), x (int), predicted_label (int)
    predicted_label: 0=noise, 1=monomer, 2=dimer, ..., 6=hexamer
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter as _gf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from prepare import (
    load_dataset, match_to_gt,
    CACHE_DIR, N_CLASSES, CLASS_NAMES, FOV_H, FOV_W, ROI_HALF,
)

# ── peak detection parameters ─────────────────────────────────────────────────
DOG_SIGMA_SMALL     = 1.0
DOG_SIGMA_LARGE     = 1.6
DOG_SIGMA_RATIO     = 1.6    # normalization factor
THRESHOLD_FACTOR    = 3.0    # multiples of noise_level
MIN_PEAK_DISTANCE   = 3      # minimum pixel separation between peaks

# ── classifier parameters ─────────────────────────────────────────────────────
N_ESTIMATORS   = 200
MAX_DEPTH      = None
CLASS_WEIGHT   = "balanced"  # compensate for class imbalance
FEATURE_COLS   = [
    "contrast",
    "peak_intensity",
    "dog_response",
    "peak_sum",
    "snr",
    "local_noise",
]


# ── peak detection ────────────────────────────────────────────────────────────

def pick_peaks(movie: np.ndarray, noise_level: float) -> pd.DataFrame:
    """DoG blob detection on a contrast movie.

    Parameters
    ----------
    movie:
        (T, H, W) float32 contrast array. Dark spots have negative values.
    noise_level:
        MAD-based noise estimate; threshold = THRESHOLD_FACTOR × noise_level.

    Returns
    -------
    pd.DataFrame with columns:
        frame, y, x, contrast, peak_intensity, dog_response,
        peak_sum, snr, local_noise
    """
    threshold = THRESHOLD_FACTOR * noise_level
    sf        = 1.0 / (DOG_SIGMA_RATIO - 1)
    T, H, W   = movie.shape
    rows      = []

    for t in range(T):
        inv = (-movie[t]).astype(np.float64)
        dog = (_gf(inv, DOG_SIGMA_SMALL) - _gf(inv, DOG_SIGMA_LARGE)) * sf

        # Find local maxima above threshold
        candidates = _find_local_maxima(dog, threshold, MIN_PEAK_DISTANCE)

        for yi, xi in candidates:
            if not (ROI_HALF <= yi < H - ROI_HALF and
                    ROI_HALF <= xi < W - ROI_HALF):
                continue

            pixel_contrast = float(movie[t, yi, xi])
            peak_int       = float(inv[yi, xi])
            dog_resp       = float(dog[yi, xi])

            r   = 4
            box = inv[max(0, yi - r): yi + r + 1,
                      max(0, xi - r): xi + r + 1]
            psum = float(np.maximum(box, 0.0).sum())

            r_in, r_out = 6, 10
            ys = np.arange(max(0, yi - r_out), min(H, yi + r_out + 1))
            xs = np.arange(max(0, xi - r_out), min(W, xi + r_out + 1))
            yg, xg = np.meshgrid(ys, xs, indexing="ij")
            dist_sq = (yg - yi) ** 2 + (xg - xi) ** 2
            annulus = inv[yg, xg][(dist_sq >= r_in**2) & (dist_sq <= r_out**2)]
            if len(annulus) >= 4:
                lnoise = float(1.4826 * np.median(
                    np.abs(annulus - np.median(annulus))))
            else:
                lnoise = float("nan")
            snr_val = (peak_int / lnoise
                       if (np.isfinite(lnoise) and lnoise > 0)
                       else float("nan"))

            rows.append(dict(
                frame          = int(t),
                y              = yi,
                x              = xi,
                contrast       = round(pixel_contrast, 6),
                peak_intensity = round(peak_int, 6),
                dog_response   = round(dog_resp, 6),
                peak_sum       = round(psum, 6),
                local_noise    = round(lnoise, 6),
                snr            = round(snr_val, 3),
            ))

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["frame", "y", "x", "contrast", "peak_intensity",
                 "dog_response", "peak_sum", "local_noise", "snr"])


def _find_local_maxima(
    img: np.ndarray,
    threshold: float,
    min_distance: int,
) -> list[tuple[int, int]]:
    """Return (y, x) positions of local maxima above threshold."""
    from scipy.ndimage import maximum_filter, label

    mask    = (img >= threshold) & (img == maximum_filter(img, size=min_distance * 2 + 1))
    labeled, n = label(mask)
    peaks   = []
    for obj_id in range(1, n + 1):
        ys, xs = np.where(labeled == obj_id)
        best   = np.argmax(img[ys, xs])
        peaks.append((int(ys[best]), int(xs[best])))
    return peaks


# ── pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Full detection + classification pipeline."""

    def __init__(self) -> None:
        self.clf     : RandomForestClassifier | None = None
        self.scaler  : StandardScaler | None = None

    def fit(self, train_data: list[dict]) -> None:
        """Train classifier on all training movies.

        Parameters
        ----------
        train_data:
            Output of prepare.load_dataset("train").
        """
        all_features = []
        all_labels   = []

        for entry in train_data:
            movie, gt_df, noise = entry["movie"], entry["gt"], entry["noise"]
            print(f"  [{entry['species']}] picking peaks …", flush=True)

            peaks_df = pick_peaks(movie, noise)
            if len(peaks_df) == 0:
                print(f"    WARNING: no peaks detected")
                continue

            labeled = match_to_gt(peaks_df, gt_df)
            feats   = labeled[FEATURE_COLS].values
            labels  = labeled["true_label"].values

            # Drop rows with NaN features
            valid = np.all(np.isfinite(feats), axis=1)
            all_features.append(feats[valid])
            all_labels.append(labels[valid])

            n_total  = int(valid.sum())
            n_noise  = int((labels[valid] == 0).sum())
            n_real   = n_total - n_noise
            print(f"    {n_total:,} peaks  ({n_real} real, {n_noise} noise)")

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        print(f"\nTraining RF on {len(X):,} peaks …", flush=True)

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.clf = RandomForestClassifier(
            n_estimators = N_ESTIMATORS,
            max_depth    = MAX_DEPTH,
            class_weight = CLASS_WEIGHT,
            n_jobs       = -1,
            random_state = 0,
        )
        self.clf.fit(X_scaled, y)
        print(f"Training complete.")

    def predict(self, movie: np.ndarray, noise_level: float) -> pd.DataFrame:
        """Run the full pipeline on one movie.

        Returns DataFrame with columns:
            frame, y, x, predicted_label
        """
        peaks_df = pick_peaks(movie, noise_level)
        if len(peaks_df) == 0:
            return pd.DataFrame(columns=["frame", "y", "x", "predicted_label"])

        feats = peaks_df[FEATURE_COLS].values
        valid = np.all(np.isfinite(feats), axis=1)

        predictions = np.zeros(len(peaks_df), dtype=int)
        if valid.any():
            X_scaled = self.scaler.transform(feats[valid])
            predictions[valid] = self.clf.predict(X_scaled)

        result = peaks_df[["frame", "y", "x"]].copy()
        result["predicted_label"] = predictions
        return result


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading training data …")
    train_data = load_dataset("train")

    pipeline = Pipeline()
    pipeline.fit(train_data)

    out_path = os.path.join(CACHE_DIR, "pipeline.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved → {out_path}")
