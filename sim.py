"""sim.py — standalone simulation library for MP autoresearch.

Extracted from mp_analysis/simulation.py and mp_analysis/attributes.py.
Frozen: do not modify.
"""

from __future__ import annotations
from types import SimpleNamespace
from typing import Literal

import json
import numpy as np
import scipy.special
from scipy.ndimage import gaussian_filter as _gf
from scipy.stats import median_abs_deviation as _mad


# ---------------------------------------------------------------------------
# PSF constants  (from mp_analysis/attributes.py)
# ---------------------------------------------------------------------------

class PSFConstants:
    """Instrument-specific constants for the iSCAT jinc PSF.

    PSF(r) = c × [a1·jinc(r, w1) + a2·jinc(r, w2)] × exp(−r²/2s²)

    Defaults calibrated for Refeyn TwoMP.
    """

    def __init__(
        self,
        a12: float = -6.04,
        w: float = 2.12,
        s: float = 4.0,
        R_aperture: float = 8.52e-3,
        R_mask: float = 1.5 * 2.5e-3,
    ) -> None:
        self.a12 = a12
        self.w   = w
        self.s   = s
        eps       = np.finfo(np.float64).eps
        self.res  = np.finfo(np.float32).resolution
        self.a1   = 1.0 / ((a12 + 1) / a12 + eps)
        self.a2   = 1.0 / ((a12 + 1) + eps)
        self.w1   = w
        self.w2   = w * R_aperture / R_mask


PSF_CONSTANTS = PSFConstants()


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _apply_periodic(pos: np.ndarray, box_size: float) -> np.ndarray:
    np.mod(pos, box_size, out=pos)
    return pos


def _apply_reflect(pos: np.ndarray, box_size: float) -> np.ndarray:
    np.mod(pos, 2.0 * box_size, out=pos)
    mask = pos > box_size
    pos[mask] = 2.0 * box_size - pos[mask]
    return pos


# ---------------------------------------------------------------------------
# Brownian simulation
# ---------------------------------------------------------------------------

class BrownianSimulation:
    """2D Brownian motion simulator matched to an iSCAT movie.

    Particles diffuse in a simulation box 2× the FOV in each dimension.
    The FOV occupies the central quarter; particles wrap (or reflect) at
    the simulation box boundary.

    Parameters
    ----------
    n_particles:
        Total particles in simulation box; expected visible in FOV ≈ n/4.
    n_frames:
        Number of time steps.
    pixel_size_nm:
        Effective pixel size in nm/pixel (Refeyn TwoMP default: 84.4).
    fov_shape:
        (H, W) in pixels.
    diffusion_coeff:
        Single D (µm²/s); ignored when *species* is given.
    species:
        Multi-species dict: {label: (fraction, D_µm2s)} or
        {label: (fraction, D_µm2s, mw_kda)}.
    frame_rate:
        Camera frame rate in Hz.
    dt:
        Time step override (seconds); defaults to 1/frame_rate.
    boundary:
        "periodic" (default) or "reflect".
    seed:
        RNG seed.
    """

    def __init__(
        self,
        n_particles: int,
        n_frames: int,
        pixel_size_nm: float = 84.4,
        fov_shape: tuple[int, int] = (59, 150),
        diffusion_coeff: float = 3.5,
        species: dict | None = None,
        frame_rate: float = 475.0,
        dt: float | None = None,
        boundary: Literal["periodic", "reflect"] = "periodic",
        seed: int | None = None,
    ) -> None:
        self.n_particles    = int(n_particles)
        self.n_frames       = int(n_frames)
        self.pixel_size_nm  = float(pixel_size_nm)
        self.fov_h, self.fov_w = int(fov_shape[0]), int(fov_shape[1])
        self.frame_rate     = float(frame_rate)
        self.dt             = float(dt) if dt is not None else 1.0 / frame_rate
        self.boundary       = boundary
        self.seed           = seed
        self.box_h          = 2 * self.fov_h
        self.box_w          = 2 * self.fov_w
        self.fov_x0         = self.fov_w / 2.0
        self.fov_y0         = self.fov_h / 2.0

        self.trajectories    : np.ndarray | None = None
        self.visibility      : np.ndarray | None = None
        self.species_ids     : np.ndarray | None = None
        self.mw_per_particle : np.ndarray | None = None
        self.species_mw_kda  : dict = {}

        self.species_labels, self._sigma_per_particle = self._build_species(
            species, diffusion_coeff
        )

    def _build_species(self, species, single_D):
        px_um = self.pixel_size_nm / 1000.0
        if species is None:
            D_px = single_D / px_um ** 2
            sigma = float(np.sqrt(2.0 * D_px * self.dt))
            self.species_ids     = np.zeros(self.n_particles, dtype=np.int32)
            self.species_mw_kda  = {"all": float("nan")}
            self.mw_per_particle = np.full(self.n_particles, float("nan"), dtype=np.float32)
            return ["all"], np.full(self.n_particles, sigma, dtype=np.float32)

        names   = list(species.keys())
        entries = [species[k] for k in names]
        fracs   = np.array([e[0] for e in entries], dtype=float)
        fracs  /= fracs.sum()
        D_vals  = np.array([e[1] for e in entries], dtype=float)
        mw_vals = np.array([e[2] if len(e) > 2 else float("nan")
                            for e in entries], dtype=np.float32)

        counts = np.round(fracs * self.n_particles).astype(int)
        counts[np.argmax(fracs)] += self.n_particles - counts.sum()
        sigma_vals = np.sqrt(2.0 * D_vals / px_um ** 2 * self.dt).astype(np.float32)
        ids    = np.repeat(np.arange(len(names)), counts).astype(np.int32)
        sigmas = sigma_vals[ids]

        self.species_ids     = ids
        self.species_mw_kda  = {n: float(mw_vals[i]) for i, n in enumerate(names)}
        self.mw_per_particle = mw_vals[ids]
        return names, sigmas

    @property
    def mean_fov_occupancy(self) -> float:
        return self.n_particles / 4.0

    def run(self) -> "BrownianSimulation":
        rng = np.random.default_rng(self.seed)
        x = rng.uniform(0.0, self.box_w, self.n_particles).astype(np.float32)
        y = rng.uniform(0.0, self.box_h, self.n_particles).astype(np.float32)
        traj = np.empty((self.n_particles, self.n_frames, 2), dtype=np.float32)
        bc   = _apply_periodic if self.boundary == "periodic" else _apply_reflect

        for t in range(self.n_frames):
            traj[:, t, 0] = x
            traj[:, t, 1] = y
            x = x + (rng.standard_normal(self.n_particles) * self._sigma_per_particle).astype(np.float32)
            y = y + (rng.standard_normal(self.n_particles) * self._sigma_per_particle).astype(np.float32)
            bc(x, float(self.box_w))
            bc(y, float(self.box_h))

        self.trajectories = traj
        self.visibility   = self._compute_visibility(traj)
        return self

    def _compute_visibility(self, traj):
        x = traj[:, :, 0]
        y = traj[:, :, 1]
        in_x = (x >= self.fov_x0) & (x < self.fov_x0 + self.fov_w)
        in_y = (y >= self.fov_y0) & (y < self.fov_y0 + self.fov_h)
        return in_x & in_y

    @property
    def fov_positions(self) -> np.ndarray:
        if self.trajectories is None:
            raise RuntimeError("Call run() first.")
        pos = self.trajectories.copy()
        pos[:, :, 0] -= self.fov_x0
        pos[:, :, 1] -= self.fov_y0
        pos[~self.visibility] = np.nan
        return pos


def simulate_brownian(
    n_particles: int,
    n_frames: int,
    pixel_size_nm: float = 84.4,
    fov_shape: tuple[int, int] = (59, 150),
    diffusion_coeff: float = 3.5,
    species: dict | None = None,
    frame_rate: float = 475.0,
    dt: float | None = None,
    boundary: Literal["periodic", "reflect"] = "periodic",
    seed: int | None = None,
) -> BrownianSimulation:
    """Create and run a BrownianSimulation, returning the result."""
    return BrownianSimulation(
        n_particles=n_particles, n_frames=n_frames,
        pixel_size_nm=pixel_size_nm, fov_shape=fov_shape,
        diffusion_coeff=diffusion_coeff, species=species,
        frame_rate=frame_rate, dt=dt, boundary=boundary, seed=seed,
    ).run()


# ---------------------------------------------------------------------------
# PSF model — physically-motivated jinc (Airy) form
# ---------------------------------------------------------------------------

class MPPSF:
    """iSCAT PSF: weighted sum of two jinc functions with Gaussian envelope.

    PSF(r) = c × [a1·jinc(r, w1) + a2·jinc(r, w2)] × exp(−r²/2s²)
    """

    def __init__(self, psfc: PSFConstants | None = None, clip_radius: int = 15) -> None:
        self.psfc        = psfc if psfc is not None else PSF_CONSTANTS
        self.clip_radius = int(clip_radius)

    def render(self, frame: np.ndarray, y0: float, x0: float, contrast: float) -> None:
        H, W = frame.shape
        r    = self.clip_radius
        iy, ix = int(round(y0)), int(round(x0))
        y_lo = max(0, iy - r);  y_hi = min(H, iy + r + 1)
        x_lo = max(0, ix - r);  x_hi = min(W, ix + r + 1)
        ys = np.arange(y_lo, y_hi, dtype=np.float64) - y0
        xs = np.arange(x_lo, x_hi, dtype=np.float64) - x0
        xx, yy = np.meshgrid(xs, ys)
        rr   = np.sqrt(xx ** 2 + yy ** 2)
        psfc = self.psfc
        res  = psfc.res
        xw1  = np.where(rr > res, rr * np.pi / psfc.w1, res)
        xw2  = np.where(rr > res, rr * np.pi / psfc.w2, res)
        jinc1 = 2.0 * scipy.special.j1(xw1) / xw1
        jinc2 = 2.0 * scipy.special.j1(xw2) / xw2
        env   = np.exp(-0.5 * (rr / psfc.s) ** 2)
        frame[y_lo:y_hi, x_lo:x_hi] += (
            contrast * (psfc.a1 * jinc1 + psfc.a2 * jinc2) * env
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def select_clean_noise_frames(
    frames: np.ndarray,
    window: int = 1000,
    mad_factor: float = 5.0,
) -> np.ndarray:
    """Boolean mask selecting frames free of large contamination events.

    Frames whose per-frame MAD exceeds mad_factor × local-median MAD are
    flagged as contaminated and excluded.
    """
    T   = len(frames)
    mad = np.array([_mad(f.ravel()) for f in frames], dtype=np.float64)
    half = window // 2
    local_median = np.empty(T, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - half); hi = min(T, t + half + 1)
        local_median[t] = np.median(mad[lo:hi])
    local_median = np.where(local_median > 0, local_median, np.median(mad))
    mask = mad <= mad_factor * local_median
    print(f"  select_clean_noise_frames: {mask.sum()}/{T} clean "
          f"({T - mask.sum()} excluded)")
    return mask


class ExperimentalNoise:
    """Noise sampled by resampling frames from an empty-bilayer contrast movie."""

    def __init__(self, noise_frames: np.ndarray) -> None:
        if noise_frames.ndim != 3:
            raise ValueError("noise_frames must be shape (T, H, W)")
        self.noise_frames = noise_frames.astype(np.float32)
        self.std = float(noise_frames.std())

    def generate(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        T, H, W = shape
        T_noise  = len(self.noise_frames)
        idx      = rng.integers(0, T_noise, size=T)
        sampled  = self.noise_frames[idx]
        if sampled.shape[1:] != (H, W):
            raise ValueError(f"Noise shape {sampled.shape[1:]} ≠ ({H},{W})")
        return sampled

    @classmethod
    def from_tiff(cls, path: str) -> "ExperimentalNoise":
        import tifffile
        frames = tifffile.imread(str(path)).astype(np.float32)
        if frames.ndim == 2:
            frames = frames[np.newaxis]
        return cls(frames)

    def __repr__(self) -> str:
        T, H, W = self.noise_frames.shape
        return f"ExperimentalNoise(T={T}, H={H}, W={W}, std={self.std:.5f})"


# ---------------------------------------------------------------------------
# Synthetic movie generation
# ---------------------------------------------------------------------------

def generate_contrast_movie(
    sim: BrownianSimulation,
    psf: MPPSF,
    noise_model: ExperimentalNoise | None = None,
    amplitudes: float | dict | None = None,
    calibration=None,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Render a synthetic iSCAT contrast movie from a BrownianSimulation.

    Parameters
    ----------
    sim:
        Completed BrownianSimulation.
    psf:
        MPPSF instance.
    noise_model:
        ExperimentalNoise instance, or None for noise-free.
    amplitudes:
        Explicit contrast: float (all particles) or dict {label: contrast}.
        Takes precedence over calibration.
    calibration:
        Object with .gradient and .offset; derives contrast from MW.
    noise_seed:
        RNG seed for noise sampling.

    Returns
    -------
    np.ndarray, shape (T, H, W), dtype float32.
    """
    H, W = sim.fov_h, sim.fov_w
    T, N = sim.n_frames, sim.n_particles
    fov_pos = sim.fov_positions

    amp_arr = np.empty(N, dtype=np.float32)
    if amplitudes is not None:
        if isinstance(amplitudes, (int, float)):
            amp_arr[:] = float(amplitudes)
        else:
            for p in range(N):
                sid = int(sim.species_ids[p]) if sim.species_ids is not None else 0
                amp_arr[p] = amplitudes.get(sim.species_labels[sid], float("nan"))
    elif calibration is not None:
        for p in range(N):
            mw = float(sim.mw_per_particle[p])
            if not np.isfinite(mw):
                raise ValueError(f"Particle {p} has no MW — supply MW in species tuple.")
            amp_arr[p] = calibration.gradient * mw + calibration.offset
    else:
        raise ValueError("Supply 'amplitudes' or 'calibration'.")

    movie = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        if t % 2000 == 0 and t > 0:
            print(f"    rendering frame {t}/{T}")
        frame   = movie[t]
        vis_idx = np.where(sim.visibility[:, t])[0]
        for p in vis_idx:
            y0 = float(fov_pos[p, t, 1])
            x0 = float(fov_pos[p, t, 0])
            psf.render(frame, y0, x0, float(amp_arr[p]))

    if noise_model is not None:
        rng    = np.random.default_rng(noise_seed)
        movie += noise_model.generate((T, H, W), rng)

    return movie


# ---------------------------------------------------------------------------
# Ground-truth DataFrames
# ---------------------------------------------------------------------------

def trajectories_to_dataframe(
    sim: BrownianSimulation,
    amplitudes: float | dict | None = None,
    calibration=None,
) -> "pd.DataFrame":
    """Convert completed simulation into a ground-truth position DataFrame.

    One row per (particle, frame) when visible in the FOV.
    Columns: frame, particle_id, species_label, x, y, x_sub, y_sub,
             true_mw_kda, true_contrast.
    """
    import pandas as pd

    fov_pos = sim.fov_positions
    N = sim.n_particles

    amp_arr = np.empty(N, dtype=np.float32)
    mw_arr  = (sim.mw_per_particle.copy() if sim.mw_per_particle is not None
               else np.full(N, float("nan"), dtype=np.float32))

    if amplitudes is not None:
        if isinstance(amplitudes, (int, float)):
            amp_arr[:] = float(amplitudes)
        else:
            for p in range(N):
                sid = int(sim.species_ids[p]) if sim.species_ids is not None else 0
                amp_arr[p] = amplitudes.get(sim.species_labels[sid], float("nan"))
    elif calibration is not None:
        for p in range(N):
            mw = float(mw_arr[p])
            amp_arr[p] = (calibration.gradient * mw + calibration.offset
                          if np.isfinite(mw) else float("nan"))
    else:
        amp_arr[:] = float("nan")

    rows = []
    for t in range(sim.n_frames):
        for p in np.where(sim.visibility[:, t])[0]:
            x_sub = float(fov_pos[p, t, 0])
            y_sub = float(fov_pos[p, t, 1])
            sid   = int(sim.species_ids[p]) if sim.species_ids is not None else 0
            rows.append(dict(
                frame         = int(t),
                particle_id   = int(p),
                species_label = sim.species_labels[sid],
                x             = int(round(x_sub)),
                y             = int(round(y_sub)),
                x_sub         = round(x_sub, 4),
                y_sub         = round(y_sub, 4),
                true_mw_kda   = round(float(mw_arr[p]), 2),
                true_contrast = round(float(amp_arr[p]), 6),
            ))
    return pd.DataFrame(rows)


def sim_ground_truth_picks(
    sim: BrownianSimulation,
    movie: np.ndarray,
    roi_half: int,
    fov_h: int,
    fov_w: int,
    dog_min_sigma: float = 1.0,
    dog_max_sigma: float = 1.6,
    dog_sigma_ratio: float = 1.6,
) -> "pd.DataFrame":
    """Ground-truth peaks at exact particle positions with DoG-consistent features.

    Places one row per (particle, frame) at the exact GT position.
    Dog response, peak_sum, local_noise, and SNR are computed with the same
    formulas used by the DoG peak picker so feature distributions match.
    """
    import pandas as pd

    fov_pos = sim.fov_positions
    sf = 1.0 / (dog_sigma_ratio - 1)
    _cached_t   = -1
    _cached_inv = None
    _cached_dog = None

    rows = []
    for t in range(sim.n_frames):
        visible = np.where(sim.visibility[:, t])[0]
        if len(visible) == 0:
            continue
        if t != _cached_t:
            _cached_inv = (-movie[t]).astype(np.float64)
            _cached_dog = (_gf(_cached_inv, dog_min_sigma) -
                           _gf(_cached_inv, dog_max_sigma)) * sf
            _cached_t = t
        inv = _cached_inv
        dog = _cached_dog

        for p in visible:
            x_f = float(fov_pos[p, t, 0])
            y_f = float(fov_pos[p, t, 1])
            yi  = int(round(y_f))
            xi  = int(round(x_f))
            if not (roi_half <= yi < fov_h - roi_half and
                    roi_half <= xi < fov_w - roi_half):
                continue

            pixel_contrast = float(movie[t, yi, xi])
            peak_int       = float(inv[yi, xi])
            dog_resp       = float(dog[yi, xi])

            r   = 4
            box = inv[max(0, yi - r): yi + r + 1,
                      max(0, xi - r): xi + r + 1]
            psum = float(np.maximum(box, 0.0).sum())

            r_in, r_out = 6, 10
            ys = np.arange(max(0, yi - r_out), min(fov_h, yi + r_out + 1))
            xs = np.arange(max(0, xi - r_out), min(fov_w, xi + r_out + 1))
            yg, xg = np.meshgrid(ys, xs, indexing="ij")
            dist_sq = (yg - yi) ** 2 + (xg - xi) ** 2
            annulus = inv[yg, xg][(dist_sq >= r_in**2) & (dist_sq <= r_out**2)]
            if len(annulus) >= 4:
                local_noise_val = float(1.4826 * np.median(
                    np.abs(annulus - np.median(annulus))))
            else:
                local_noise_val = float("nan")
            snr_val = (peak_int / local_noise_val
                       if (np.isfinite(local_noise_val) and local_noise_val > 0)
                       else float("nan"))

            rows.append(dict(
                frame          = int(t),
                y              = yi,
                x              = xi,
                y_sub          = round(y_f, 4),
                x_sub          = round(x_f, 4),
                contrast       = round(pixel_contrast, 6),
                peak_intensity = round(peak_int, 6),
                dog_response   = round(dog_resp, 6),
                peak_sum       = round(psum, 6),
                local_noise    = round(local_noise_val, 6),
                snr            = round(snr_val, 3),
                particle_id    = int(p),
            ))

    df = pd.DataFrame(rows)
    n_vis  = int(sim.visibility.sum())
    n_edge = n_vis - len(df)
    print(f"  sim_ground_truth_picks: {len(df):,} observations "
          f"({sim.n_frames} frames, {n_vis} vis particle-frames, "
          f"{n_edge} edge-excluded)")
    return df


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(mc_file: str) -> SimpleNamespace:
    """Load a Refeyn .mc calibration file.

    Returns SimpleNamespace(gradient, offset) where
        contrast = gradient × mass_kDa + offset.
    """
    with open(mc_file) as f:
        data = json.load(f)
    if "gradient" not in data or "offset" not in data:
        raise KeyError(f"Missing 'gradient' or 'offset' in {mc_file}")
    calib = SimpleNamespace(
        gradient = float(data["gradient"]),
        offset   = float(data["offset"]),
        file     = str(mc_file),
    )
    print(f"Calibration: gradient={calib.gradient:.4e}  offset={calib.offset:.4e}")
    return calib
