"""Paper §2.4 generative model (source-faithful reconstruction with documented gaps).

Does not implement student-lab synthetic benchmarks. There is no AIRI simulation
script; this module is a narrow reconstruction of the published recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfiltfilt

from .config import SimulationConfig
from .forward_model import ForwardModel


def squared_euclidean_rdm(series: NDArray[np.floating]) -> NDArray[np.float64]:
    """Eq. 1: d_ij = ||s^i - s^j||^2 on rows of ``series`` (C, T)."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"series must be (C, T), got {series.shape}")
    gram = series @ series.T
    energy = np.diag(gram)
    rdm = energy[:, None] + energy[None, :] - 2.0 * gram
    np.fill_diagonal(rdm, 0.0)
    return np.maximum(rdm, 0.0)


def add_symmetric_rdm_noise(
    d0: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    relative_std: float,
) -> NDArray[np.float64]:
    """Assumed Υ_d: small symmetric Gaussian on unique pairs, clip, zero diag."""
    d0 = np.asarray(d0, dtype=np.float64)
    n_cond = d0.shape[0]
    if d0.shape != (n_cond, n_cond):
        raise ValueError("D0 must be square")
    ii, jj = np.triu_indices(n_cond, k=1)
    upper = d0[ii, jj]
    scale = relative_std * float(np.std(upper, ddof=1)) if upper.size > 1 else relative_std
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(relative_std)
    noise = rng.standard_normal(upper.size) * scale
    values = np.maximum(upper + noise, 0.0)
    noisy = np.zeros((n_cond, n_cond), dtype=np.float64)
    noisy[ii, jj] = values
    noisy[jj, ii] = values
    return noisy


def filter_gaussian_rows(
    rng: np.random.Generator,
    n_conditions: int,
    n_times: int,
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
    zero_phase: bool = True,
) -> NDArray[np.float64]:
    """Z: Gaussian rows, 6th-order Butterworth low-pass."""
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    raw = rng.standard_normal((n_conditions, n_times))
    if zero_phase:
        return np.asarray(sosfiltfilt(sos, raw, axis=-1), dtype=np.float64)
    return np.asarray(sosfilt(sos, raw, axis=-1), dtype=np.float64)


def source_erp(
    mixing: NDArray[np.floating],
    rng: np.random.Generator,
    n_times: int,
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
    zero_phase: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """S = M Z. Returns S, Z each (C, T)."""
    mixing = np.asarray(mixing, dtype=np.float64)
    n_conditions = mixing.shape[0]
    if mixing.shape != (n_conditions, n_conditions):
        raise ValueError(f"M must be C×C, got {mixing.shape}")
    z = filter_gaussian_rows(
        rng,
        n_conditions,
        n_times,
        fs_hz=fs_hz,
        cutoff_hz=cutoff_hz,
        order=order,
        zero_phase=zero_phase,
    )
    return mixing @ z, z


def fft_pink_noise(
    rng: np.random.Generator,
    n_signals: int,
    n_times: int,
    *,
    fs_hz: float,
    exponent: float = 1.0,
) -> NDArray[np.float64]:
    """Unit-RMS FFT 1/f noise. PSD ∝ f^{-exponent}. DC bin zeroed.

    Assumed reconstruction of '1/f noise' (paper cites Ossadtchi et al. 2018).
    """
    white = rng.standard_normal((n_signals, n_times))
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs_hz)
    scale = np.zeros_like(freqs)
    positive = freqs > 0.0
    scale[positive] = freqs[positive] ** (-0.5 * exponent)
    pink = np.fft.irfft(spec * scale, n=n_times, axis=-1)
    rms = np.sqrt(np.mean(pink * pink, axis=-1, keepdims=True))
    return pink / np.maximum(rms, 1e-12)


def cortical_one_over_f_sensor_noise(
    gain: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    n_epochs: int,
    n_times: int,
    n_sources: int,
    fs_hz: float,
    exponent: float,
    loci_mode: str = "per_epoch",
) -> NDArray[np.float64]:
    """1000 randomly seeded cortical 1/f sources, projected, shape (n_epochs, N, T).

    ``per_epoch`` (original SIM-P1): redraw locations every epoch.
    ``fixed`` (review forensic): seed 1000 locations once; only 1/f series
    change across epochs. Not claimed as paper-faithful.
    """
    gain = np.asarray(gain, dtype=np.float64)
    n_channels, n_vertices = gain.shape
    if loci_mode == "per_epoch":
        index = rng.integers(0, n_vertices, size=(n_epochs, n_sources))
    elif loci_mode == "fixed":
        fixed = rng.integers(0, n_vertices, size=n_sources)
        index = np.broadcast_to(fixed, (n_epochs, n_sources)).copy()
    else:
        raise ValueError(f"Unknown noise loci mode {loci_mode!r}")
    pink = fft_pink_noise(
        rng, n_epochs * n_sources, n_times, fs_hz=fs_hz, exponent=exponent
    ).reshape(n_epochs, n_sources, n_times)
    selected = gain[:, index]  # (N, n_epochs, n_sources)
    return np.einsum("nes,est->ent", selected, pink, optimize=True)


def rms(array: NDArray[np.floating]) -> float:
    array = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def scale_noise_to_snr(
    signal: NDArray[np.floating],
    noise: NDArray[np.floating],
    snr: float,
) -> tuple[NDArray[np.float64], float]:
    """Return gamma * noise and gamma so RMS(signal)/RMS(gamma noise) = snr."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if snr <= 0.0:
        raise ValueError("SNR must be positive")
    noise_rms = rms(noise)
    if noise_rms == 0.0:
        raise ValueError("noise matrix has zero RMS")
    gamma = rms(signal) / (float(snr) * noise_rms)
    return gamma * noise, float(gamma)


def topography_with_forward_error(
    g: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    sigma_rel: float,
    delta_mode: str = "literal_covariance",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """g + δ.

    ``literal_covariance``: δ ~ N(0, (σ||g||)^2 I), the printed paper formula.
    ``norm_15pct``: same isotropic direction, rescaled so ||δ|| = σ||g||.
    Both consume one standard-normal draw of length N.
    """
    g = np.asarray(g, dtype=np.float64).ravel()
    direction = rng.standard_normal(g.shape)
    scale = float(sigma_rel) * float(np.linalg.norm(g))
    if delta_mode == "literal_covariance":
        delta = direction * scale
    elif delta_mode == "norm_15pct":
        denom = float(np.linalg.norm(direction))
        delta = np.zeros_like(direction) if denom == 0.0 else direction * (scale / denom)
    else:
        raise ValueError(f"Unknown delta mode {delta_mode!r}")
    return g + delta, delta


def sample_separated_vertices(
    rng: np.random.Generator,
    vertices: NDArray[np.floating],
    n_sources: int,
    min_sep_m: float,
) -> NDArray[np.intp]:
    """Random cortical vertices with pairwise distance >= min_sep_m."""
    vertices = np.asarray(vertices, dtype=np.float64)
    n_vertices = vertices.shape[0]
    order = rng.permutation(n_vertices)
    chosen: list[int] = []
    chosen_xyz: list[NDArray[np.float64]] = []
    for index in order:
        xyz = vertices[int(index)]
        if chosen_xyz:
            dist = np.linalg.norm(np.stack(chosen_xyz, axis=0) - xyz, axis=1)
            if np.min(dist) < min_sep_m:
                continue
        chosen.append(int(index))
        chosen_xyz.append(xyz)
        if len(chosen) == n_sources:
            return np.asarray(chosen, dtype=np.intp)
    raise RuntimeError(
        f"Could not place {n_sources} sources with min separation {min_sep_m} m"
    )


def mix_signal_and_noise(
    signal_ct: NDArray[np.floating],
    noise_ict: NDArray[np.floating],
    snr: float,
    *,
    gamma_mode: str = "per_trial",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast (C, N, T) signal over trials; scale noise to SNR.

    ``per_trial`` (original SIM-P1): a separate γ per trial.
    ``global``: one γ from RMS(signal) / (SNR · RMS(all noise)).
    ``noise_ict`` is (I_c, C, N, T).
    """
    signal_ct = np.asarray(signal_ct, dtype=np.float64)
    noise_ict = np.asarray(noise_ict, dtype=np.float64)
    if snr <= 0.0:
        raise ValueError("SNR must be positive")
    n_trials = noise_ict.shape[0]
    signal_rms = rms(signal_ct)
    if gamma_mode == "per_trial":
        noise_rms = np.sqrt(np.mean(noise_ict.reshape(n_trials, -1) ** 2, axis=1))
        if np.any(noise_rms == 0.0):
            raise ValueError("noise matrix has zero RMS for at least one trial")
        gammas = signal_rms / (float(snr) * noise_rms)
        trials = signal_ct[None, ...] + gammas[:, None, None, None] * noise_ict
        return trials, np.asarray(gammas, dtype=np.float64)
    if gamma_mode == "global":
        noise_rms = rms(noise_ict)
        if noise_rms == 0.0:
            raise ValueError("noise matrix has zero RMS")
        gamma = signal_rms / (float(snr) * noise_rms)
        trials = signal_ct[None, ...] + gamma * noise_ict
        return trials, np.full(n_trials, gamma, dtype=np.float64)
    raise ValueError(f"Unknown SNR gamma mode {gamma_mode!r}")


@dataclass
class SingleSourceDraw:
    vertex: int
    mixing: NDArray[np.float64]
    source: NDArray[np.float64]
    d0: NDArray[np.float64]
    d_target: NDArray[np.float64]
    topography: NDArray[np.float64]
    g_true: NDArray[np.float64]
    trials: NDArray[np.float64]  # (I_c, C, N, T)
    averages: NDArray[np.float64]  # (C, N, T)
    noiseless_average: NDArray[np.float64]
    gammas: NDArray[np.float64]
    snr: float


def simulate_single_source(
    forward: ForwardModel,
    rng: np.random.Generator,
    mixing: NDArray[np.floating],
    *,
    config: SimulationConfig,
    snr: float,
) -> SingleSourceDraw:
    n_cond = mixing.shape[0]
    source, _z = source_erp(
        mixing,
        rng,
        config.n_times,
        fs_hz=config.fs_hz,
        cutoff_hz=config.butter_cutoff_hz,
        order=config.butter_order,
        zero_phase=config.butter_zero_phase,
    )
    d0 = squared_euclidean_rdm(source)
    d_target = add_symmetric_rdm_noise(d0, rng, relative_std=config.upsilon_d_rel_std)
    vertex = int(rng.integers(0, forward.n_vertices))
    g_true = forward.gain[:, vertex]
    topo, _delta = topography_with_forward_error(
        g_true,
        rng,
        sigma_rel=config.sigma_delta_rel,
        delta_mode=config.delta_mode,
    )
    signal = topo[None, :, None] * source[:, None, :]  # (C, N, T)
    noise = cortical_one_over_f_sensor_noise(
        forward.gain,
        rng,
        n_epochs=config.i_c * n_cond,
        n_times=config.n_times,
        n_sources=config.n_noise_sources,
        fs_hz=config.fs_hz,
        exponent=config.pink_psd_exponent,
        loci_mode=config.noise_loci_mode,
    ).reshape(config.i_c, n_cond, forward.n_channels, config.n_times)
    trials, gammas = mix_signal_and_noise(
        signal, noise, snr, gamma_mode=config.snr_gamma_mode
    )
    averages = trials.mean(axis=0)
    return SingleSourceDraw(
        vertex=vertex,
        mixing=np.asarray(mixing, dtype=np.float64),
        source=source,
        d0=d0,
        d_target=d_target,
        topography=topo,
        g_true=g_true,
        trials=trials,
        averages=averages,
        noiseless_average=signal,
        gammas=gammas,
        snr=float(snr),
    )


@dataclass
class MultiSourceDraw:
    vertices: NDArray[np.intp]
    mixings: NDArray[np.float64]  # (P, C, C)
    sources: NDArray[np.float64]  # (P, C, T)
    d0: NDArray[np.float64]  # (P, C, C)
    d_target: NDArray[np.float64]
    topographies: NDArray[np.float64]  # (P, N)
    g_true: NDArray[np.float64]
    trials: NDArray[np.float64]
    averages: NDArray[np.float64]
    noiseless_average: NDArray[np.float64]
    gammas: NDArray[np.float64]
    snr: float


def simulate_multi_source(
    forward: ForwardModel,
    rng: np.random.Generator,
    mixings: NDArray[np.floating],
    *,
    config: SimulationConfig,
    snr: float,
    n_conditions: int,
) -> MultiSourceDraw:
    mixings = np.asarray(mixings, dtype=np.float64)
    n_sources = mixings.shape[0]
    if mixings.shape[1:] != (n_conditions, n_conditions):
        raise ValueError("mixings must be (P, C, C)")
    sources = np.empty((n_sources, n_conditions, config.n_times), dtype=np.float64)
    d0 = np.empty((n_sources, n_conditions, n_conditions), dtype=np.float64)
    d_target = np.empty_like(d0)
    for p in range(n_sources):
        sources[p], _ = source_erp(
            mixings[p],
            rng,
            config.n_times,
            fs_hz=config.fs_hz,
            cutoff_hz=config.butter_cutoff_hz,
            order=config.butter_order,
            zero_phase=config.butter_zero_phase,
        )
        d0[p] = squared_euclidean_rdm(sources[p])
        d_target[p] = add_symmetric_rdm_noise(
            d0[p], rng, relative_std=config.upsilon_d_rel_std
        )
    vertices = sample_separated_vertices(
        rng, forward.vertices, n_sources, config.min_sep_m
    )
    g_true = forward.gain[:, vertices].T  # (P, N)
    topographies = np.empty_like(g_true)
    for p in range(n_sources):
        topographies[p], _ = topography_with_forward_error(
            g_true[p],
            rng,
            sigma_rel=config.sigma_delta_rel,
            delta_mode=config.delta_mode,
        )
    signal = np.einsum("pn,pct->cnt", topographies, sources)
    if config.eq16_single_matrix:
        # Literal / near-literal Eq. (16): one X_c per condition, no I_c.
        # Noise is generated independently per condition (one Upsilon_x
        # matrix per X_c). Not shared across conditions. I_c is unused.
        noise = cortical_one_over_f_sensor_noise(
            forward.gain,
            rng,
            n_epochs=n_conditions,
            n_times=config.n_times,
            n_sources=config.n_noise_sources,
            fs_hz=config.fs_hz,
            exponent=config.pink_psd_exponent,
            loci_mode=config.noise_loci_mode,
        )
        noise = noise.reshape(1, n_conditions, forward.n_channels, config.n_times)
        trials, gammas = mix_signal_and_noise(
            signal, noise, snr, gamma_mode=config.snr_gamma_mode
        )
        averages = np.asarray(trials[0], dtype=np.float64)
    else:
        noise = cortical_one_over_f_sensor_noise(
            forward.gain,
            rng,
            n_epochs=config.i_c * n_conditions,
            n_times=config.n_times,
            n_sources=config.n_noise_sources,
            fs_hz=config.fs_hz,
            exponent=config.pink_psd_exponent,
            loci_mode=config.noise_loci_mode,
        ).reshape(config.i_c, n_conditions, forward.n_channels, config.n_times)
        trials, gammas = mix_signal_and_noise(
            signal, noise, snr, gamma_mode=config.snr_gamma_mode
        )
        averages = trials.mean(axis=0)
    return MultiSourceDraw(
        vertices=vertices,
        mixings=mixings,
        sources=sources,
        d0=d0,
        d_target=d_target,
        topographies=topographies,
        g_true=g_true,
        trials=trials,
        averages=averages,
        noiseless_average=signal,
        gammas=gammas,
        snr=float(snr),
    )


def subset_conditions_multi(draw: MultiSourceDraw, n_conditions: int) -> dict[str, Any]:
    """Slice a C=6 draw to the first ``n_conditions`` conditions (Fig. 6 / D14)."""
    if n_conditions > draw.averages.shape[0]:
        raise ValueError("requested more conditions than were simulated")
    c = int(n_conditions)
    return {
        "vertices": draw.vertices,
        "g_true": draw.g_true,
        "topographies": draw.topographies,
        "sources": draw.sources[:, :c],
        "d0": draw.d0[:, :c, :c],
        "d_target": draw.d_target[:, :c, :c],
        "trials": draw.trials[:, :c],
        "averages": draw.averages[:c],
        "noiseless_average": draw.noiseless_average[:c],
    }
