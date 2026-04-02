"""
SciPy translation of the MATLAB time-spectrum workflow in codes.txt.

This script:
1) Loads ProcessedEEG from EEG_processed_data.mat
2) Computes a sliding-window PSD time-frequency matrix
3) Derives total power, median frequency, peak frequency
4) Converts power to dB
5) Saves figures and analysis outputs
"""
#%%
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import welch


def _to_float_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).squeeze().astype(float)


def load_processed_eeg(mat_path: Path) -> Tuple[np.ndarray, float, np.ndarray]:
    try:
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "ProcessedEEG" not in data:
            raise KeyError("ProcessedEEG was not found in MAT file.")

        processed = data["ProcessedEEG"]
        eeg_signal = _to_float_1d(processed.eeg_final)
        fs = float(np.asarray(processed.Fs_processed).squeeze())
        t_vector = _to_float_1d(processed.t_1000hz)
        return eeg_signal, fs, t_vector
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5; read via h5py.
        import h5py

        with h5py.File(mat_path, "r") as f:
            if "ProcessedEEG" not in f:
                raise KeyError("ProcessedEEG was not found in MAT file (HDF5).")
            g = f["ProcessedEEG"]
            eeg_signal = _to_float_1d(np.array(g["eeg_final"]))
            fs = float(np.asarray(g["Fs_processed"]).squeeze())
            t_vector = _to_float_1d(np.array(g["t_1000hz"]))
            return eeg_signal, fs, t_vector


def compute_spectrogram_scipy(
    signal: np.ndarray,
    fs: float,
    window_size_s: float,
    overlap: float,
    freq_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_samples = int(round(window_size_s * fs))
    step_samples = int(round(window_size_s * (1.0 - overlap) * fs))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Invalid window size or overlap configuration.")

    total_samples = signal.size
    if total_samples < window_samples:
        raise ValueError("Signal length is shorter than one analysis window.")

    num_windows = int(np.floor((total_samples - window_samples) / step_samples) + 1)

    spectrogram_rows = []
    time_vector = np.zeros(num_windows, dtype=float)
    freq_vector = None

    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        segment = signal[start:end]

        freqs, psd = welch(
            segment,
            fs=fs,
            window="hann",
            nperseg=window_samples,
            noverlap=0,
            detrend="constant",
            scaling="density",
        )

        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if freq_vector is None:
            freq_vector = freqs[mask]

        spectrogram_rows.append(psd[mask])
        time_vector[i] = ((start + end - 1) / 2.0) / fs

    spectrogram_matrix = np.vstack(spectrogram_rows)
    return spectrogram_matrix, np.asarray(freq_vector), time_vector


def compute_metrics(
    spectrogram_matrix: np.ndarray, freq_vector: np.ndarray, freq_range: Tuple[float, float]
) -> Dict[str, np.ndarray]:
    freq_mask = (freq_vector >= freq_range[0]) & (freq_vector <= freq_range[1])
    freq_in_band = freq_vector[freq_mask]
    spectrum_in_band = spectrogram_matrix[:, freq_mask]

    total_power = np.sum(spectrum_in_band, axis=1)

    median_freq = np.zeros(spectrum_in_band.shape[0], dtype=float)
    for i in range(spectrum_in_band.shape[0]):
        cur = spectrum_in_band[i]
        cumulative = np.cumsum(cur)
        if cumulative.size > 0 and cumulative[-1] > 0:
            idx = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
            idx = min(idx, freq_in_band.size - 1)
            median_freq[i] = freq_in_band[idx]

    peak_freq_idx = np.argmax(spectrum_in_band, axis=1)
    peak_freq = freq_in_band[peak_freq_idx]

    eps = np.finfo(float).tiny
    spectrogram_db = 10.0 * np.log10(np.maximum(spectrogram_matrix, eps))
    total_power_db = 10.0 * np.log10(np.maximum(total_power, eps))

    return {
        "total_power": total_power,
        "median_freq": median_freq,
        "peak_freq": peak_freq,
        "spectrogram_db": spectrogram_db,
        "total_power_db": total_power_db,
    }


def _pick_top_indices(values: np.ndarray, count: int, min_spacing_bins: int) -> np.ndarray:
    ranked = np.argsort(values)[::-1]
    selected: list[int] = []
    for idx in ranked:
        if values[idx] <= 0:
            continue
        if all(abs(idx - s) >= min_spacing_bins for s in selected):
            selected.append(int(idx))
            if len(selected) >= count:
                break
    if not selected:
        return np.array([], dtype=int)
    return np.array(sorted(selected), dtype=int)


def detect_moving_noise_ridges(
    spectrogram_matrix: np.ndarray,
    freq_vector: np.ndarray,
    num_lines: int = 6,
    min_freq_hz: float = 4.0,
    max_freq_hz: float = 40.0,
    init_smooth_windows: int = 8,
    min_spacing_hz: float = 4.0,
    search_radius_hz: float = 2.0,
    seed_freqs_hz: Optional[np.ndarray] = None,
    max_drift_hz: float = 6.0,
) -> np.ndarray:
    """
    Detect moving narrowband noise lines by frame-wise ridge tracking.
    Returns ridge indices with shape (num_lines, n_times).
    """
    n_times, n_freq = spectrogram_matrix.shape
    if n_times == 0 or n_freq == 0:
        return np.empty((0, 0), dtype=int)

    band = (freq_vector >= float(min_freq_hz)) & (freq_vector <= float(max_freq_hz))
    band_idx = np.where(band)[0]
    if band_idx.size == 0:
        return np.empty((0, n_times), dtype=int)

    bin_hz = float(np.median(np.diff(freq_vector))) if freq_vector.size > 1 else 0.25
    spacing_bins = max(1, int(round(float(min_spacing_hz) / max(bin_hz, 1e-6))))
    radius_bins = max(1, int(round(float(search_radius_hz) / max(bin_hz, 1e-6))))
    drift_bins = max(radius_bins, int(round(float(max_drift_hz) / max(bin_hz, 1e-6))))

    init_count = max(1, min(n_times, int(init_smooth_windows)))
    seed_profile = 0.7 * np.median(spectrogram_matrix, axis=0) + 0.3 * np.mean(
        spectrogram_matrix[:init_count, :], axis=0
    )
    if seed_freqs_hz is not None and np.asarray(seed_freqs_hz).size > 0:
        seed_freqs = np.asarray(seed_freqs_hz, dtype=float)
        seed_freqs = seed_freqs[(seed_freqs >= float(min_freq_hz)) & (seed_freqs <= float(max_freq_hz))]
        if seed_freqs.size == 0:
            return np.empty((0, n_times), dtype=int)
        seeds = np.array([int(np.argmin(np.abs(freq_vector - f))) for f in seed_freqs], dtype=int)
        seeds = np.unique(seeds)
    else:
        seed_vals = seed_profile[band_idx]
        seed_local = _pick_top_indices(seed_vals, int(num_lines), spacing_bins)
        if seed_local.size == 0:
            return np.empty((0, n_times), dtype=int)
        seeds = band_idx[seed_local]
    num_tracks = seeds.size
    ridges = np.zeros((num_tracks, n_times), dtype=int)
    first_frame = spectrogram_matrix[0, :]
    seed_centers = seeds.copy()
    for k, seed in enumerate(seeds):
        lo = max(band_idx[0], seed - radius_bins)
        hi = min(band_idx[-1], seed + radius_bins)
        ridges[k, 0] = lo + int(np.argmax(first_frame[lo : hi + 1]))

    for t in range(1, n_times):
        frame = spectrogram_matrix[t, :]
        chosen_current: list[int] = []
        for k in range(num_tracks):
            prev = ridges[k, t - 1]
            center = int(seed_centers[k])
            lo = max(band_idx[0], prev - radius_bins, center - drift_bins)
            hi = min(band_idx[-1], prev + radius_bins, center + drift_bins)
            local = frame[lo : hi + 1]
            local_idx = int(np.argmax(local))
            candidate = lo + local_idx

            if chosen_current:
                for used in chosen_current:
                    if abs(candidate - used) < spacing_bins:
                        local_masked = local.copy()
                        for u in chosen_current:
                            mlo = max(lo, u - spacing_bins)
                            mhi = min(hi, u + spacing_bins)
                            local_masked[mlo - lo : mhi - lo + 1] = -np.inf
                        if np.isfinite(local_masked).any():
                            candidate = lo + int(np.argmax(local_masked))
                        break

            ridges[k, t] = candidate
            chosen_current.append(candidate)

    return ridges


def spectral_subtraction_denoise(
    spectrogram_matrix: np.ndarray,
    ridges: np.ndarray,
    time_vector: np.ndarray,
    band_half_width_hz: float,
    guard_hz: float,
    reference_hz: float,
    alpha: float,
    floor_ratio: float,
    freq_vector: np.ndarray,
    adaptive_alpha: bool = True,
    alpha_power_gamma: float = 1.0,
    alpha_scale_min: float = 0.35,
    alpha_scale_max: float = 2.5,
    alpha_smooth_windows: int = 15,
    min_db_floor: float = 0.0,
    fill_neighborhood_hz: float = 1.0,
    fill_time_window_s: float = 5.0,
    notch_skip_hz: float = 0.5,
) -> np.ndarray:
    """
    Subtract local background around moving ridge lines in power domain.
    """
    cleaned = spectrogram_matrix.copy()
    if ridges.size == 0:
        return cleaned

    bin_hz = float(np.median(np.diff(freq_vector))) if freq_vector.size > 1 else 0.25
    half_bins = max(0, int(round(float(band_half_width_hz) / max(bin_hz, 1e-6))))
    guard_bins = max(1, int(round(float(guard_hz) / max(bin_hz, 1e-6))))
    ref_bins = max(1, int(round(float(reference_hz) / max(bin_hz, 1e-6))))
    fill_bins = max(1, int(round(float(fill_neighborhood_hz) / max(bin_hz, 1e-6))))
    notch_skip_bins = max(0, int(round(float(notch_skip_hz) / max(bin_hz, 1e-6))))
    floor_ratio = float(np.clip(floor_ratio, 0.0, 1.0))
    min_power_floor = float(10.0 ** (float(min_db_floor) / 10.0))
    dt = float(np.median(np.diff(time_vector))) if time_vector.size > 1 else 1.0
    time_bins = max(0, int(round(float(fill_time_window_s) / max(dt, 1e-6))))

    n_times, n_freq = cleaned.shape
    if adaptive_alpha:
        total_power_t = np.sum(spectrogram_matrix, axis=1)
        ref_power = max(float(np.median(total_power_t)), np.finfo(float).tiny)
        alpha_scale_t = (total_power_t / ref_power) ** float(alpha_power_gamma)
        alpha_scale_t = np.clip(alpha_scale_t, float(alpha_scale_min), float(alpha_scale_max))

        if int(alpha_smooth_windows) > 1:
            w = int(alpha_smooth_windows)
            kernel = np.ones(w, dtype=float) / w
            alpha_scale_t = np.convolve(alpha_scale_t, kernel, mode="same")
            alpha_scale_t = np.clip(alpha_scale_t, float(alpha_scale_min), float(alpha_scale_max))
    else:
        alpha_scale_t = np.ones(n_times, dtype=float)

    for t in range(n_times):
        alpha_t = float(alpha) * float(alpha_scale_t[t])
        src_frame = spectrogram_matrix[t, :]
        for center in ridges[:, t]:
            left_sig = max(0, center - half_bins)
            right_sig = min(n_freq - 1, center + half_bins)

            left_ref_a = max(0, left_sig - guard_bins - ref_bins)
            left_ref_b = max(-1, left_sig - guard_bins - 1)
            right_ref_a = min(n_freq, right_sig + guard_bins + 1)
            right_ref_b = min(n_freq - 1, right_sig + guard_bins + ref_bins)

            refs = []
            if left_ref_b >= left_ref_a:
                refs.append(cleaned[t, left_ref_a : left_ref_b + 1])
            if right_ref_b >= right_ref_a:
                refs.append(cleaned[t, right_ref_a : right_ref_b + 1])
            if not refs:
                continue

            ref_power = float(np.median(np.concatenate(refs)))
            sig_slice = slice(left_sig, right_sig + 1)
            original = cleaned[t, sig_slice]
            subtracted = original - alpha_t * ref_power
            lower_bound = np.maximum(original * floor_ratio, min_power_floor)
            clipped = np.maximum(subtracted, lower_bound)

            # Replace denoised ridge band with only higher-frequency neighborhood mean.
            right_fill_a = min(n_freq, right_sig + 1)
            right_fill_b = min(n_freq - 1, right_sig + fill_bins)

            if right_fill_b >= right_fill_a:
                t_lo = max(0, t - time_bins)
                t_hi = min(n_times - 1, t + time_bins)
                block = spectrogram_matrix[t_lo : t_hi + 1, right_fill_a : right_fill_b + 1]
                valid_mask = np.ones_like(block, dtype=bool)

                # Skip regions where any tracked notch/ridge may pass through.
                if notch_skip_bins > 0 and ridges.size > 0:
                    for tt in range(t_lo, t_hi + 1):
                        for c_notch in ridges[:, tt]:
                            lo_skip = max(right_fill_a, int(c_notch) - notch_skip_bins)
                            hi_skip = min(right_fill_b, int(c_notch) + notch_skip_bins)
                            if hi_skip >= lo_skip:
                                valid_mask[tt - t_lo, lo_skip - right_fill_a : hi_skip - right_fill_a + 1] = False

                fill_candidates = block[valid_mask]
                if fill_candidates.size > 0:
                    fill_value = float(np.mean(fill_candidates))
                else:
                    fill_value = float(np.mean(src_frame[right_fill_a : right_fill_b + 1]))
                cleaned[t, sig_slice] = max(fill_value, min_power_floor)
            else:
                cleaned[t, sig_slice] = clipped

    return cleaned


def save_figures(
    eeg_signal: np.ndarray,
    t_vector: np.ndarray,
    time_vector: np.ndarray,
    freq_vector: np.ndarray,
    metrics: Dict[str, np.ndarray],
    output_dir: Path,
    dbrange: Tuple[float, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    display_mask = (freq_vector >= 0.5) & (freq_vector <= 40.0)
    display_freq = freq_vector[display_mask]
    display_spec_db = metrics["spectrogram_db"][:, display_mask].T

    fig_main = plt.figure(figsize=(14, 9))
    gs = fig_main.add_gridspec(4, 2, width_ratios=[40, 1], wspace=0.05, hspace=0.5)
    ax1 = fig_main.add_subplot(gs[0, 0])
    ax1.plot(t_vector, eeg_signal, "b", linewidth=0.8)
    ax1.set_title("EEG signal used for analysis")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.set_xlim(0, t_vector[-1])
    ax1.grid(True, alpha=0.3)
    fig_main.add_subplot(gs[0, 1]).axis("off")

    ax2 = fig_main.add_subplot(gs[1, 0], sharex=ax1)
    im2 = ax2.imshow(
        display_spec_db,
        origin="lower",
        aspect="auto",
        extent=[time_vector[0], time_vector[-1], display_freq[0], display_freq[-1]],
        cmap="jet",
        vmin=dbrange[0],
        vmax=dbrange[1],
    )
    cax2 = fig_main.add_subplot(gs[1, 1])
    fig_main.colorbar(im2, cax=cax2, label="Power (dB)")
    ax2.set_title(f"Time-frequency map (dB) [{dbrange[0]:.0f}, {dbrange[1]:.0f}]")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 40)
    ax2.set_xlim(0, t_vector[-1])

    ax3 = fig_main.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(time_vector, metrics["total_power_db"], "r-", linewidth=2)
    ax3.set_title("Total power (0.5-50 Hz, dB)")
    ax3.set_ylabel("Power (dB)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim(0, t_vector[-1])
    ax3.grid(True, alpha=0.3)
    fig_main.add_subplot(gs[2, 1]).axis("off")

    ax4 = fig_main.add_subplot(gs[3, 0], sharex=ax1)
    ax4.plot(time_vector, metrics["median_freq"], "g-", linewidth=2, label="Median frequency")
    ax4.plot(time_vector, metrics["peak_freq"], "m-", linewidth=2, label="Peak frequency")
    ax4.set_title("Frequency features over time")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_xlabel("Time (s)")
    ax4.set_xlim(0, t_vector[-1])
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    fig_main.add_subplot(gs[3, 1]).axis("off")

    fig_main.suptitle(f"Overall EEG spectral analysis [{dbrange[0]:.0f}, {dbrange[1]:.0f}] dB")
    fig_main.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06, hspace=0.55, wspace=0.05)
    fig_main.savefig(output_dir / "overall_spectral_analysis_dB.png", dpi=150)
    plt.close(fig_main)

    fig_detail = plt.figure(figsize=(16, 6))
    axd = fig_detail.add_subplot(1, 1, 1)
    imd = axd.imshow(
        display_spec_db,
        origin="lower",
        aspect="auto",
        extent=[time_vector[0], time_vector[-1], display_freq[0], display_freq[-1]],
        cmap="jet",
        vmin=dbrange[0],
        vmax=dbrange[1],
    )
    fig_detail.colorbar(imd, ax=axd, label="Power (dB)")
    axd.set_title(f"Detailed time-frequency map [{dbrange[0]:.0f}, {dbrange[1]:.0f}] dB")
    axd.set_xlabel("Time (s)")
    axd.set_ylabel("Frequency (Hz)")
    axd.set_ylim(0, 40)
    axd.grid(True, alpha=0.2)
    fig_detail.tight_layout()
    fig_detail.savefig(output_dir / "detailed_spectrogram_dB.png", dpi=150)
    plt.close(fig_detail)

    color_ranges = np.array([[-5, 15], [-2, 10], [0, 12], [-3, 8], [-8, 20]], dtype=float)
    range_names = ["Wide [-5,15]", "Mid [-2,10]", "High contrast [0,12]", "Narrow [-3,8]", "Very wide [-8,20]"]
    actual_range = [float(np.min(display_spec_db)), float(np.max(display_spec_db))]

    fig_cmp = plt.figure(figsize=(16, 10))
    for i in range(5):
        ax = fig_cmp.add_subplot(2, 3, i + 1)
        im = ax.imshow(
            display_spec_db,
            origin="lower",
            aspect="auto",
            extent=[time_vector[0], time_vector[-1], display_freq[0], display_freq[-1]],
            cmap="jet",
            vmin=color_ranges[i, 0],
            vmax=color_ranges[i, 1],
        )
        fig_cmp.colorbar(im, ax=ax)
        ax.set_title(f"{range_names[i]} dB")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 40)

    ax6 = fig_cmp.add_subplot(2, 3, 6)
    im6 = ax6.imshow(
        display_spec_db,
        origin="lower",
        aspect="auto",
        extent=[time_vector[0], time_vector[-1], display_freq[0], display_freq[-1]],
        cmap="jet",
        vmin=actual_range[0],
        vmax=actual_range[1],
    )
    fig_cmp.colorbar(im6, ax=ax6)
    ax6.set_title(f"Actual [{actual_range[0]:.1f}, {actual_range[1]:.1f}] dB")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Frequency (Hz)")
    ax6.set_ylim(0, 40)

    fig_cmp.suptitle("Color range comparison")
    fig_cmp.tight_layout()
    fig_cmp.savefig(output_dir / "color_range_comparison.png", dpi=150)
    plt.close(fig_cmp)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple EEG time-spectrum analysis (SciPy version)")
    parser.add_argument(
        "--input-mat",
        type=str,
        default=r"E:\#Preprocessed_Data\Others\260326_Spectrum\EEG_processed_data.mat",
        help="Path to EEG_processed_data.mat",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "python_spectral_analysis"),
        help="Directory where outputs are saved",
    )
    parser.add_argument("--db-min", type=float, default=-25.0, help="Colorbar min dB")
    parser.add_argument("--db-max", type=float, default=10.0, help="Colorbar max dB")
    parser.add_argument(
        "--denoise-method",
        type=str,
        choices=["none", "spectral_subtraction"],
        default="spectral_subtraction",
        help="Denoise method to apply before metrics/plots.",
    )
    parser.add_argument(
        "--ss-num-lines",
        type=int,
        default=6,
        help="Number of moving noise lines to track.",
    )
    parser.add_argument(
        "--ss-min-freq",
        type=float,
        default=4.0,
        help="Minimum frequency (Hz) for ridge detection.",
    )
    parser.add_argument(
        "--ss-max-freq",
        type=float,
        default=40.0,
        help="Maximum frequency (Hz) for ridge detection.",
    )
    parser.add_argument(
        "--ss-init-smooth-windows",
        type=int,
        default=8,
        help="Number of initial windows to average for ridge seeding.",
    )
    parser.add_argument(
        "--ss-min-spacing",
        type=float,
        default=4.0,
        help="Minimum spacing (Hz) between tracked lines.",
    )
    parser.add_argument(
        "--ss-search-radius",
        type=float,
        default=2.0,
        help="Per-time ridge search radius (Hz).",
    )
    parser.add_argument(
        "--ss-max-drift",
        type=float,
        default=6.0,
        help="Maximum drift (Hz) away from each seeded ridge center.",
    )
    parser.add_argument(
        "--ss-band-half-width",
        type=float,
        default=0.35,
        help="Half-width (Hz) of each line band to suppress.",
    )
    parser.add_argument(
        "--ss-guard",
        type=float,
        default=0.5,
        help="Guard width (Hz) around line before reference region.",
    )
    parser.add_argument(
        "--ss-reference-width",
        type=float,
        default=1.5,
        help="Reference width (Hz) per side used for noise estimate.",
    )
    parser.add_argument(
        "--ss-alpha",
        type=float,
        default=1.2,
        help="Subtraction strength in power domain.",
    )
    parser.add_argument(
        "--ss-floor-ratio",
        type=float,
        default=0.15,
        help="Lower floor ratio after subtraction to avoid over-suppression.",
    )
    parser.add_argument(
        "--ss-adaptive-alpha",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use time-varying subtraction strength based on current total power (1=on).",
    )
    parser.add_argument(
        "--ss-alpha-power-gamma",
        type=float,
        default=1.0,
        help="Nonlinearity for adaptive alpha scaling by total power.",
    )
    parser.add_argument(
        "--ss-alpha-scale-min",
        type=float,
        default=0.35,
        help="Minimum scaling factor for adaptive alpha.",
    )
    parser.add_argument(
        "--ss-alpha-scale-max",
        type=float,
        default=2.5,
        help="Maximum scaling factor for adaptive alpha.",
    )
    parser.add_argument(
        "--ss-alpha-smooth-windows",
        type=int,
        default=15,
        help="Temporal smoothing window for adaptive alpha scale.",
    )
    parser.add_argument(
        "--ss-min-db-floor",
        type=float,
        default=0.0,
        help="Hard minimum dB floor after spectral subtraction.",
    )
    parser.add_argument(
        "--ss-fill-neighborhood",
        type=float,
        default=1.0,
        help="Higher-frequency neighborhood width (Hz) used to mean-fill each removed ridge band.",
    )
    parser.add_argument(
        "--ss-fill-time-window",
        type=float,
        default=5.0,
        help="Use +/- this many seconds when computing fill mean.",
    )
    parser.add_argument(
        "--ss-notch-skip",
        type=float,
        default=0.5,
        help="Skip +/- this many Hz around tracked notch paths when filling.",
    )
    parser.add_argument(
        "--ss-seed-freqs",
        type=str,
        default="",
        help="Optional comma-separated seed line frequencies (Hz) for ridge tracking.",
    )
    return parser


def _resolve_args(args: Optional[Any], parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args is None:
        # CLI mode (also notebook-safe due to parse_known_args)
        parsed, _ = parser.parse_known_args()
        return parsed

    defaults = parser.parse_args([])
    allowed = set(vars(defaults).keys())

    if isinstance(args, dict):
        for key, value in args.items():
            if key not in allowed:
                raise ValueError(f"Unknown argument key: {key}")
            setattr(defaults, key, value)
        return defaults

    if isinstance(args, argparse.Namespace):
        for key, value in vars(args).items():
            if key not in allowed:
                raise ValueError(f"Unknown argument key: {key}")
            setattr(defaults, key, value)
        return defaults

    raise TypeError("args must be None, dict, or argparse.Namespace")


def main(args: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run analysis in CLI or interactive mode.

    Usage:
        - CLI: main()
        - Jupyter: main({"denoise_method": "spectral_subtraction", "db_min": -20, "db_max": 8})
    """
    parser = _build_parser()
    # Jupyter/ipykernel appends its own CLI args (for example --f=kernel.json).
    # parse_known_args keeps our script flags while ignoring unrelated ones.
    args = _resolve_args(args, parser)

    input_mat = Path(args.input_mat)
    output_dir = Path(args.output_dir)

    print("=== Simple time-frequency analysis (Python/SciPy) ===")
    print(f"Loading data from: {input_mat}")
    if not input_mat.exists():
        raise FileNotFoundError(f"Input MAT file not found: {input_mat}")

    eeg_signal, fs, t_vector = load_processed_eeg(input_mat)
    print(f"Loaded successfully. Samples={eeg_signal.size}, Duration={t_vector[-1]:.2f}s, Fs={fs:.1f}Hz")

    window_size = 4.0
    overlap = 0.5
    freq_range = (0.5, 50.0)

    spectrogram_matrix, freq_vector, time_vector = compute_spectrogram_scipy(
        eeg_signal, fs, window_size, overlap, freq_range
    )
    spectrogram_matrix_raw = spectrogram_matrix.copy()

    denoise_method = str(args.denoise_method)

    removed_noise_freqs = np.array([], dtype=float)
    ss_info: Dict[str, np.ndarray] = {}
    processed_signal = eeg_signal.copy()

    if denoise_method == "spectral_subtraction":
        seed_freqs = None
        seed_text = str(args.ss_seed_freqs).strip()
        if seed_text:
            tokens = [tok.strip() for tok in seed_text.split(",") if tok.strip()]
            seed_freqs = np.array([float(tok) for tok in tokens], dtype=float)

        ridges = detect_moving_noise_ridges(
            spectrogram_matrix=spectrogram_matrix_raw,
            freq_vector=freq_vector,
            num_lines=int(args.ss_num_lines),
            min_freq_hz=float(args.ss_min_freq),
            max_freq_hz=float(args.ss_max_freq),
            init_smooth_windows=int(args.ss_init_smooth_windows),
            min_spacing_hz=float(args.ss_min_spacing),
            search_radius_hz=float(args.ss_search_radius),
            seed_freqs_hz=seed_freqs,
            max_drift_hz=float(args.ss_max_drift),
        )
        spectrogram_matrix = spectral_subtraction_denoise(
            spectrogram_matrix=spectrogram_matrix_raw,
            ridges=ridges,
            time_vector=time_vector,
            band_half_width_hz=float(args.ss_band_half_width),
            guard_hz=float(args.ss_guard),
            reference_hz=float(args.ss_reference_width),
            alpha=float(args.ss_alpha),
            floor_ratio=float(args.ss_floor_ratio),
            freq_vector=freq_vector,
            adaptive_alpha=bool(int(args.ss_adaptive_alpha)),
            alpha_power_gamma=float(args.ss_alpha_power_gamma),
            alpha_scale_min=float(args.ss_alpha_scale_min),
            alpha_scale_max=float(args.ss_alpha_scale_max),
            alpha_smooth_windows=int(args.ss_alpha_smooth_windows),
            min_db_floor=float(args.ss_min_db_floor),
            fill_neighborhood_hz=float(args.ss_fill_neighborhood),
            fill_time_window_s=float(args.ss_fill_time_window),
            notch_skip_hz=float(args.ss_notch_skip),
        )

        if ridges.size > 0:
            removed_noise_freqs = np.mean(freq_vector[ridges], axis=1)
        ss_info = {
            "ridges_idx": ridges,
            "ridges_mean_freq_hz": removed_noise_freqs,
        }

    metrics = compute_metrics(spectrogram_matrix, freq_vector, freq_range)

    dbrange = (args.db_min, args.db_max)
    save_figures(
        eeg_signal=processed_signal,
        t_vector=t_vector,
        time_vector=time_vector,
        freq_vector=freq_vector,
        metrics=metrics,
        output_dir=output_dir,
        dbrange=dbrange,
    )

    summary = {
        "spectrogram": spectrogram_matrix,
        "spectrogram_raw": spectrogram_matrix_raw,
        "spectrogram_dB": metrics["spectrogram_db"],
        "freq_vector": freq_vector,
        "time_vector": time_vector,
        "total_power": metrics["total_power"],
        "total_power_dB": metrics["total_power_db"],
        "median_frequency": metrics["median_freq"],
        "peak_frequency": metrics["peak_freq"],
        "analysis_params": {
            "window_size": window_size,
            "overlap": overlap,
            "tapers": "scipy_welch_fallback",
            "input_mat_path": str(input_mat),
            "output_dir": str(output_dir),
            "freq_range": np.array(freq_range, dtype=float),
            "denoise_method": denoise_method,
            "removed_noise_freqs_hz": removed_noise_freqs,
            "ss_num_lines": int(args.ss_num_lines),
            "ss_min_freq": float(args.ss_min_freq),
            "ss_max_freq": float(args.ss_max_freq),
            "ss_init_smooth_windows": int(args.ss_init_smooth_windows),
            "ss_min_spacing": float(args.ss_min_spacing),
            "ss_search_radius": float(args.ss_search_radius),
            "ss_max_drift": float(args.ss_max_drift),
            "ss_band_half_width": float(args.ss_band_half_width),
            "ss_guard": float(args.ss_guard),
            "ss_reference_width": float(args.ss_reference_width),
            "ss_alpha": float(args.ss_alpha),
            "ss_floor_ratio": float(args.ss_floor_ratio),
            "ss_adaptive_alpha": bool(int(args.ss_adaptive_alpha)),
            "ss_alpha_power_gamma": float(args.ss_alpha_power_gamma),
            "ss_alpha_scale_min": float(args.ss_alpha_scale_min),
            "ss_alpha_scale_max": float(args.ss_alpha_scale_max),
            "ss_alpha_smooth_windows": int(args.ss_alpha_smooth_windows),
            "ss_min_db_floor": float(args.ss_min_db_floor),
            "ss_fill_neighborhood": float(args.ss_fill_neighborhood),
            "ss_fill_time_window": float(args.ss_fill_time_window),
            "ss_notch_skip": float(args.ss_notch_skip),
            "ss_seed_freqs": str(args.ss_seed_freqs),
            "ss_ridges_mean_freq_hz": ss_info.get("ridges_mean_freq_hz", np.array([], dtype=float)),
        },
        "user_color_range": np.array(dbrange, dtype=float),
    }
    savemat(output_dir / "simple_spectral_results.mat", {"SimpleSpectralResults": summary})

    total_power_db = metrics["total_power_db"]
    median_freq = metrics["median_freq"]
    peak_freq = metrics["peak_freq"]
    print("\n=== Summary (dB units) ===")
    print(f"Mean total power: {np.mean(total_power_db):.2f} dB")
    print(f"Power range: {np.min(total_power_db):.2f} to {np.max(total_power_db):.2f} dB")
    print(f"Power std: {np.std(total_power_db):.2f} dB")
    print(f"Median frequency: {np.mean(median_freq):.2f} +- {np.std(median_freq):.2f} Hz")
    print(f"Peak frequency: {np.mean(peak_freq):.2f} +- {np.std(peak_freq):.2f} Hz")
    print(f"Dynamic range: {np.max(total_power_db) - np.min(total_power_db):.2f} dB")
    if denoise_method == "spectral_subtraction":
        if removed_noise_freqs.size > 0:
            rounded = ", ".join([f"{f:.2f}" for f in removed_noise_freqs[:20]])
            suffix = " ..." if removed_noise_freqs.size > 20 else ""
            print(f"Tracked moving-noise lines mean freq (Hz): {rounded}{suffix}")
        else:
            print("Tracked moving-noise lines mean freq (Hz): none detected")
    print(f"\nOutputs saved to: {output_dir}")
    return {
        "output_dir": str(output_dir),
        "summary_mat": str(output_dir / "simple_spectral_results.mat"),
        "denoise_method": denoise_method,
        "removed_noise_freqs_hz": removed_noise_freqs,
        "ss_ridges_mean_freq_hz": ss_info.get("ridges_mean_freq_hz", np.array([], dtype=float)),
        "mean_total_power_db": float(np.mean(total_power_db)),
        "power_range_db": (float(np.min(total_power_db)), float(np.max(total_power_db))),
    }

#%%
if __name__ == "__main__":
    main({
    "input_mat": r"E:\#Preprocessed_Data\Others\260326_Spectrum\EEG_processed_data.mat",
    "denoise_method": "spectral_subtraction",
    "ss_num_lines": 6,
    "ss_seed_freqs": "6,12,18,24,31,37",
    "ss_min_freq": 4.0,
    "ss_max_freq": 40.0,
    "ss_search_radius": 5.0,
    "ss_max_drift": 5.0,
    "ss_min_db_floor": -100,
    "ss_min_spacing": 3.0,
    "ss_band_half_width": 0.3,
    "ss_guard": 0.5,
    "ss_reference_width": 3.0,
    "ss_alpha": 1e3,                 # 先从 250 开始，不要 1e3
    "ss_floor_ratio": 0.0,
    "ss_adaptive_alpha": 1,
    "ss_alpha_power_gamma": 1.0,
    "ss_alpha_scale_min": 0.1,        # 低功率段更保守
    "ss_alpha_scale_max": 100,         # 高功率段更积极
    "ss_alpha_smooth_windows": 2,     # 更平滑
    "db_min": -20,
    "db_max": 8,
    "ss_fill_time_window": 0.5,
    "ss_fill_neighborhood": 0.5,
    "ss_notch_skip": 0.3,
})

# #denoise_method: "spectral_subtraction" / "none"
# ss_num_lines: 追踪线条数（默认 6）
# ss_min_freq, ss_max_freq: 追踪频段
# ss_seed_freqs: 手动给初始线频（逗号分隔，推荐）
# ss_search_radius: 每帧搜索半径（Hz）
# ss_max_drift: 相对初始线允许最大漂移（Hz）
# ss_band_half_width: 要抑制的线宽半宽（Hz）
# ss_guard: 保护带（Hz）
# ss_reference_width: 两侧噪声参考带宽（Hz）
# ss_alpha: 谱减强度（越大越强）
# ss_floor_ratio: 下限保护，防止过减