import numpy as np
from scipy.signal import find_peaks, welch

try:
    from config import (
        HR_PEAK_PROMINENCE_FACTOR, HR_PEAK_DISTANCE_FACTOR,
        HR_MIN_VALID, HR_MAX_VALID, RR_MIN_VALID, RR_MAX_VALID,
        DEBUG_MODE
    )
except ImportError:
    HR_PEAK_PROMINENCE_FACTOR = 0.6
    HR_PEAK_DISTANCE_FACTOR = 0.4
    HR_MIN_VALID = 30
    HR_MAX_VALID = 200
    RR_MIN_VALID = 0.3
    RR_MAX_VALID = 2.0
    DEBUG_MODE = False

EPS = 1e-8

def _safe_length_req(fs, min_seconds=1.0):
    # Minimum number of samples required for processing.
    return max(2, int(round(min_seconds * float(fs))))


def calculate_hr_from_peaks(bvp_signal, fs):
    # Calculate heart rate and NN intervals from BVP signal using peak detection.
    if bvp_signal is None:
        return None, None

    sig = np.asarray(bvp_signal, dtype=float)

    # Check for minimum required signal length
    if sig.size < _safe_length_req(fs, min_seconds=1.0):
        return None, None

    # Handle NaNs by interpolation
    if np.isnan(sig).any():
        idx = np.arange(len(sig))
        good = ~np.isnan(sig)
        if good.sum() < 2:
            return None, None
        sig = np.interp(idx, idx[good], sig[good])

    try:
        # Detrend signal (remove mean)
        sig = sig - np.mean(sig)

        signal_std = float(np.std(sig))
        if signal_std <= 0:
            return None, None

        # Calculate prominence threshold
        prominence = signal_std * HR_PEAK_PROMINENCE_FACTOR

        # Calculate minimum peak distance in samples
        peak_distance = max(1, int(round(HR_PEAK_DISTANCE_FACTOR * float(fs))))

        # Find peaks
        peaks, props = find_peaks(sig, distance=peak_distance, prominence=prominence)

        if len(peaks) < 2:
            if DEBUG_MODE:
                print("HR: Insufficient peaks detected for HR calculation")
            return None, None

        # Calculate RR intervals in seconds
        rr_intervals_sec = np.diff(peaks) / float(fs)

        # Filter physiologically invalid intervals
        valid_mask = (rr_intervals_sec >= RR_MIN_VALID) & (rr_intervals_sec <= RR_MAX_VALID)
        valid_intervals = rr_intervals_sec[valid_mask]

        if len(valid_intervals) < 1:
            if DEBUG_MODE:
                print("HR: No valid RR intervals after filtering")
            return None, None

        # Calculate mean HR in BPM
        mean_rr_sec = float(np.mean(valid_intervals))
        hr_bpm = 60.0 / (mean_rr_sec + EPS)

        # Sanity check HR range
        if hr_bpm < HR_MIN_VALID or hr_bpm > HR_MAX_VALID:
            if DEBUG_MODE:
                print(f"HR: Calculated HR out of range: {hr_bpm:.1f} bpm")
            return None, None

        # Convert valid intervals to milliseconds (NN intervals)
        nn_intervals_ms = np.asarray(valid_intervals * 1000.0, dtype=float)

        return float(hr_bpm), nn_intervals_ms

    except Exception as e:
        if DEBUG_MODE:
            print(f"HR calculation error: {e}")
        return None, None


def get_hr_from_spectrum(bvp_signal, fs):
    # Extract heart rate from frequency spectrum (Welch).
    if bvp_signal is None:
        return None
    sig = np.asarray(bvp_signal, dtype=float)

    if sig.size < _safe_length_req(fs, min_seconds=2.0):
        return None

    # Interpolate NaNs if present
    if np.isnan(sig).any():
        idx = np.arange(len(sig))
        good = ~np.isnan(sig)
        if good.sum() < 2:
            return None
        sig = np.interp(idx, idx[good], sig[good])

    try:
        # Detrend signal and calculate PSD
        sig = sig - np.mean(sig)
        nperseg = min(256, len(sig))
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

        # Mask for HR frequency band (0.5 - 4.0 Hz)
        hr_mask = (freqs >= 0.5) & (freqs <= 4.0)
        if not np.any(hr_mask):
            return None

        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]
        # Find peak frequency in the HR band
        peak_idx = int(np.argmax(hr_psd))
        peak_freq = float(hr_freqs[peak_idx])
        # Convert frequency to BPM
        hr_bpm = float(peak_freq * 60.0)

        if HR_MIN_VALID <= hr_bpm <= HR_MAX_VALID:
            return hr_bpm
        return None

    except Exception as e:
        if DEBUG_MODE:
            print(f"Spectral HR calculation error: {e}")
        return None


def calculate_hr_hybrid(bvp_signal, fs, use_spectral_fallback=True):
    # Hybrid: prefer time-domain peaks, fall back to spectral.
    hr_bpm, nn_intervals = calculate_hr_from_peaks(bvp_signal, fs)
    if hr_bpm is None and use_spectral_fallback:
        hr_bpm = get_hr_from_spectrum(bvp_signal, fs)
        nn_intervals = None
    return hr_bpm, nn_intervals


def validate_heart_rate(hr_bpm, previous_hr=None, max_change=30):
    # Validate HR: absolute range + relative jump check.
    if hr_bpm is None:
        return False
    try:
        hr_val = float(hr_bpm)
    except Exception:
        return False
    if hr_val < HR_MIN_VALID or hr_val > HR_MAX_VALID:
        return False
    if previous_hr is not None:
        try:
            prev = float(previous_hr)
            # Check for sudden jumps
            if abs(hr_val - prev) > float(max_change):
                if DEBUG_MODE:
                    print(f"HR: Sudden change detected: {hr_val - prev:.1f} bpm")
                return False
        except Exception:
            pass
    return True


def smooth_heart_rate(current_hr, previous_hr, alpha=0.3):
    # EMA smoothing.
    if current_hr is None:
        return previous_hr
    if previous_hr is None:
        return float(current_hr)
    return float(alpha * float(current_hr) + (1.0 - alpha) * float(previous_hr))