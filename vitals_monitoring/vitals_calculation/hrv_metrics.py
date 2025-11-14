import numpy as np
from scipy.signal import welch

try:
    from config import (
        LF_MIN, LF_MAX, HF_MIN, HF_MAX, WELCH_NPERSEG,
        DEBUG_MODE
    )
except ImportError:
    LF_MIN = 0.04
    LF_MAX = 0.15
    HF_MIN = 0.15
    HF_MAX = 0.4
    WELCH_NPERSEG = 256
    DEBUG_MODE = False

# TIME-DOMAIN HRV METRICS

def calculate_sdnn(nn_intervals_ms):
    # Calculate SDNN (Standard Deviation of NN-intervals).
    if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
        return None
    
    try:
        sdnn = np.std(nn_intervals_ms)
        return float(sdnn)
    except Exception as e:
        if DEBUG_MODE:
            print(f"SDNN calculation error: {e}")
        return None


def calculate_rmssd(nn_intervals_ms):
    # Calculate RMSSD (Root Mean Square of Successive Differences).
    if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
        return None
    
    try:
        successive_diffs = np.diff(nn_intervals_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        return float(rmssd)
    except Exception as e:
        if DEBUG_MODE:
            print(f"RMSSD calculation error: {e}")
        return None


def calculate_pnn50(nn_intervals_ms):
    # Calculate pNN50 (percentage of successive NN intervals > 50ms apart).
    if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
        return None
    
    try:
        successive_diffs = np.abs(np.diff(nn_intervals_ms))
        count_above_50 = np.sum(successive_diffs > 50)
        pnn50 = (count_above_50 / len(successive_diffs)) * 100
        return float(pnn50)
    except Exception as e:
        if DEBUG_MODE:
            print(f"pNN50 calculation error: {e}")
        return None


# FREQUENCY-DOMAIN HRV METRICS

def calculate_hrv_frequency_domain(nn_intervals_ms, fs_rr=None):
    # Calculate frequency-domain HRV metrics (LF, HF, LF/HF ratio).
    if nn_intervals_ms is None or len(nn_intervals_ms) < 10:
        return None
    
    try:
        # Convert ms to seconds
        nn_intervals_sec = nn_intervals_ms / 1000.0
        
        # Estimate approximate RR sampling rate if needed
        if fs_rr is None:
            mean_nn = np.mean(nn_intervals_sec)
            fs_rr = 1.0 / mean_nn
        
        # Resample to regular 4 Hz grid (standard for HRV)
        fs_rr_standard = 4.0
        n_samples = int(len(nn_intervals_sec) * (fs_rr_standard / fs_rr))
        
        # Create time vector and interpolate
        t_original = np.cumsum(nn_intervals_sec)
        t_resampled = np.linspace(0, t_original[-1], n_samples)
        rr_resampled = np.interp(t_resampled, t_original, nn_intervals_sec)
        
        # Welch's method for power spectral density
        freqs, psd = welch(
            rr_resampled,
            fs=fs_rr_standard,
            nperseg=min(WELCH_NPERSEG, len(rr_resampled))
        )
        
        # Extract LF and HF bands
        lf_mask = (freqs >= LF_MIN) & (freqs <= LF_MAX)
        hf_mask = (freqs >= HF_MIN) & (freqs <= HF_MAX)
        
        # Calculate power by integrating PSD (trapezoidal rule)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        
        # Calculate normalized powers
        total_power = lf_power + hf_power + 1e-6
        lf_norm = (lf_power / total_power) * 100
        hf_norm = (hf_power / total_power) * 100
        
        # Calculate LF/HF ratio
        lf_hf_ratio = lf_power / (hf_power + 1e-6)
        
        return {
            'lf': float(lf_power),
            'hf': float(hf_power),
            'lf_hf_ratio': float(lf_hf_ratio),
            'lf_norm': float(lf_norm),
            'hf_norm': float(hf_norm)
        }
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"Frequency domain HRV error: {e}")
        return None


def calculate_all_hrv_metrics(nn_intervals_ms):
    # Calculate all HRV metrics (time + frequency domain).
    if nn_intervals_ms is None or len(nn_intervals_ms) < 10:
        return {}
    
    metrics = {}
    
    # Time domain
    metrics['sdnn'] = calculate_sdnn(nn_intervals_ms)
    metrics['rmssd'] = calculate_rmssd(nn_intervals_ms)
    metrics['pnn50'] = calculate_pnn50(nn_intervals_ms)
    
    # Frequency domain
    freq_metrics = calculate_hrv_frequency_domain(nn_intervals_ms)
    if freq_metrics:
        metrics.update(freq_metrics)
    
    return metrics


def get_hrv_status(sdnn):
    # Classify HRV status based on SDNN.
    if sdnn is None:
        return "Unknown", (128, 128, 128)
    elif sdnn < 30:
        return "Low (Stressed)", (0, 0, 255)
    elif sdnn < 80:
        return "Fair", (0, 165, 255)
    elif sdnn < 150:
        return "Good", (0, 255, 0)
    else:
        return "Excellent", (255, 255, 0)