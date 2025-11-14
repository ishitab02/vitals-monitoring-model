from typing import Tuple, Dict, Optional
import numpy as np
from scipy.signal import resample, find_peaks, welch

EPS = 1e-8

def _resample_to(sig: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    if sig is None:
        return None
    sig = np.asarray(sig, dtype=float)
    n_src = len(sig)
    if n_src < 2:
        return sig.copy()
    duration = n_src / float(src_fs)
    n_dst = max(1, int(round(duration * float(dst_fs))))
    if n_dst == n_src and abs(src_fs - dst_fs) < 1e-6:
        return sig.copy()
    # original time axis and new time axis
    t_src = np.linspace(0.0, duration, n_src)
    t_dst = np.linspace(0.0, duration, n_dst)
    try:
        dst = np.interp(t_dst, t_src, sig)
    except Exception:
        # fallback 
        dst = resample(sig, n_dst)
    return dst


def _signal_quality_metric(sig: np.ndarray, fs: float) -> float:
    if sig is None or len(sig) < 4:
        return 0.0
    try:
        sig = np.asarray(sig, dtype=float)
        sig = sig - np.mean(sig)
        # compute PSD
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        if len(psd) == 0:
            return 0.0
        # HR band (0.7 - 3.5 Hz)
        hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
        hr_power = np.trapz(psd[hr_mask], freqs[hr_mask]) if np.any(hr_mask) else 0.0
        total_power = np.trapz(psd, freqs) + EPS
        ratio = hr_power / total_power
        # map ratio to 0..1 
        q = (ratio - 0.02) / (0.3 - 0.02)
        q = float(np.clip(q, 0.0, 1.0))
        return q
    except Exception:
        return 0.0

def fuse_bvp_signal(bvp_a: Optional[np.ndarray], fs_a: float,
                    bvp_b: Optional[np.ndarray], fs_b: float,
                    out_fs: float) -> Tuple[Optional[np.ndarray], Dict]:
    
    # Fuse two BVP signals (a and b) into a single output at sampling rate out_fs.
    info = {'qa': 0.0, 'qb': 0.0, 'used_a': False, 'used_b': False}

    # Validate
    if (bvp_a is None or len(bvp_a) < 4) and (bvp_b is None or len(bvp_b) < 4):
        return None, info

    # Resample
    sig_a = _resample_to(bvp_a, fs_a, out_fs) if bvp_a is not None else None
    sig_b = _resample_to(bvp_b, fs_b, out_fs) if bvp_b is not None else None

    if sig_a is None and sig_b is not None:
        info['used_b'] = True
        info['qb'] = _signal_quality_metric(sig_b, out_fs)
        fused = sig_b - np.mean(sig_b)
        return fused, info
    if sig_b is None and sig_a is not None:
        info['used_a'] = True
        info['qa'] = _signal_quality_metric(sig_a, out_fs)
        fused = sig_a - np.mean(sig_a)
        return fused, info

    qa = _signal_quality_metric(sig_a, out_fs)
    qb = _signal_quality_metric(sig_b, out_fs)
    info['qa'] = float(qa)
    info['qb'] = float(qb)

    if qa < 0.01 and qb < 0.01:
        return None, info

    # Compute weights 
    power = 2.0
    wa = (qa + EPS) ** power
    wb = (qb + EPS) ** power
    wsum = wa + wb + EPS
    wa /= wsum
    wb /= wsum
    info['used_a'] = True
    info['used_b'] = True
    info['w_a'] = float(wa)
    info['w_b'] = float(wb)

    # Align lengths
    n = max(len(sig_a), len(sig_b))
    def _pad_or_trim(x, n):
        if len(x) == n:
            return x
        if len(x) < n:
            return np.concatenate([x, np.full(n - len(x), np.mean(x) if len(x)>0 else 0.0)])
        return x[-n:]
    sig_a2 = _pad_or_trim(sig_a, n)
    sig_b2 = _pad_or_trim(sig_b, n)

    # Normalize each to zero-mean (to avoid scale dominance)
    def norm0(x):
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        s = np.std(x) + EPS
        return x / s
    na = norm0(sig_a2)
    nb = norm0(sig_b2)

    fused = wa * na + wb * nb

    return fused, info


def detect_peaks_and_hr(bvp_signal: Optional[np.ndarray], fs: float,
                        min_distance_sec: float = 0.45, prominence: float = 0.15) -> Tuple[Optional[np.ndarray], Optional[float]]:
    
    # Detect peaks in BVP signal and compute HR (bpm).
    if bvp_signal is None:
        return None, None
    sig = np.asarray(bvp_signal, dtype=float)
    if sig.size < 4:
        return None, None

    # simple preprocessing
    sig = sig - np.mean(sig)
    if np.std(sig) > 0:
        sig = sig / (np.std(sig) + EPS)

    # convert min distance to samples
    min_dist_samp = max(1, int(round(min_distance_sec * float(fs))))

    try:
        peaks, props = find_peaks(sig, distance=min_dist_samp, prominence=prominence)
    except Exception:
        try:
            peaks, props = find_peaks(sig, distance=min_dist_samp)
        except Exception:
            return None, None

    if len(peaks) < 2:
        return peaks if len(peaks)>0 else None, None

    # compute HR from peak intervals
    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / float(fs)
    # filter invalid intervals
    valid = (rr_sec > 0.25) & (rr_sec < 3.0)  
    if valid.sum() < 1:
        return peaks, None
    rr_sec_valid = rr_sec[valid]
    mean_rr = float(np.mean(rr_sec_valid))
    hr_bpm = 60.0 / (mean_rr + EPS)

    return peaks, float(hr_bpm)

# expose public API
__all__ = [
    "fuse_bvp_signal",
    "detect_peaks_and_hr",
    "_resample_to",
]
