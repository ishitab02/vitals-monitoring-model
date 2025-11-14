import numpy as np
from scipy.signal import butter, filtfilt, detrend
import cv2

# Import config
try:
    from config import (
        BVP_FILTER_LOW, BVP_FILTER_HIGH, BVP_FILTER_ORDER,
        BVP_SMOOTH_WINDOW, USE_WINDOWED_NORMALIZATION,
        RPPG_COLOR_SPACE, EPSILON, ENABLE_ADAPTIVE_FILTER,
        ADAPTIVE_FILTER_MARGIN, DEBUG_MODE
    )
except ImportError:
    # Fallback defaults
    BVP_FILTER_LOW = 0.7
    BVP_FILTER_HIGH = 3.0
    BVP_FILTER_ORDER = 2
    BVP_SMOOTH_WINDOW = 5
    USE_WINDOWED_NORMALIZATION = True
    RPPG_COLOR_SPACE = 'RGB'
    EPSILON = 1e-6
    ENABLE_ADAPTIVE_FILTER = True
    ADAPTIVE_FILTER_MARGIN = 0.3
    DEBUG_MODE = False

# Core signal processing
def detrend_signal(signal):
    # Apply linear detrending to remove slow drifts.
    if signal is None or len(signal) == 0:
        return np.array([])
    
    try:
        return detrend(signal, type='linear')
    except Exception as e:
        if DEBUG_MODE:
            print(f"Detrend error: {e}")
        return np.array([])


def bandpass_filter(signal, fs, low=None, high=None, order=None):
    if signal is None or len(signal) == 0:
        return np.array([])

    if low is None:
        low = BVP_FILTER_LOW
    if high is None:
        high = BVP_FILTER_HIGH
    if order is None:
        order = BVP_FILTER_ORDER

    # Check for minimum required signal length
    min_len = max( int(fs * (2.0 / low)), 64 )
    if len(signal) < min_len:
        if DEBUG_MODE:
            print(f"bandpass: signal too short ({len(signal)} < {min_len}), returning raw detrended")
        return detrend_signal(signal)

    try:
        # Calculate normalized frequencies
        nyquist = 0.5 * fs
        low_norm = max(1e-6, low / nyquist)
        high_norm = min(0.999999, high / nyquist)
        if low_norm >= high_norm:
            if DEBUG_MODE:
                print(f"bandpass: invalid normalized freqs ({low_norm}, {high_norm})")
            return detrend_signal(signal)

        # Design and apply Butterworth filter
        b, a = butter(order, [low_norm, high_norm], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception as e:
        if DEBUG_MODE:
            print(f"Bandpass filter error: {e}")
        return detrend_signal(signal)

def smooth_signal(signal: np.ndarray, window_size: int = None) -> np.ndarray:
    # Apply moving average filter to smooth signal.
    if signal is None or len(signal) == 0:
        return np.array([])
    
    if window_size is None:
        window_size = BVP_SMOOTH_WINDOW
    
    window_size = max(1, int(window_size))
    if window_size == 1:
        return signal.copy()
    
    try:
        # Create convolution kernel
        kernel = np.ones(window_size) / float(window_size)
        # Apply padding and convolution
        pad_len = window_size // 2
        padded = np.pad(signal, (pad_len, pad_len), mode='reflect')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed
    except Exception as e:
        if DEBUG_MODE:
            print(f"Smoothing error: {e}")
        return signal.copy()

def normalize_rgb_signals(rgb_signals, windowed=None):
    # Normalize RGB signals with optional windowed approach.
    if rgb_signals is None or rgb_signals.shape[0] == 0:
        return np.array([])
    
    if windowed is None:
        windowed = USE_WINDOWED_NORMALIZATION
    
    try:
        if windowed:
            # Windowed (per-channel) normalization
            rgb_norm = np.zeros_like(rgb_signals)
            for ch in range(3):
                channel_data = rgb_signals[:, ch]
                ch_mean = np.mean(channel_data) + EPSILON
                ch_std = np.std(channel_data) + EPSILON
                rgb_norm[:, ch] = (channel_data - ch_mean) / ch_std
            return rgb_norm
        else:
            # Global normalization
            rgb_mean = np.mean(rgb_signals) + EPSILON
            rgb_std = np.std(rgb_signals) + EPSILON
            return (rgb_signals - rgb_mean) / rgb_std
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"RGB normalization error: {e}")
        return rgb_signals.copy()


def rgb_to_colorspace(rgb_signals, colorspace='RGB'):
    # Convert RGB to alternative color space (YCbCr, HSV, etc).
    if colorspace == 'RGB':
        return rgb_signals
    
    if colorspace == 'YCbCr':
        # YCbCr conversion
        R, G, B = rgb_signals[:, 0], rgb_signals[:, 1], rgb_signals[:, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 0.564 * (B - Y)
        Cr = 0.713 * (R - Y)
        return np.stack([Y, Cb, Cr], axis=1)
    
    elif colorspace == 'HSV':
        # Approximate HSV (simplified)
        R, G, B = rgb_signals[:, 0], rgb_signals[:, 1], rgb_signals[:, 2]
        max_val = np.maximum(np.maximum(R, G), B)
        min_val = np.minimum(np.minimum(R, G), B)
        V = max_val
        S = (max_val - min_val) / (max_val + EPSILON)
        return np.stack([S, V, (R + G + B) / 3], axis=1)
    
    else:
        if DEBUG_MODE:
            print(f"Unknown colorspace: {colorspace}. Using RGB.")
        return rgb_signals


# Signal quality metrics

def get_signal_snr(bvp_signal, fs):
    # Calculate Signal-to-Noise Ratio (SNR) in dB.
    if len(bvp_signal) < 30:
        return 0.0
    
    try:
        # Signal power (variance)
        signal_power = np.var(bvp_signal)
        
        # Noise power (variance of high-frequency component)
        noise_power = np.var(np.diff(bvp_signal))
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / (noise_power + EPSILON))
        else:
            snr_db = 40.0
        
        return float(snr_db)
    except Exception:
        return 0.0


def get_signal_stability(bvp_signal, fs, window_size=30):
    # Calculate signal stability (consistency across windows).
    if len(bvp_signal) < window_size * 2:
        return 0.5
    
    try:
        # Split into windows and compute variance
        windows = [bvp_signal[i:i+window_size] 
                   for i in range(0, len(bvp_signal) - window_size, window_size)]
        
        if len(windows) < 2:
            return 1.0
        
        # Calculate variance of variance across windows
        variances = [np.var(w) for w in windows]
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        # Calculate Coefficient of Variation (CV)
        cv = std_var / (mean_var + EPSILON)
        # Stability is inversely related to CV
        stability = 1.0 / (1.0 + cv)
        
        return np.clip(stability, 0, 1)
    except Exception:
        return 0.5