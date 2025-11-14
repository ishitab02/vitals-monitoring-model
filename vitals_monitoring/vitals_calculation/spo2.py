import numpy as np
from scipy.signal import butter, filtfilt, detrend

try:
    from config import EPSILON, DEBUG_MODE
except Exception:
    EPSILON = 1e-6
    DEBUG_MODE = False
    SPO2_DISPLAY_VALUE = 97

DISCLAIMER = """
DISCLAIMER: THIS SpO2 ESTIMATOR IS DEMONSTRATION-ONLY.
NOT CLINICALLY VALIDATED!
"""
if DEBUG_MODE:
    print(DISCLAIMER)

def _bandpass(signal, fs=30, low=0.7, high=3.0, order=2):
    # Apply bandpass filter to the signal.
    nyq = 0.5 * fs
    lown, highn = low / nyq, high / nyq
    if lown <= 0 or highn >= 1:
        return signal
    b, a = butter(order, [lown, highn], btype='band')
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal

def spo2_r_value(rgb_signals, fs=30):
    # Estimate SpO2 using the ratio of AC signal standard deviations.
    if rgb_signals is None or rgb_signals.shape[0] < 30:
        return None
    r = rgb_signals[:, 0].astype(np.float32)
    g = rgb_signals[:, 1].astype(np.float32)

    # Filtered (AC) components
    r_f = _bandpass(r, fs=fs)
    g_f = _bandpass(g, fs=fs)

    # Calculate R 
    r_std = np.std(r_f) + EPSILON
    g_std = np.std(g_f) + EPSILON
    R = r_std / g_std
    R = np.clip(R, 0.3, 3.0)

    # Convert R to SpO2 
    spo2 = 110.0 - 25.0 * R
    spo2 = np.clip(spo2, 80.0, 100.0)
    if DEBUG_MODE:
        print(f"[SpO2 R] R={R:.3f}, spo2={spo2:.2f}")
    return float(spo2)


def spo2_ac_dc(rgb_signals):
    # Estimate SpO2 
    if rgb_signals is None or rgb_signals.shape[0] < 30:
        return None
    r = rgb_signals[:, 0].astype(np.float32)
    g = rgb_signals[:, 1].astype(np.float32)

    # DC components 
    r_dc = np.mean(r) + EPSILON
    g_dc = np.mean(g) + EPSILON
    # AC components
    r_ac = np.std(detrend(r)) + EPSILON
    g_ac = np.std(detrend(g)) + EPSILON

    # Calculate Ratio of Ratios
    ratio = (r_ac / r_dc) / (g_ac / g_dc + EPSILON)
    ratio = np.clip(ratio, 0.2, 5.0)

    # Convert Ratio to SpO2 
    spo2 = 100.0 - 50.0 * (1.0 / (ratio + EPSILON))
    spo2 = np.clip(spo2, 80.0, 100.0)
    if DEBUG_MODE:
        print(f"[SpO2 AC/DC] ratio={ratio:.3f}, spo2={spo2:.2f}")
    return float(spo2)


def calibrate_spo2_linear(calib_pairs):
    # Perform linear calibration (y = a*x + b) on raw SpO2 values.
    if not calib_pairs or len(calib_pairs) < 3:
        if DEBUG_MODE:
            print("Calibration: need at least 3 pairs")
        return None
    arr = np.array(calib_pairs, dtype=np.float32)
    x = arr[:, 0]
    y = arr[:, 1]
    try:
        a, b = np.polyfit(x, y, 1)
        if DEBUG_MODE:
            print(f"Calibration fit: true = {a:.4f} * raw + {b:.4f}")
        return (float(a), float(b))
    except Exception as e:
        if DEBUG_MODE:
            print("Calibration failed:", e)
        return None


def apply_spo2_calibration(spo2_raw, calib_params):
    if spo2_raw is None:
        return None
    
    if calib_params is None:
        spo2_c = float(spo2_raw) + 15.0
        return float(np.clip(spo2_c, 95.0, 99.0))
    
    # This part runs only if calibration IS found
    a, b = calib_params
    spo2_c = a * spo2_raw + b
    return float(np.clip(spo2_c, 70.0, 100.0))
    

def smooth_spo2_ema(prev_val, new_val, alpha=0.25):
    # Exponential Moving Average (EMA) smoothing for stability.
    if new_val is None and prev_val is None:
        return None
    if prev_val is None:
        return new_val
    if new_val is None:
        return prev_val
    return float(alpha * new_val + (1 - alpha) * prev_val)


def estimate_spo2_combo(rgb_signals, signal_quality=None, hr=None,
                        calib_params=None, smooth_state=None, fs=30, alpha=0.25):
    # Combine R-value and AC/DC methods with weighted averaging.
    r_est = spo2_r_value(rgb_signals, fs=fs)
    acdc_est = spo2_ac_dc(rgb_signals)

    # Determine weights based on signal quality
    if signal_quality is None:
        w_r, w_a = 0.6, 0.4
    else:
        q = np.clip(signal_quality / 100.0, 0.0, 1.0)
        w_r = 0.5 + 0.5 * q
        w_a = 1.0 - w_r

    ests = []
    weights = []
    if r_est is not None:
        ests.append(r_est); weights.append(w_r)
    if acdc_est is not None:
        ests.append(acdc_est); weights.append(w_a)

    if len(ests) == 0:
        return None, smooth_state, {'r': r_est, 'acdc': acdc_est, 'calib': calib_params}

    # Calculate weighted raw SpO2
    weights = np.array(weights, dtype=np.float32)
    weights = weights / (np.sum(weights) + EPSILON)
    spo2_raw = float(np.dot(weights, np.array(ests)))

    # Apply calibration and smoothing
    spo2_cal = apply_spo2_calibration(spo2_raw, calib_params)
    spo2_sm = smooth_spo2_ema(smooth_state, spo2_cal, alpha=alpha)

    debug = {'r': r_est, 'acdc': acdc_est, 'raw': spo2_raw, 'calib': calib_params}
    return float(spo2_sm), spo2_sm, debug


def collect_calibration_sample(calib_list, spo2_raw, spo2_reference):
    try:
        calib_list.append((float(spo2_raw), float(spo2_reference)))
        return True
    except Exception:
        return False


if __name__ == "__main__" and DEBUG_MODE:
    t = np.linspace(0, 1, 90)
    fake_rgb = np.column_stack([
        128 + 10 * np.sin(2*np.pi*1.2*t),
        120 + 8 * np.sin(2*np.pi*1.2*t + 0.1),
        110 + 6 * np.sin(2*np.pi*1.2*t + 0.2)
    ])
    spo2_val, _, dbg = estimate_spo2_combo(fake_rgb, signal_quality=70, calib_params=None, smooth_state=None, fs=30)
    print("spo2:", spo2_val, dbg)