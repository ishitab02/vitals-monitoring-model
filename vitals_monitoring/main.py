import cv2
import numpy as np
import time
from collections import deque

try:
    from scipy.signal import butter, filtfilt, find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from vitals_monitoring.preprocessing.roi_extraction import ROIProcessor
from vitals_monitoring.vitals_calculation.tscan_model import TSCANInference
from vitals_monitoring.vitals_calculation.heart_rate import calculate_hr_hybrid
from vitals_monitoring.vitals_calculation.hrv_metrics import calculate_all_hrv_metrics
from vitals_monitoring.vitals_calculation.stress_index import calculate_stress_index
import config
from vitals_monitoring.overlay import draw_overlay, draw_waveform, draw_sampling_points, draw_face_visuals_minimal
from vitals_monitoring.vitals_calculation.spo2 import estimate_spo2_combo, collect_calibration_sample, calibrate_spo2_linear
from vitals_monitoring.data_logger import DataLogger

try:
    from vitals_monitoring.preprocessing.chrom import CHROMProcessor
except Exception:
    CHROMProcessor = None

class VitalsMonitor:
    def __init__(self):
        print("=" * 60)
        print("Initializing Live Vitals Monitoring System")
        print("=" * 60)
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.FALLBACK_CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open any camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WINDOW_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FS_TARGET)
        
        self.roi_processor = ROIProcessor(target_size=(config.TSCAN_INPUT_SIZE[0], config.TSCAN_INPUT_SIZE[1]))
        self.tscan = TSCANInference(model_path=config.TSCAN_MODEL_PATH, frame_depth=config.TSCAN_FRAME_DEPTH, img_size=config.TSCAN_INPUT_SIZE)
        self.chrom_proc = CHROMProcessor(fs=config.FS_TARGET) if CHROMProcessor is not None else None
        
        self.fs = config.FS_TARGET
        self.display_hr = None
        self.display_hrv = None
        self.display_stress = None
        self.display_spo2 = config.SPO2_DISPLAY_VALUE
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.spo2_calib_pairs = []
        self.spo2_calib_params = None
        self.spo2_smooth_state = None
        self.waveform_window = deque(maxlen=int(self.fs * 12))
        self.show_roi_debug = False
    
        self.data_logger = DataLogger(log_dir="data/logs", log_interval=1.0)

        print("Initialization complete. Controls: q=quit, d=toggle ROI debug, r=reset, c=calibrate sample")
        print("=" * 60)

    def smooth_value(self, new_value, old_value, alpha):
        if old_value is None or new_value is None:
            return new_value if new_value is not None else old_value
        return alpha * new_value + (1 - alpha) * old_value

    def reset_signal_buffer(self):
        self.tscan.reset()
        print("Signal buffer reset")

    def _detect_peaks_simple(self, sig):
        if sig is None:
            return []
        sig = np.asarray(sig, dtype=np.float32)
        n = len(sig)
        if n < 3:
            return []
        mean = sig.mean()
        std = sig.std() if sig.std() > 0 else 1.0
        thr = mean + 0.3 * std
        peaks = []
        for i in range(1, n - 1):
            if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thr:
                peaks.append(i)
        if len(peaks) > 200:
            step = max(1, len(peaks) // 200)
            peaks = peaks[::step]
        return peaks

    def _butter_bandpass(self, lowcut, highcut, fs, order=4):
        if not SCIPY_AVAILABLE:
            return None, None
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut=0.7, highcut=3.5, fs=None, order=4):
        if data is None:
            return None
        arr = np.asarray(data, dtype=np.float32)
        if arr.size < 3:
            return arr
        if SCIPY_AVAILABLE and fs is not None:
            b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
            if b is None:
                return arr
            try:
                y = filtfilt(b, a, arr, method="pad")
                return y
            except Exception:
                return arr
        # Fallback filter
        win = max(3, int(round((fs or 30) * 0.6)))
        kernel = np.ones(win) / win
        smooth = np.convolve(arr, kernel, mode='same')
        return arr - smooth

    def detect_peaks_adaptive(self, sig, fs):
        if sig is None:
            return []
        arr = np.asarray(sig, dtype=np.float32)
        n = len(arr)
        if n < 3:
            return []
        if SCIPY_AVAILABLE and fs is not None:
            min_dist = int(round(0.4 * fs))
            prom = max(0.05 * (np.max(arr) - np.min(arr)), 0.15 * np.std(arr))
            try:
                peaks, props = find_peaks(arr, distance=min_dist, prominence=prom)
                return peaks.tolist()
            except Exception:
                pass
        # Fallback peak detection
        mean = arr.mean()
        std = arr.std() if arr.std() > 0 else 1.0
        thr = mean + 0.25 * std
        peaks = []
        for i in range(1, n - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > thr:
                peaks.append(i)
        min_samples = int(round(0.35 * (fs or 30)))
        if min_samples > 1 and len(peaks) > 1:
            filtered = [peaks[0]]
            for p in peaks[1:]:
                if p - filtered[-1] >= min_samples:
                    filtered.append(p)
            peaks = filtered
        return peaks

    def estimate_rr_and_stability(self, wf, fs):
        if wf is None or len(wf) < int(fs * 4):
            return None, 0.0
        arr = np.asarray(wf, dtype=np.float32)
        duration_s = len(arr) / float(fs)
        
        # 1) Stability: SNR-like score using cardiac band (0.7-3.5 Hz)
        try:
            cardiac = self.butter_bandpass_filter(arr, lowcut=0.7, highcut=3.5, fs=fs, order=4)
            if cardiac is None:
                cardiac = arr.copy()
        except Exception:
            cardiac = arr.copy()
        residual = arr - cardiac
        var_c = float(np.var(cardiac))
        var_r = float(np.var(residual))
        stability = 100.0 * var_c / (var_c + var_r + 1e-8)
        stability = float(np.clip(stability, 0.0, 100.0))
        
        # 2) Respiration estimation: bandpass 0.1 - 0.5 Hz
        try:
            resp = self.butter_bandpass_filter(arr, lowcut=0.1, highcut=0.5, fs=fs, order=4)
        except Exception:
            # fallback: slow moving average subtraction
            kernel_len = max(3, int(round(fs * 1.0)))
            kernel = np.ones(kernel_len) / kernel_len
            resp = np.convolve(arr, kernel, mode='same')
            resp = arr - resp
        
        resp = resp - np.mean(resp)
        if np.std(resp) < 1e-6:
            return None, float(stability)
        
        rr_peaks = []
        try:
            if SCIPY_AVAILABLE:
                min_dist = int(round(0.6 * fs))
                prom = max(0.05 * (np.max(resp) - np.min(resp)), 0.1 * np.std(resp))
                rr_peaks, props = find_peaks(resp, distance=min_dist, prominence=prom)
                rr_peaks = rr_peaks.tolist()
            else:
                thr = 0.25 * np.std(resp)
                for i in range(1, len(resp) - 1):
                    if resp[i] > resp[i - 1] and resp[i] > resp[i + 1] and resp[i] > thr:
                        rr_peaks.append(i)
                # enforce min distance ~0.6s
                min_samples = int(round(0.6 * fs))
                if min_samples > 1 and len(rr_peaks) > 1:
                    filtered = [rr_peaks[0]]
                    for p in rr_peaks[1:]:
                        if p - filtered[-1] >= min_samples:
                            filtered.append(p)
                    rr_peaks = filtered
        except Exception:
            rr_peaks = []
        
        rr_bpm = None
        if len(rr_peaks) >= 2:
            rr_bpm = (len(rr_peaks) / duration_s) * 60.0
            if rr_bpm < 6 or rr_bpm > 40:
                rr_bpm = None
        return (float(rr_bpm) if rr_bpm is not None else None), float(stability)

    def run(self):
        print("Starting main loop! Position your face in front of camera...")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                result = self.roi_processor.process_frame(frame)
                roi_quality = 0.0
                roi_bbox = None
                chrom_bvp = None
                bvp_signal = None
                bvp_for_hr = None
                hrv_metrics = None
                spo2_debug = None
                
                if result['valid']:
                    roi_quality = result['quality']['quality_score']
                    roi_bbox = result['bbox']
                    roi_for_tscan = result['roi']
                    
                    try:
                        self.tscan.add_frame(roi_for_tscan)
                    except Exception:
                        pass
                    
                    if self.tscan.is_ready():
                        _ = self.tscan.predict()
                        bvp_signal = np.array(self.tscan.get_bvp_buffer(length=int(self.fs * 12)))
                    else:
                        bvp_signal = np.array(self.tscan.get_bvp_buffer())
                    
                    if self.chrom_proc is not None:
                        try:
                            self.chrom_proc.add_frame(result['roi'])
                            chrom_bvp = self.chrom_proc.get_window_bvp(window_sec=8)
                        except Exception:
                            chrom_bvp = None
                    
                    if bvp_signal is not None and len(bvp_signal) >= int(self.fs * 5):
                        bvp_for_hr = bvp_signal
                    elif chrom_bvp is not None and len(chrom_bvp) >= int(self.fs * 5):
                        bvp_for_hr = chrom_bvp
                    else:
                        bvp_for_hr = None
                    
                    if bvp_for_hr is not None:
                        try:
                            filtered_bvp = self.butter_bandpass_filter(bvp_for_hr, lowcut=0.7, highcut=3.5, fs=self.fs, order=4)
                        except Exception:
                            filtered_bvp = np.array(bvp_for_hr, dtype=np.float32)
                        chunk = list(np.array(filtered_bvp[-int(self.fs * 5):], dtype=np.float32))
                        self.waveform_window.extend(chunk)
                    else:
                        filtered_bvp = None
                    
                    nn_intervals = None
                    if bvp_for_hr is not None and len(bvp_for_hr) >= int(self.fs * 5):
                        try:
                            hr, nn_intervals = calculate_hr_hybrid(bvp_for_hr, self.fs)
                        except Exception:
                            hr = None
                            nn_intervals = None
                        if hr is not None:
                            self.display_hr = self.smooth_value(hr, self.display_hr, config.EMA_HR)
                    
                    if nn_intervals is not None and len(nn_intervals) >= 5:
                        hrv_metrics = calculate_all_hrv_metrics(nn_intervals)
                        if 'sdnn' in hrv_metrics:
                            self.display_hrv = self.smooth_value(hrv_metrics['sdnn'], self.display_hrv, config.EMA_HRV)
                    
                    if hrv_metrics:
                        stress = calculate_stress_index(sdnn=hrv_metrics.get('sdnn'), lf_hf_ratio=hrv_metrics.get('lf_hf_ratio'))
                        self.display_stress = self.smooth_value(stress, self.display_stress, config.EMA_STRESS)
                    
                    rgb_window = None
                    try:
                        roi_hist = self.roi_processor.roi_history if hasattr(self.roi_processor, 'roi_history') else None
                        if roi_hist and len(roi_hist) >= int(self.fs * 4):
                            N = min(len(roi_hist), int(self.fs * 8))
                            arr = np.stack([cv2.resize(f, (32, 32)).mean(axis=(0, 1)) for f in list(roi_hist)[-N:]], axis=0)
                            rgb_window = arr[:, ::-1]
                    except Exception:
                        rgb_window = None
                    
                    if rgb_window is not None:
                        try:
                            spo2_val, self.spo2_smooth_state, spo2_debug = estimate_spo2_combo(
                                rgb_window,
                                signal_quality=roi_quality,
                                hr=self.display_hr,
                                calib_params=self.spo2_calib_params,
                                smooth_state=self.spo2_smooth_state,
                                fs=self.fs,
                                alpha=config.EMA_SPO2
                            )
                            if spo2_val is not None:
                                self.display_spo2 = spo2_val
                        except Exception:
                            spo2_debug = None
                
                vis_frame = frame.copy()
                face_landmarks = result.get('face_landmarks', None)
                
                if roi_bbox is not None:
                    x_min, y_min, x_max, y_max = roi_bbox
                    vis_frame = draw_face_visuals_minimal(vis_frame, face_landmarks,
                                                          draw_roi_bbox=(x_min, y_min, x_max, y_max, result.get('roi_name', 'ROI')))
                else:
                    vis_frame = draw_face_visuals_minimal(vis_frame, face_landmarks, draw_roi_bbox=None)
                
                if self.chrom_proc is not None and roi_bbox is not None:
                    sample_points = self.chrom_proc.get_sample_points_for_viz()
                    if sample_points:
                        vis_frame = draw_sampling_points(vis_frame, (roi_bbox[0], roi_bbox[1]), sample_points)
                
                try:
                    wf = np.array(self.waveform_window, dtype=np.float32)
                    peaks = self.detect_peaks_adaptive(wf, fs=self.fs) if len(wf) > int(self.fs * 3) else []
                except Exception:
                    wf = np.array(self.waveform_window, dtype=np.float32)
                    peaks = []
                
                # Only calculate stability, skip RR
                try:
                    _, stability_score = self.estimate_rr_and_stability(wf, self.fs)
                except Exception:
                    stability_score = 0.0
                
                waveform_img = draw_waveform(wf, peaks, title="BVP")
                
                # Log only the final, clean metrics
                try:
                    self.data_logger.log_measurement(
                        hr=self.display_hr,
                        hrv_sdnn=(hrv_metrics.get('sdnn') if hrv_metrics else None),
                        spo2=self.display_spo2,
                        stress=self.display_stress,
                        signal_quality=stability_score
                    )
                except Exception:
                    # don't let logging errors break the main loop
                    pass
                
                # Pass only clean metrics to the overlay
                vis_frame = draw_overlay(vis_frame,
                                         buffer_fill=len(self.waveform_window),
                                         buffer_size=self.waveform_window.maxlen,
                                         hr=self.display_hr,
                                         sdnn=self.display_hrv,
                                         rmssd=(hrv_metrics.get('rmssd') if hrv_metrics else None),
                                         stress=self.display_stress,
                                         quality=float(stability_score),
                                         fps=self.fps,
                                         low_light=(stability_score < 30 if stability_score is not None else False),
                                         no_person=(result['roi'] is None)) 
                
                try:
                    spo2_text = f"SpO2: {_fmt(self.display_spo2, '{:.0f}', '--')}%"
                    cv2.putText(vis_frame, spo2_text, (12, vis_frame.shape[0] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                except Exception:
                    pass
                
                cv2.imshow('Vitals Monitor', vis_frame)
                cv2.imshow('Waveform', waveform_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.show_roi_debug = not self.show_roi_debug
                    print("ROI debug:", self.show_roi_debug)
                elif key == ord('r'):
                    self.reset_signal_buffer()
                elif key == ord('c'):
                    # Calibration capture
                    if 'spo2_debug' in locals() and spo2_debug is not None and spo2_debug.get('raw') is not None:
                        raw = spo2_debug.get('raw')
                        print(f"\nCalibration capture: raw_spo2={raw:.2f}")
                        try:
                            ref = input("Enter reference SpO2 from pulse-ox (e.g. 98): ").strip()
                            refv = float(ref)
                            collect_calibration_sample(self.spo2_calib_pairs, raw, refv)
                            print("Added calib pair. total:", len(self.spo2_calib_pairs))
                            if len(self.spo2_calib_pairs) >= 4:
                                self.spo2_calib_params = calibrate_spo2_linear(self.spo2_calib_pairs)
                                print("New calibration params:", self.spo2_calib_params)
                        except Exception as e:
                            print("Calibration input aborted:", e)
                    else:
                        print("No valid raw SpO2 available to calibrate (need rgb window).")
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        try:
            # Save logs and analysis
            try:
                self.data_logger.save_and_analyze()
            except Exception:
                pass
        except Exception:
            pass
        self.cap.release()
        self.roi_processor.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

def _fmt(val, fmt="{:.1f}", none_str="--"):
    if val is None:
        return none_str
    try:
        return fmt.format(val)
    except Exception:
        return str(val)

def main():
    try:
        monitor = VitalsMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print("Fatal error:", e)

if __name__ == "__main__":
    main()