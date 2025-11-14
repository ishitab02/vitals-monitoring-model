import cv2
import numpy as np
from datetime import datetime

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except Exception:
    MP_AVAILABLE = False

try:
    from config import (
        FONT, FONT_SCALE, FONT_THICKNESS,
        SHOW_FPS_COUNTER, SHOW_SIGNAL_QUALITY_METER,
        SHOW_BUFFER_FILL_STATE, COLOR_ROI_BOX, FONT_COLOR,
        COLOR_DOT_GOOD, COLOR_DOT_BAD, COLOR_WAVEFORM_BG, COLOR_WAVEFORM_LINE, 
         COLOR_WAVEFORM_PEAK, COLOR_GRID, COLOR_TITLE
    )
except Exception:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    SHOW_FPS_COUNTER = True
    SHOW_SIGNAL_QUALITY_METER = True
    SHOW_BUFFER_FILL_STATE = True
    DEBUG_MODE = False
    COLOR_WAVEFORM_BG = (0, 0, 0)
    COLOR_WAVEFORM_LINE = (0, 180, 0)
    COLOR_WAVEFORM_PEAK = (0, 0, 255)
    COLOR_GRID = (230, 230, 230)
    COLOR_TITLE = (50, 50, 50)

QUALITY_HRV_MIN = 65.0

# helpers
def _fmt(val, fmt="{:.1f}", none_str="--"):
    if val is None:
        return none_str
    try:
        return fmt.format(val)
    except Exception:
        return str(val)

def _text_pos(y_index, col_x, line_h=28, top_margin=18):
    return (col_x + 8, top_margin + y_index * line_h)

def _draw_side_panel(frame, items, width=240, alpha=0.78):
    h, w = frame.shape[:2]
    x0 = w - width
    panel = np.zeros((h, width, 3), dtype=np.uint8)
    panel[:] = (0, 0, 0)  
    frame[:, x0:w, :] = (frame[:, x0:w, :].astype(np.float32) * (1.0 - alpha) +
                         panel.astype(np.float32) * alpha).astype(np.uint8)


    for i, (text, small) in enumerate(items):
        fs = FONT_SCALE * (0.9 if small else 1.0)
        x, y = _text_pos(i, x0)
        cv2.putText(frame, text, (x, y), FONT, fs, (255, 255, 255), FONT_THICKNESS, lineType=cv2.LINE_AA)

    return frame

# face visualizations 
def draw_face_visuals_minimal(frame, face_landmarks, draw_roi_bbox=None):
    h, w = frame.shape[:2]

    if MP_AVAILABLE and hasattr(face_landmarks, "landmark"):
        try:

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=1)
            )
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(230,230,230), thickness=1, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(230,230,230), thickness=1)
            )
            # eyes, lips, brows colored white
            groups = [
                (mp_face_mesh.FACEMESH_LEFT_EYE, (230,230,230)),
                (mp_face_mesh.FACEMESH_RIGHT_EYE, (230,230,230)),
                (mp_face_mesh.FACEMESH_LIPS, (255,255,255)),
                (mp_face_mesh.FACEMESH_LEFT_EYEBROW, (230,230,230)),
                (mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (230,230,230)),
            ]
            for gm, col in groups:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, gm,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=col, thickness=1, circle_radius=0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=col, thickness=1)
                )
        except Exception:
            pass

        # Draw ROI bbox 
        if draw_roi_bbox is not None:
            try:
                x_min, y_min, x_max, y_max, label = draw_roi_bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_ROI_BOX, 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, label, (x_min, y_min - 8), FONT, 0.6, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            except Exception:
                pass

        return frame

    # Fallback if mediapipe not available
    if face_landmarks is None:
        return frame

    try:
        face_oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
            365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
            132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        pts = []
        for i in face_oval_idx:
            try:
                lm = face_landmarks[i]
                pts.append((int(lm.x * w), int(lm.y * h)))
            except Exception:
                continue
        if len(pts) > 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (230,230,230), 1, lineType=cv2.LINE_AA)
    except Exception:
        pass

    # fallback ROI bbox
    if draw_roi_bbox is not None:
        try:
            x_min, y_min, x_max, y_max, label = draw_roi_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,255,255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, label, (x_min, y_min - 8), FONT, 0.6, (255,255,255), 1, lineType=cv2.LINE_AA)
        except Exception:
            pass

    return frame

# sampling points
def draw_sampling_points(frame, roi_top_left, sample_points, color_good=COLOR_DOT_GOOD, color_bad=COLOR_DOT_BAD, max_draw=300):
    if sample_points is None or len(sample_points) == 0:
        return frame

    x0, y0 = roi_top_left
    h, w = frame.shape[:2]
    step = max(1, len(sample_points) // max_draw)
    r = 1
    for (x_local, y_local, valid) in sample_points[::step]:
        fx = int(round(x0 + x_local))
        fy = int(round(y0 + y_local))
        if fx < 0 or fy < 0 or fx >= w or fy >= h:
            continue
        col = color_good if valid else color_bad
        cv2.circle(frame, (fx, fy), r, col, -1, lineType=cv2.LINE_AA)
    return frame

# overlay
def draw_overlay(frame, buffer_fill, buffer_size,
                 hr, sdnn, rmssd, stress, quality,
                 fps=None, low_light=False, no_person=False):
    h, w = frame.shape[:2]
    items = []

    # timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items.append((timestamp, True))

    # buffer / fps
    if SHOW_BUFFER_FILL_STATE:
        buffer_pct = (buffer_fill / buffer_size) * 100 if buffer_size else 0
        items.append((f"Buffer: {int(buffer_fill)}/{int(buffer_size)} ({buffer_pct:.0f}%)", True))
    if SHOW_FPS_COUNTER and fps is not None:
        items.append((f"FPS: {fps:.1f}", True))

    items.append(("", True))

    # main vitals 
    items.append((f"HR: {_fmt(hr, '{:.0f}', '--')} bpm", False))

    # HRV 
    quality_val = quality if quality is not None else 0.0
    hrv_ok = quality_val >= QUALITY_HRV_MIN
    if not hrv_ok:
        items.append(("HRV: Unreliable", True))
    else:
        items.append((f"SDNN: {_fmt(sdnn, '{:.0f}', '--')} ms", True))
        items.append((f"RMSSD: {_fmt(rmssd, '{:.0f}', '--')} ms", True))
        #items.append((f"LF/HF: {_fmt(lf_hf, '{:.2f}', '--')}", True))

    items.append(("", True))

    frame = _draw_side_panel(frame, items, width=240, alpha=0.78)

    if hrv_ok and stress is not None:
        txt = f"Stress: {_fmt(stress, '{:.0f}', '--')}%"
    else:
        txt = "Stress: --"
    cv2.putText(frame, txt, (12, h - 44), FONT, 0.7, (255,255,255), 1, lineType=cv2.LINE_AA)

    if SHOW_SIGNAL_QUALITY_METER and quality is not None:
        cv2.putText(frame, f"Quality: {_fmt(quality, '{:.0f}', '--')}%", (12, h - 18),
                    FONT, 0.7, (255,255,255), 1, lineType=cv2.LINE_AA)

    # notify
    notify_y = 18
    if no_person:
        cv2.putText(frame, "NO FACE DETECTED", (12, notify_y), FONT, 0.65, (255,255,255), 1, lineType=cv2.LINE_AA)
        notify_y += 28
    if low_light:
        cv2.putText(frame, "LOW LIGHT!", (12, notify_y), FONT, 0.65, (255,255,255), 1, lineType=cv2.LINE_AA)
        notify_y += 28

    return frame

# waveform 
def draw_waveform(bvp_signal, peaks, title="BVP", width=640, height=160, bg_color=COLOR_WAVEFORM_BG):
    canvas = np.ones((height, width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)

    # Handle no-signal case
    if bvp_signal is None or (hasattr(bvp_signal, "__len__") and len(bvp_signal) == 0):
        cv2.putText(canvas, "No signal yet...", (width // 2 - 80, height // 2),
                    FONT, 0.6, (160, 160, 160), 1, lineType=cv2.LINE_AA)
        return canvas

    sig = np.array(bvp_signal, dtype=np.float32)

    # slight smoothing for nicer display
    if sig.size > 3:
        sig = np.convolve(sig, np.ones(3) / 3.0, mode="same")

    sig_min = np.min(sig)
    sig_max = np.max(sig)
    sig_range = max(sig_max - sig_min, 1e-6)

    margin = 12
    y_scaled = (1.0 - (sig - sig_min) / sig_range) * (height - 2*margin) + margin
    x_scaled = np.linspace(0, width - 1, len(sig)).astype(np.int32)

    pts = np.column_stack((x_scaled, y_scaled)).astype(np.int32)

    # Light gray horizontal grid lines
    for i in range(margin, height - margin, 40):
        cv2.line(canvas, (0, i), (width, i), (230, 230, 230), 1, lineType=cv2.LINE_AA)

    # waveform line
    if pts.shape[0] >= 2:
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=(0, 180, 0),  
                      thickness=1, lineType=cv2.LINE_AA)

    # RED peaks
    if peaks:
        for peak in peaks:
            if 0 <= peak < len(pts):
                px, py = int(pts[peak, 0]), int(pts[peak, 1])
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)   
                cv2.circle(canvas, (px, py), 3, (50,50,50), 1, lineType=cv2.LINE_AA)    

    # Title
    cv2.putText(canvas, title, (8, 18), FONT, 0.6, (50, 50, 50), 1, lineType=cv2.LINE_AA)

    return canvas
