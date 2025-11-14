import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# fallback
try:
    from config import (
        FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES,
        HEAD_YAW_THRESHOLD, ROI_COLORS, MEDIAPIPE_STATIC_IMAGE_MODE,
        MEDIAPIPE_MAX_NUM_FACES, MEDIAPIPE_REFINE_LANDMARKS,
        MEDIAPIPE_MIN_DETECTION_CONFIDENCE, MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        DEBUG_MODE
    )
except Exception:
    FOREHEAD_INDICES = [70, 69, 67, 105, 107, 297, 299, 300]
    LEFT_CHEEK_INDICES = [117, 118, 119, 120, 147, 213, 205]
    RIGHT_CHEEK_INDICES = [346, 347, 348, 349, 376, 433, 425]
    HEAD_YAW_THRESHOLD = 0.06
    ROI_COLORS = {"forehead": (0, 0, 255), "left_cheek": (0, 0, 255), "right_cheek": (0, 0, 255)}
    MEDIAPIPE_STATIC_IMAGE_MODE = False
    MEDIAPIPE_MAX_NUM_FACES = 1
    MEDIAPIPE_REFINE_LANDMARKS = True
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
    DEBUG_MODE = False

try:
    mp_face_mesh = mp.solutions.face_mesh
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


class ROIDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_faces=MEDIAPIPE_MAX_NUM_FACES,
            refine_landmarks=MEDIAPIPE_REFINE_LANDMARKS,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )

        self.roi_indices = {
            "forehead": FOREHEAD_INDICES,
            "left_cheek": LEFT_CHEEK_INDICES,
            "right_cheek": RIGHT_CHEEK_INDICES
        }

        self.face_detected = False
        self.landmarks = None
        self.landmarks_list = None
        self.frame_size = None

        self.roi_stability_buffer = deque(maxlen=5)
        self.current_roi_name = "forehead"

        if DEBUG_MODE:
            print("[ROIDetector] Initialized MediaPipe FaceMesh")

    def _landmark_obj_to_list(self):
        if self.landmarks is None:
            return None
        try:
            return self.landmarks.landmark
        except Exception:
            return self.landmarks

    def get_face_landmarks_object(self):
        return self.landmarks

    def detect_face(self, frame):
        
        # Detect face and store landmarks.
       
        if frame is None:
            return False

        self.frame_size = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results and results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]
            self.landmarks_list = self._landmark_obj_to_list()
            self.face_detected = True
            if DEBUG_MODE:
                print("[ROIDetector] Face detected: landmarks stored")
            return True

        self.face_detected = False
        self.landmarks = None
        self.landmarks_list = None
        if DEBUG_MODE:
            print("[ROIDetector] No face detected")
        return False

    def get_landmark_coords(self, landmark_idx):
        """
        Return pixel coords (x, y) for a landmark index.
        """
        if not self.face_detected or self.landmarks is None or self.frame_size is None:
            return None

        h, w = self.frame_size

        try:
            if hasattr(self.landmarks, "landmark"):
                lm = self.landmarks.landmark[landmark_idx]
            else:
                lm = self.landmarks[landmark_idx]
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            return (x, y)
        except Exception:
            return None

  
    # ROI extraction 
    
    def extract_roi_from_landmarks(self, frame, landmark_indices, padding=0.15):
        """
        Compute a bounding box from landmarks, add padding, and return
        the cropped ROI image and bbox (x_min, y_min, x_max, y_max).
        """
        if not self.face_detected or self.landmarks is None:
            return None, None

        h, w = self.frame_size
        points = []
        for idx in landmark_indices:
            coord = self.get_landmark_coords(idx)
            if coord is not None:
                points.append(coord)

        if len(points) < 3:
            return None, None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - padding * width))
        x_max = min(w, int(x_max + padding * width))
        y_min = max(0, int(y_min - padding * height))
        y_max = min(h, int(y_max + padding * height))

        if x_max <= x_min or y_max <= y_min:
            return None, None

        roi = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)
        return roi, bbox

    # Head pose & ROI selection
    
    def calculate_head_pose(self):
        
        # Simple yaw-based head pose estimate.
        # Returns dict with 'yaw' and 'facing_forward' keys, or None on failure.
        
        if not self.face_detected:
            return None

        try:
            nose = self.get_landmark_coords(1)
            left_eye = self.get_landmark_coords(33)
            right_eye = self.get_landmark_coords(263)
            if nose is None or left_eye is None or right_eye is None:
                return None

            eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
            face_width = max(1.0, abs(right_eye[0] - left_eye[0]))
            yaw = (nose[0] - eye_center_x) / face_width

            facing_forward = abs(yaw) < HEAD_YAW_THRESHOLD
            return {"yaw": yaw, "facing_forward": facing_forward}
        except Exception:
            return None

    def select_best_roi(self, frame):
    
        # Choose one ROI based on head pose and stability buffer.
        if not self.face_detected:
            return None, None, None

        pose = self.calculate_head_pose()
        if pose is None:
            chosen = "forehead"
        else:
            if pose["facing_forward"]:
                chosen = "forehead"
            else:
                # yaw positive -> face turned right -> choose left_cheek
                chosen = "left_cheek" if pose["yaw"] > 0 else "right_cheek"

        roi, bbox = self.extract_roi_from_landmarks(frame, self.roi_indices.get(chosen, FOREHEAD_INDICES))

        # maintain stability buffer
        self.roi_stability_buffer.append(chosen)
        if len(self.roi_stability_buffer) >= 3:
            counts = {}
            for n in self.roi_stability_buffer:
                counts[n] = counts.get(n, 0) + 1
            stable_choice = max(counts, key=counts.get)
            if stable_choice != chosen:
                alt_roi, alt_bbox = self.extract_roi_from_landmarks(frame, self.roi_indices.get(stable_choice, FOREHEAD_INDICES))
                if alt_roi is not None:
                    roi, bbox = alt_roi, alt_bbox
                    chosen = stable_choice

        self.current_roi_name = chosen
        return roi, chosen, bbox
    
    # ROI quality
   
    def calculate_roi_quality(self, roi):
        # Compute brightness, contrast, size, and aggregate quality score [0..100].
        
        if roi is None or roi.size == 0:
            return {"brightness": 0.0, "contrast": 0.0, "size": 0, "quality_score": 0.0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        size = int(roi.shape[0] * roi.shape[1])

        # brightness score: optimal around 80-180
        brightness_score = 100.0 if 80 <= brightness <= 180 else max(0.0, 100.0 - abs(brightness - 130.0))
        contrast_score = min(100.0, contrast * 2.0)
        size_score = min(100.0, (size / 10000.0) * 100.0)

        quality_score = brightness_score * 0.4 + contrast_score * 0.3 + size_score * 0.3
        quality_score = float(np.clip(quality_score, 0.0, 100.0))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "size": size,
            "quality_score": quality_score
        }

    # Convenience utilities
    def get_all_rois(self, frame):
        if not self.face_detected:
            return {}

        rois = {}
        for name, indices in self.roi_indices.items():
            roi, bbox = self.extract_roi_from_landmarks(frame, indices)
            if roi is not None:
                q = self.calculate_roi_quality(roi)
                rois[name] = {"image": roi, "bbox": bbox, "quality": q}
        return rois

    def draw_landmarks(self, frame, draw_mesh=True, draw_roi=True):
       
        #Draw landmarks and ROI box on frame for visualization.
        if not self.face_detected:
            return frame

        h, w = self.frame_size

        if draw_mesh and MP_AVAILABLE and self.landmarks is not None:
            try:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    self.landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=0),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(200,200,200), thickness=1)
                )
            except Exception:
                draw_mesh = False

        if not draw_mesh:
            try:
                for lm in self.landmarks_list:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)
            except Exception:
                pass

        if draw_roi:
            roi, roi_name, bbox = self.select_best_roi(frame)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                color = ROI_COLORS.get(roi_name, (255, 255, 255))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"{roi_name}", (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, lineType=cv2.LINE_AA)

        return frame

    def is_roi_valid(self, quality_threshold=40.0):
        return self.face_detected

    def release(self):
        try:
            if hasattr(self, "face_mesh") and self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass


# High-level processor wrapper
class ROIProcessor:

    def __init__(self, target_size=(128, 128)):
        self.detector = ROIDetector()
        self.target_size = target_size
        self.roi_history = deque(maxlen=300)

    def process_frame(self, frame, resize=True, normalize=False):
        # Detect face, pick best ROI and preprocess it.
        face_ok = self.detector.detect_face(frame)
        if not face_ok:
            return {'roi': None, 'roi_name': None, 'bbox': None, 'quality': None, 'valid': False, 'face_landmarks': None}

        roi, roi_name, bbox = self.detector.select_best_roi(frame)
        quality = None
        valid = False
        out_roi = None

        if roi is not None:
            quality = self.detector.calculate_roi_quality(roi)
            valid = quality['quality_score'] >= 40.0

            out_roi = roi
            if resize and out_roi is not None:
                out_roi = cv2.resize(out_roi, self.target_size, interpolation=cv2.INTER_LINEAR)
            if normalize and out_roi is not None:
                out_roi = out_roi.astype(np.float32) / 255.0

            self.roi_history.append(out_roi)

        face_landmarks_obj = self.detector.get_face_landmarks_object()

        return {
            'roi': out_roi,
            'roi_name': roi_name,
            'bbox': bbox,
            'quality': quality,
            'valid': valid,
            'face_landmarks': face_landmarks_obj
        }

    def get_roi_for_tscan(self, frame):
        # Return ROI image suitable for TS-CAN (not resized/normalized by default)
        
        res = self.process_frame(frame, resize=False, normalize=False)
        if res['valid']:
            return res['roi']
        return None

    def visualize(self, frame, show_all=False):
        # Draw visualization over the frame.
        return self.detector.draw_landmarks(frame, draw_mesh=True, draw_roi=True)

    def release(self):
        self.detector.release()

# small test runner (manual)
def _test():
    cap = cv2.VideoCapture(0)
    rp = ROIProcessor(target_size=(128, 128))
    show_all = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            info = rp.process_frame(frame)
            vis = rp.visualize(frame.copy(), show_all=show_all)
            quality_score = info['quality']['quality_score'] if info['quality'] else '--'
            text = f"ROI: {info['roi_name']} Q:{quality_score:.1f}" if info['roi_name'] else "No face detected"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, lineType=cv2.LINE_AA)
            cv2.imshow("ROI test", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                show_all = not show_all
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rp.release()


if __name__ == "__main__":
    _test()