# import cv2
# import numpy as np

# DEFAULT_SPACING = 12  
# DEFAULT_SKIN_HSV_LO = np.array([0, 10, 60])
# DEFAULT_SKIN_HSV_HI = np.array([25, 255, 255])

# def skin_mask_from_roi(roi_bgr):
#     if roi_bgr is None or roi_bgr.size == 0:
#         return None
#     hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, DEFAULT_SKIN_HSV_LO, DEFAULT_SKIN_HSV_HI)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return mask

# def sample_grid_in_bbox(bbox, spacing=DEFAULT_SPACING):
#     x_min, y_min, x_max, y_max = bbox
#     w = x_max - x_min
#     h = y_max - y_min
#     pts = []
#     if w <= 0 or h <= 0:
#         return pts
#     for yy in range(5, h-5, spacing):
#         for xx in range(5, w-5, spacing):
#             pts.append((xx, yy))
#     return pts

# def compute_frame_sample(roi_bgr, spacing=DEFAULT_SPACING, use_skin_mask=True):
#     if roi_bgr is None or roi_bgr.size == 0:
#         return None, 0.0, []

#     h, w = roi_bgr.shape[:2]
#     bbox = (0,0,w,h)
#     pts = sample_grid_in_bbox(bbox, spacing=spacing)
#     if len(pts) == 0:
#         return None, 0.0, []
#     mask = None
#     if use_skin_mask:
#         mask = skin_mask_from_roi(roi_bgr)
#     samples = []
#     sample_points = []
#     for (x, y) in pts:
#         valid = True
#         if mask is not None:
#             valid = bool(mask[y,x] > 0)
#         sample_points.append((x, y, valid))
#         if not valid:
#             continue
#         pixel = roi_bgr[y, x, :].astype(np.float32) 
#         r, g, b = pixel[2], pixel[1], pixel[0]
#         s = (r - g) / (r + g + b + 1e-8)  
#         samples.append(s)
#     if len(samples) == 0:
#         return None, 0.0, sample_points
#     sample_val = float(np.mean(samples))
#     valid_ratio = len(samples) / float(len(pts))
#     return sample_val, valid_ratio, sample_points

# class CHROMProcessor:
#     # Maintains a buffer of per-frame CHROM samples and provides windowed BVP segments.
#     def __init__(self, fs=30, spacing=12, window_sec=8):
#         self.fs = fs
#         self.spacing = spacing
#         self.window_len = int(window_sec * fs)
#         self.buffer = []
#         self.quality_buffer = []
#         self.sample_points_last = []  

#     def add_frame(self, roi_bgr):
#         sample_val, valid_ratio, sample_points = compute_frame_sample(
#             roi_bgr, spacing=self.spacing, use_skin_mask=True
#         )
#         self.sample_points_last = sample_points
#         if sample_val is None:
#             self.buffer.append(np.nan)
#             self.quality_buffer.append(0.0)
#         else:
#             self.buffer.append(sample_val)
#             self.quality_buffer.append(valid_ratio)
    
#         if len(self.buffer) > self.window_len * 4:
#             # cap to 4 windows for memory
#             self.buffer = self.buffer[-self.window_len*4:]
#             self.quality_buffer = self.quality_buffer[-self.window_len*4:]
#         return sample_val, valid_ratio, sample_points

#     def get_window_bvp(self, window_sec=None):
#         wl = self.window_len if window_sec is None else int(window_sec * self.fs)
#         if len(self.buffer) < wl:
#             return None
#         arr = np.array(self.buffer[-wl:], dtype=np.float32)

#         # simple linear interpolation of NaNs
#         if np.isnan(arr).any():
#             n = len(arr)
#             idx = np.arange(n)
#             good = ~np.isnan(arr)
#             if good.sum() < 2:
#                 return None
#             arr = np.interp(idx, idx[good], arr[good])
#         # detrend (remove mean)
#         arr = arr - np.mean(arr)
#         return arr

#     def get_quality(self):
#         if len(self.quality_buffer) == 0:
#             return 0.0
#         vals = np.array(self.quality_buffer[-self.window_len:], dtype=np.float32)
#         return float(np.nanmean(vals))

#     def get_sample_points_for_viz(self):
#         return self.sample_points_last


import cv2
import numpy as np

DEFAULT_SPACING = 12

def skin_mask_from_roi(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    
    # YCrCb skin thresholds 
    Y_MIN = 80
    CR_MIN = 135
    CR_MAX = 180
    CB_MIN = 85
    CB_MAX = 135
    
    # Create the mask
    mask = cv2.inRange(ycrcb, 
                       (Y_MIN, CR_MIN, CB_MIN), 
                       (255, CR_MAX, CB_MAX))
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def sample_grid_in_bbox(bbox, spacing=DEFAULT_SPACING):
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    pts = []
    if w <= 0 or h <= 0:
        return pts
    for yy in range(5, h-5, spacing):
        for xx in range(5, w-5, spacing):
            pts.append((xx, yy))
    return pts

def compute_frame_sample(roi_bgr, spacing=DEFAULT_SPACING, use_skin_mask=True):
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0, []

    h, w = roi_bgr.shape[:2]
    bbox = (0,0,w,h)
    pts = sample_grid_in_bbox(bbox, spacing=spacing)
    if len(pts) == 0:
        return None, 0.0, []
    mask = None
    if use_skin_mask:
        mask = skin_mask_from_roi(roi_bgr)
    samples = []
    sample_points = []
    for (x, y) in pts:
        valid = True
        if mask is not None:
            valid = bool(mask[y,x] > 0)
        sample_points.append((x, y, valid))
        if not valid:
            continue
        pixel = roi_bgr[y, x, :].astype(np.float32) 
        r, g, b = pixel[2], pixel[1], pixel[0]
        s = (r - g) / (r + g + b + 1e-8)  
        samples.append(s)
    if len(samples) == 0:
        return None, 0.0, sample_points
    sample_val = float(np.mean(samples))
    valid_ratio = len(samples) / float(len(pts))
    return sample_val, valid_ratio, sample_points

class CHROMProcessor:
    # Maintains a buffer of per-frame CHROM samples and provides windowed BVP segments
    def __init__(self, fs=30, spacing=12, window_sec=8):
        self.fs = fs
        self.spacing = spacing
        self.window_len = int(window_sec * fs)
        self.buffer = []
        self.quality_buffer = []
        self.sample_points_last = []  

    def add_frame(self, roi_bgr):
        sample_val, valid_ratio, sample_points = compute_frame_sample(
            roi_bgr, spacing=self.spacing, use_skin_mask=True
        )
        self.sample_points_last = sample_points
        if sample_val is None:
            self.buffer.append(np.nan)
            self.quality_buffer.append(0.0)
        else:
            self.buffer.append(sample_val)
            self.quality_buffer.append(valid_ratio)
    
        if len(self.buffer) > self.window_len * 4:
            
            self.buffer = self.buffer[-self.window_len*4:]
            self.quality_buffer = self.quality_buffer[-self.window_len*4:]
        return sample_val, valid_ratio, sample_points

    def get_window_bvp(self, window_sec=None):
        wl = self.window_len if window_sec is None else int(window_sec * self.fs)
        if len(self.buffer) < wl:
            return None
        arr = np.array(self.buffer[-wl:], dtype=np.float32)

        # simple linear interpolation of NaNs
        if np.isnan(arr).any():
            n = len(arr)
            idx = np.arange(n)
            good = ~np.isnan(arr)
            if good.sum() < 2:
                return None
            arr = np.interp(idx, idx[good], arr[good])
        
        arr = arr - np.mean(arr)
        return arr

    def get_quality(self):
        if len(self.quality_buffer) == 0:
            return 0.0
        vals = np.array(self.quality_buffer[-self.window_len:], dtype=np.float32)
        return float(np.nanmean(vals))

    def get_sample_points_for_viz(self):
        return self.sample_points_last
