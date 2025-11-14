from collections import deque
import os
import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception as e:
    torch = None
    nn = None
try:
    from config import (
        TSCAN_MODEL_PATH,
        TSCAN_FRAME_DEPTH,
        TSCAN_INPUT_SIZE,
        DEBUG_MODE,
        TSCAN_USE_GPU,
        BUFFER_SIZE,
    )
except Exception:
    TSCAN_MODEL_PATH = "models/UBFC_TSCAN.pth"
    TSCAN_BATCH_SIZE = 1
    TSCAN_FRAME_DEPTH = 10
    TSCAN_INPUT_SIZE = (36, 36)
    DEBUG_MODE = False
    TSCAN_USE_GPU = False
    BUFFER_SIZE = 300
def _debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print("[TSCAN]", *args, **kwargs)
def _clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if isinstance(k, str) and k.startswith("module."):
            new_k = k[len("module.") :]
        cleaned[new_k] = v
    return cleaned

if torch is not None and nn is not None:
    # class TSCAN(nn.Module):
    #     def __init__(self, frame_depth=10, img_size=36):
    #         super().__init__()
    #         self.frame_depth = frame_depth
    #         self.img_size = img_size
    #         self.motion_conv1 = nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
    #         self.motion_bn1 = nn.BatchNorm3d(16)
    #         self.motion_conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    #         self.motion_bn2 = nn.BatchNorm3d(32)
    #         self.motion_conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    #         self.motion_bn3 = nn.BatchNorm3d(64)
    #         self.appear_conv1 = nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
    #         self.appear_bn1 = nn.BatchNorm3d(16)
    #         self.appear_conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    #         self.appear_bn2 = nn.BatchNorm3d(32)
    #         self.appear_conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    #         self.appear_bn3 = nn.BatchNorm3d(64)
    #         self.attention_conv = nn.Conv3d(128, 1, kernel_size=1)
    #         self.pool = nn.AdaptiveAvgPool3d((frame_depth, 1, 1))
    #         self.dropout = nn.Dropout(0.25)
    #         self.fc = nn.Linear(64 * frame_depth, frame_depth)
    #         self.relu = nn.ReLU(inplace=True)
    #         self.tanh = nn.Tanh()
    #     def forward(self, x):
    #         diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    #         pad = torch.zeros_like(diff[:, :, :1, :, :])
    #         diff = torch.cat([diff, pad], dim=2)
    #         m = self.relu(self.motion_bn1(self.motion_conv1(diff)))
    #         m = self.relu(self.motion_bn2(self.motion_conv2(m)))
    #         m = self.relu(self.motion_bn3(self.motion_conv3(m)))
    #         a = self.relu(self.appear_bn1(self.appear_conv1(x)))
    #         a = self.relu(self.appear_bn2(self.appear_conv2(a)))
    #         a = self.relu(self.appear_bn3(self.appear_conv3(a)))
    #         combined = torch.cat([m, a], dim=1)
    #         att = torch.sigmoid(self.attention_conv(combined))
    #         attended = a * att
    #         pooled = self.pool(attended)
    #         pooled = pooled.view(pooled.size(0), -1)
    #         pooled = self.dropout(pooled)
    #         out = self.tanh(self.fc(pooled))
    #         return out
    
    class TSCAN(nn.Module):
        def __init__(self, frame_depth=10, img_size=36):
            super().__init__()
            self.frame_depth = frame_depth
            self.img_size = img_size
            
            self.motion_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.motion_bn1 = nn.BatchNorm2d(32)
            self.motion_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.motion_bn2 = nn.BatchNorm2d(32)
            self.motion_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.motion_bn3 = nn.BatchNorm2d(64)

            self.appear_conv1 = nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
            self.appear_bn1 = nn.BatchNorm3d(16)
            self.appear_conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.appear_bn2 = nn.BatchNorm3d(32)
            self.appear_conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.appear_bn3 = nn.BatchNorm3d(64)

            self.attention_conv = nn.Conv3d(128, 1, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool3d((frame_depth, 1, 1))
            self.dropout = nn.Dropout(0.25)
            self.fc = nn.Linear(64 * frame_depth, frame_depth)
            self.relu = nn.ReLU(inplace=True)
            self.tanh = nn.Tanh()

        def forward(self, x):
            diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            pad = torch.zeros_like(diff[:, :, :1, :, :])
            diff = torch.cat([diff, pad], dim=2)
            
            B, C, D, H, W = diff.shape
            m = diff.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            
            m = self.relu(self.motion_bn1(self.motion_conv1(m)))
            m = self.relu(self.motion_bn2(self.motion_conv2(m)))
            m = self.relu(self.motion_bn3(self.motion_conv3(m)))
            
            _, C_out, H_out, W_out = m.shape
            m = m.view(B, D, C_out, H_out, W_out)
            m = m.permute(0, 2, 1, 3, 4)

            a = self.relu(self.appear_bn1(self.appear_conv1(x)))
            a = self.relu(self.appear_bn2(self.appear_conv2(a)))
            a = self.relu(self.appear_bn3(self.appear_conv3(a)))

            combined = torch.cat([m, a], dim=1) 
            
            att = torch.sigmoid(self.attention_conv(combined))
            attended = a * att
            pooled = self.pool(attended)
            pooled = pooled.view(pooled.size(0), -1)
            pooled = self.dropout(pooled)
            out = self.tanh(self.fc(pooled))
            return out
        
else:
    TSCAN = None
    _debug_print("torch not available: TSCAN model class disabled - using fallback!")
class TSCANInference:
    def __init__(self, model_path=None, device=None, frame_depth=None, img_size=None, normalize="-1_1"):
        self.frame_depth = int(frame_depth or TSCAN_FRAME_DEPTH)
        self.img_size = tuple(img_size or TSCAN_INPUT_SIZE)
        self.normalize = normalize
        if device is None:
            if torch is not None and TSCAN_USE_GPU:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu") if torch is not None else None
        else:
            self.device = torch.device(device) if torch is not None else None
        _debug_print("device:", self.device)
        self.model = None
        if torch is not None and TSCAN is not None:
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=self.img_size[0])
            if model_path is None:
                model_path = TSCAN_MODEL_PATH
            self._load_checkpoint(model_path)
            if self.device is not None:
                self.model.to(self.device)
            self.model.eval()
        else:
            _debug_print("Torch not found - model will not run. predict() will return dummy values.")
        self.frame_buffer = deque(maxlen=self.frame_depth)
        self.bvp_buffer = deque(maxlen=int(BUFFER_SIZE * 4))
    def _load_checkpoint(self, model_path):
        if not model_path or not os.path.exists(model_path):
            _debug_print(f"TS-CAN checkpoint not found at {model_path}. Skipping weight load.")
            return
        if torch is None:
            _debug_print("Torch not available: cannot load checkpoint!")
            return
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            state = ckpt
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:
                    state = ckpt["model_state_dict"]
                elif "model" in ckpt and isinstance(ckpt["model"], dict):
                    state = ckpt["model"]
                if not all(isinstance(v, torch.Tensor) for v in state.values()):
                    for v in state.values():
                        if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                            state = v
                            break
            else:
                state = ckpt
            state = _clean_state_dict(state)
            model_dict = self.model.state_dict()
            filtered_state = {}
            skipped = []
            for k, v in state.items():
                if k in model_dict:
                    try:
                        ckpt_shape = tuple(v.shape)
                    except Exception:
                        skipped.append((k, "non-tensor", tuple(model_dict[k].shape)))
                        continue
                    model_shape = tuple(model_dict[k].shape)
                    if ckpt_shape == model_shape:
                        filtered_state[k] = v
                    else:
                        skipped.append((k, ckpt_shape, model_shape))
            if len(filtered_state) == 0:
                _debug_print("No matching keys found between checkpoint and model. Skipping weight load.")
            else:
                model_dict.update(filtered_state)
                try:
                    self.model.load_state_dict(model_dict)
                    _debug_print(f"Weights loaded (partial). Copied {len(filtered_state)} params, skipped {len(skipped)} mismatched params.")
                except Exception as e:
                    print(f"[TSCAN] Warning: error while loading filtered state_dict: {e}")
                    if DEBUG_MODE:
                        import traceback; traceback.print_exc()
            if len(skipped) and DEBUG_MODE:
                for key, ckpt_shape, model_shape in skipped[:10]:
                    print("[TSCAN] skipped", key, "ckpt", ckpt_shape, "model", model_shape)
                if len(skipped) > 10:
                    print("[TSCAN] ...and", len(skipped)-10, "more skipped keys")
        except Exception as e:
            print(f"[TSCAN] Warning: failed to load checkpoint {model_path}: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
    def _preprocess_frame(self, frame_bgr):
        import cv2
        if frame_bgr is None:
            raise ValueError("None frame passed to preprocess")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32)
        if self.normalize == "-1_1":
            arr = (arr / 127.5) - 1.0
        elif self.normalize == "0_1":
            arr = arr / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return arr
    def add_frame(self, frame_bgr):
        try:
            pre = self._preprocess_frame(frame_bgr)
            self.frame_buffer.append(pre)
        except Exception as e:
            if DEBUG_MODE:
                print("[TSCAN] preprocess error:", e)
    def is_ready(self):
        return len(self.frame_buffer) >= self.frame_depth
    def predict(self):
        if self.model is None or torch is None:
            _debug_print("No model available: predict() returning None")
            return None
        if not self.is_ready():
            return None
        try:
            arr = np.stack(list(self.frame_buffer), axis=0)
            arr = np.expand_dims(arr, axis=0)
            arr = np.transpose(arr, (0, 2, 1, 3, 4)).astype(np.float32)
            inp = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                out = self.model(inp)
            out_np = out.cpu().numpy().squeeze()
            out_np = np.atleast_1d(out_np).astype(np.float32)
            for v in out_np:
                self.bvp_buffer.append(float(v))
            return out_np
        except Exception as e:
            if DEBUG_MODE:
                print("[TSCAN] Inference error:", e)
            return None
    def get_recent_bvp(self, seconds=10, fs=30.0):
        if len(self.bvp_buffer) == 0:
            return np.array([], dtype=np.float32)
        n = int(max(1, round(seconds * float(fs))))
        arr = np.array(list(self.bvp_buffer)[-n:], dtype=np.float32)
        return arr
    def get_bvp_buffer(self, length: int = None, seconds: float = None, fs: float = 30.0):
        if len(self.bvp_buffer) == 0:
            return np.array([], dtype=np.float32)
        if length is not None:
            n = int(max(0, int(length)))
            if n == 0:
                return np.array([], dtype=np.float32)
            arr = np.array(list(self.bvp_buffer)[-n:], dtype=np.float32)
            return arr
        if seconds is not None:
            n = int(max(1, round(float(seconds) * float(fs))))
            arr = np.array(list(self.bvp_buffer)[-n:], dtype=np.float32)
            return arr
        return np.array(list(self.bvp_buffer), dtype=np.float32)
    def clear_bvp_buffer(self):
        self.bvp_buffer.clear()
    def reset(self):
        self.frame_buffer.clear()
        self.clear_bvp_buffer()
    def set_device(self, device_str):
        if torch is None or self.model is None:
            _debug_print("set_device: torch/model not available.")
            return
        self.device = torch.device(device_str)
        self.model.to(self.device)
        _debug_print("Moved model to", self.device)
def test_tscan_model(model_path=None):
    instance = TSCANInference(model_path=model_path, frame_depth=TSCAN_FRAME_DEPTH, img_size=TSCAN_INPUT_SIZE)
    import cv2
    for i in range(TSCAN_FRAME_DEPTH):
        dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        instance.add_frame(dummy)
    out = instance.predict()
    if out is not None and len(out) >= 1:
        print("[TSCAN TEST] OK - output len:", len(out))
        return True
    else:
        print("[TSCAN TEST] NO OUTPUT")
        return False
if __name__ == "__main__":
    test_tscan_model(model_path=TSCAN_MODEL_PATH)