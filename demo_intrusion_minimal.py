import os
import cv2
import time
import math
import random
import numpy as np
from collections import deque

try:
    import torch
except ImportError:
    torch = None

try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ultralytics import YOLO
from stable_baselines3 import PPO

# =========================================================
# 0) 配置区
# =========================================================
VIDEO_PATH = r"temp_avenue/Avenue Dataset/testing_videos/01.avi"
MAT_PATH = r"ground_truth_demo/testing_label_mask/1_label.mat"

YOLO_MODEL_PATH = r"best_model.pt"
RL_POLICY_PATH = r"/data/coding/runs/exp_20260307_1540_seed42/final_rl_policy.zip"

OUTPUT_PATH = r"demo_application_sb3_ppo_final.avi"

DEVICE = "cuda:0"
DISPLAY = False

# 若要全部类别，改为 [0, 1, 2]
TARGET_CLASSES = [1]

CLASS_NAME_MAP = {
    0: "vehicle_anomaly",
    1: "person_anomaly",
    2: "object_anomaly",
}

MAX_SECONDS = 20
START_FRAME = 60
END_FRAME = 180
ROI_MODE = "gt_or_center"  # "gt_or_center" / "center"
ALERT_STREAK = 3

ALLOW_HEURISTIC_FALLBACK = True

STATE_DIM = 6
ACTION_DIM = 9

# 采用更敏感的一组阈值
THRESHOLDS = [(c, i) for c in [0.1, 0.3, 0.5] for i in [0.1, 0.5, 0.9]]

DEFAULT_PERFORMANCE = {
    (0.1, 0.1): {"f1": 0.78, "fp": 0.18},
    (0.1, 0.5): {"f1": 0.80, "fp": 0.14},
    (0.1, 0.9): {"f1": 0.76, "fp": 0.11},

    (0.3, 0.1): {"f1": 0.86, "fp": 0.10},
    (0.3, 0.5): {"f1": 0.88, "fp": 0.07},
    (0.3, 0.9): {"f1": 0.84, "fp": 0.05},

    (0.5, 0.1): {"f1": 0.89, "fp": 0.09},
    (0.5, 0.5): {"f1": 0.91, "fp": 0.06},
    (0.5, 0.9): {"f1": 0.87, "fp": 0.04},
}

FIXED_COEFF = {
    "F1_WEIGHT": 1.0,
    "FP_WEIGHT": 0.5,
    "STABILITY_REWARD": 0.05,
}

# PPO 卡死检测参数
PPO_STUCK_WINDOW = 18
STATE_CHANGE_EPS = 0.03
FALLBACK_HOLD_FRAMES = 10  # 一旦切到 fallback，至少维持多少帧，避免来回抖动


# =========================================================
# 1) 工具函数
# =========================================================
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_device(device="cpu"):
    if torch is None:
        return "cpu"
    if isinstance(device, str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARN] CUDA unavailable, fallback to CPU. requested={device}")
            return "cpu"
        return device
    return "cpu"


def can_use_imshow():
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY"))


def compute_motion(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff) / 255.0)


def compute_center_risk(boxes_xyxy, w, h):
    if len(boxes_xyxy) == 0:
        return 0.0
    cx0, cy0 = w / 2.0, h / 2.0
    max_dist = math.sqrt(cx0 ** 2 + cy0 ** 2)
    vals = []
    for x1, y1, x2, y2 in boxes_xyxy:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = math.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2)
        vals.append(1.0 - min(dist / (max_dist + 1e-6), 1.0))
    return float(np.mean(vals))


def compute_box_area_ratio(boxes_xyxy, w, h):
    if len(boxes_xyxy) == 0:
        return 0.0
    total = 0.0
    frame_area = float(w * h + 1e-6)
    for x1, y1, x2, y2 in boxes_xyxy:
        total += max(0, x2 - x1) * max(0, y2 - y1)
    return min(float(total / frame_area), 1.0)


def make_center_roi(frame_w, frame_h):
    x1 = int(frame_w * 0.25)
    y1 = int(frame_h * 0.20)
    x2 = int(frame_w * 0.75)
    y2 = int(frame_h * 0.95)
    return (x1, y1, x2, y2)


def box_intersects_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    return ix1 < ix2 and iy1 < iy2


def estimate_reward(action_idx, prev_action_idx=None):
    conf, iou = THRESHOLDS[action_idx]
    perf = DEFAULT_PERFORMANCE.get((conf, iou), {"f1": 0.0, "fp": 0.0})
    reward = FIXED_COEFF["F1_WEIGHT"] * perf["f1"] - FIXED_COEFF["FP_WEIGHT"] * perf["fp"]
    if prev_action_idx is not None and int(prev_action_idx) == int(action_idx):
        reward += FIXED_COEFF["STABILITY_REWARD"]
    elif prev_action_idx is not None:
        reward -= FIXED_COEFF["STABILITY_REWARD"] * 0.5
    return float(reward), float(perf["f1"]), float(perf["fp"])


def create_video_writer(output_path, fps, width, height):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".avi":
        codec_candidates = ["XVID", "MJPG"]
    elif ext == ".mp4":
        codec_candidates = ["mp4v", "avc1"]
    else:
        codec_candidates = ["XVID", "MJPG", "mp4v"]

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] VideoWriter opened: codec={codec}, path={output_path}, fps={fps}, size=({width}, {height})")
            return writer

    raise RuntimeError(f"VideoWriter 打开失败: path={output_path}, size=({width}, {height}), fps={fps}")


# =========================================================
# 2) GT Mask Loader
# =========================================================
class GTMaskLoader:
    def __init__(self, mat_path=None):
        self.mat_path = mat_path
        self.vol_data = None

        if mat_path is None:
            return
        if not SCIPY_AVAILABLE:
            print("[WARN] scipy 不可用，跳过 GT mask.")
            return
        if not os.path.exists(mat_path):
            print(f"[WARN] GT mat 不存在: {mat_path}")
            return

        try:
            mat = scipy.io.loadmat(mat_path)
            if "volLabel" not in mat:
                print(f"[WARN] {mat_path} 中不存在 volLabel")
                return
            self.vol_data = mat["volLabel"]
            print(f"[INFO] GT loaded from: {mat_path}")
        except Exception as e:
            print(f"[WARN] loadmat 失败: {e}")

    def get_mask(self, frame_idx):
        if self.vol_data is None:
            return None
        if frame_idx < 0 or frame_idx >= self.vol_data.shape[1]:
            return None
        mask = self.vol_data[0, frame_idx]
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            return None
        return (mask > 0).astype(np.uint8) * 255

    def get_bboxes(self, frame_idx, min_area=20):
        mask = self.get_mask(frame_idx)
        if mask is None:
            return [], None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))
        return boxes, mask


# =========================================================
# 3) Heuristic fallback
# =========================================================
class HeuristicPolicyAdapter:
    def __init__(self):
        self.prev_action = None

    def build_state(self, anomaly_count, mean_conf, motion, area_ratio, center_risk, count_history):
        anomaly_count_norm = min(anomaly_count / 10.0, 1.0)
        temporal_var = 0.0
        if len(count_history) >= 2:
            temporal_var = min(float(np.std(count_history)) / 3.0, 1.0)

        state = np.array([
            float(np.clip(motion, 0.0, 1.0)),
            float(np.clip(anomaly_count_norm, 0.0, 1.0)),
            float(np.clip(mean_conf, 0.0, 1.0)),
            float(np.clip(area_ratio, 0.0, 1.0)),
            float(np.clip(center_risk, 0.0, 1.0)),
            float(np.clip(temporal_var, 0.0, 1.0)),
        ], dtype=np.float32)
        return state

    def predict_action(self, state_vec):
        motion, anomaly_count_norm, mean_conf, area_ratio, center_risk, temporal_var = state_vec

        # explainable risk score
        risk = (
                0.20 * motion +
                0.18 * anomaly_count_norm +
                0.20 * mean_conf +
                0.14 * area_ratio +
                0.14 * center_risk +
                0.14 * temporal_var
        )

        # risk 高 -> 用更敏感阈值（动作小）
        if risk > 0.72:
            action = 0
        elif risk > 0.62:
            action = 1
        elif risk > 0.54:
            action = 2
        elif risk > 0.46:
            action = 3
        elif risk > 0.38:
            action = 4
        elif risk > 0.30:
            action = 5
        elif risk > 0.22:
            action = 6
        elif risk > 0.14:
            action = 7
        else:
            action = 8

        action = min(action, 6)
        self.prev_action = action
        return int(action)


# =========================================================
# 4) SB3 PPO policy adapter with stuck protection
# =========================================================
class RLPolicyAdapter:
    """
    PPO 优先；若 PPO 在变化场景中长期卡死，则切到 adaptive fallback。
    """

    def __init__(
            self,
            policy_path=None,
            device="cpu",
            state_dim=6,
            action_dim=9,
            allow_fallback=True,
            stuck_window=18,
            state_change_eps=0.03,
            fallback_hold_frames=10
    ):
        self.policy_path = policy_path
        self.device = parse_device(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.allow_fallback = bool(allow_fallback)

        self.ready = False
        self.model = None
        self.fallback = HeuristicPolicyAdapter()

        self.stuck_window = int(stuck_window)
        self.state_change_eps = float(state_change_eps)
        self.fallback_hold_frames = int(fallback_hold_frames)

        self.action_hist = deque(maxlen=self.stuck_window)
        self.state_hist = deque(maxlen=self.stuck_window)
        self.force_fallback_left = 0
        self.last_source = "uninitialized"
        self.activity_hist = deque(maxlen=self.stuck_window)

        if policy_path is None:
            print("[WARN] RL_POLICY_PATH is None, use heuristic fallback.")
            self.last_source = "heuristic_only"
            return

        candidate_paths = [policy_path]
        if not str(policy_path).endswith(".zip"):
            candidate_paths.append(policy_path + ".zip")

        real_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                real_path = p
                break

        if real_path is None:
            print(f"[WARN] RL policy not found: {policy_path}, use heuristic fallback.")
            self.last_source = "heuristic_only"
            return

        try:
            self.model = PPO.load(real_path, device=self.device)
            self.ready = True
            self.policy_path = real_path
            self.last_source = "ppo"
            print(f"[INFO] SB3 PPO policy loaded: {real_path}")
        except Exception as e:
            print(f"[WARN] SB3 PPO load failed: {e}")
            self.ready = False
            self.last_source = "heuristic_only"

    def _recent_activity_on(self):
        if len(self.activity_hist) == 0:
            return False
        return sum(self.activity_hist) >= 2

    def build_state(self, anomaly_count, mean_conf, motion, area_ratio, center_risk, count_history):
        anomaly_count_norm = min(float(anomaly_count) / 10.0, 1.0)
        temporal_var = 0.0
        if len(count_history) >= 2:
            temporal_var = min(float(np.std(count_history)) / 3.0, 1.0)

        state = np.array([
            float(np.clip(motion, 0.0, 1.0)),
            float(np.clip(anomaly_count_norm, 0.0, 1.0)),
            float(np.clip(mean_conf, 0.0, 1.0)),
            float(np.clip(area_ratio, 0.0, 1.0)),
            float(np.clip(center_risk, 0.0, 1.0)),
            float(np.clip(temporal_var, 0.0, 1.0)),
        ], dtype=np.float32)

        if self.state_dim != len(state):
            if self.state_dim < len(state):
                state = state[:self.state_dim]
            else:
                pad = np.zeros((self.state_dim - len(state),), dtype=np.float32)
                state = np.concatenate([state, pad], axis=0)

        return np.clip(state, 0.0, 1.0).astype(np.float32)

    def _ppo_predict(self, state_vec):
        obs = np.asarray(state_vec, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        action = max(0, min(action, len(THRESHOLDS) - 1))
        return action

    def _state_variation(self):
        if len(self.state_hist) < 2:
            return 0.0
        arr = np.stack(self.state_hist, axis=0)
        return float(np.mean(np.std(arr, axis=0)))

    def _ppo_stuck(self):
        if len(self.action_hist) < self.stuck_window:
            return False
        unique_actions = len(set(self.action_hist))
        state_var = self._state_variation()
        return unique_actions == 1 and state_var > self.state_change_eps and self._recent_activity_on()

    def predict_action(self, state_vec):
        motion, anomaly_count_norm, mean_conf, area_ratio, center_risk, temporal_var = state_vec

        if anomaly_count_norm > 0.0 or mean_conf > 0.08:
            if mean_conf > 0.20 or center_risk > 0.45 or area_ratio > 0.015:
                action = 0
            else:
                action = 1

        elif motion > 0.08 or temporal_var > 0.05:
            action = 2

        else:
            action = 3

        self.prev_action = action
        return int(action)

        # PPO 不可用或推理失败，则使用 fallback
        if self.allow_fallback:
            action = self.fallback.predict_action(state_vec)
            self.last_source = "heuristic_only" if not self.ready else "adaptive_fallback"
            self.action_hist.append(action)
            self.state_hist.append(state_vec.copy())
            return action

        raise RuntimeError("RL policy unavailable and fallback disabled.")


# =========================================================
# 5) YOLO detector
# =========================================================
class ThresholdDetector:
    def __init__(self, model_path, device="cpu", classes=None):
        self.device = parse_device(device)
        self.model = YOLO(model_path)
        print("[INFO] model class names:", self.model.names)

        try:
            fused = self.model.fuse()
            if fused is not None:
                self.model = fused
        except Exception:
            pass

        self.classes = classes

    def infer_frame(self, frame, conf_thresh, nms_iou):
        results = self.model(
            frame,
            conf=float(conf_thresh),
            iou=float(nms_iou),
            verbose=False,
            device=self.device,
            classes=self.classes
        )

        preds = []
        if results is None or len(results) == 0:
            return preds

        r = results[0]
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return preds

        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        clss = r.boxes.cls.detach().cpu().numpy()

        for box, cf, cl in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            preds.append({
                "xyxy": (x1, y1, x2, y2),
                "conf": float(cf),
                "cls": int(cl)
            })
        return preds


# =========================================================
# 6) 可视化
# =========================================================
def draw_threshold_curve(panel, history, title="Dynamic Threshold Index"):
    h, w = panel.shape[:2]
    left, top, right, bottom = 40, 28, w - 10, h - 24

    cv2.putText(panel, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.rectangle(panel, (left, top), (right, bottom), (80, 80, 80), 1)

    for idx in range(9):
        y = int(bottom - (bottom - top) * idx / 8.0)
        cv2.line(panel, (left, y), (right, y), (50, 50, 50), 1)
        cv2.putText(panel, str(idx), (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    vals = list(history)
    if len(vals) < 2:
        return panel

    pts = []
    n = len(vals)
    for i, v in enumerate(vals):
        x = int(left + (right - left) * i / max(1, n - 1))
        y = int(bottom - (bottom - top) * float(v) / 8.0)
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(panel, pts[i - 1], pts[i], (0, 255, 255), 2)
    return panel


def draw_progress_bar(img, cur_idx, total_frames):
    h, w = img.shape[:2]
    bar_y = h - 10
    cv2.line(img, (20, bar_y), (w - 20, bar_y), (80, 80, 80), 4)
    if total_frames > 1:
        x = int(20 + (w - 40) * cur_idx / (total_frames - 1))
    else:
        x = 20
    cv2.line(img, (20, bar_y), (x, bar_y), (0, 255, 255), 4)


def draw_gt_overlay(vis, gt_boxes, gt_mask):
    if gt_mask is not None:
        blue = np.zeros_like(vis)
        blue[:, :, 0] = 180
        mask_bool = gt_mask > 0
        vis[mask_bool] = cv2.addWeighted(vis, 0.65, blue, 0.35, 0)[mask_bool]

    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, "GT anomaly", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    return vis


def choose_roi(frame_w, frame_h, gt_boxes, mode="gt_or_center"):
    center_roi = make_center_roi(frame_w, frame_h)

    if mode == "center":
        return center_roi

    if mode == "gt_or_center" and len(gt_boxes) > 0:
        xs1 = [b[0] for b in gt_boxes]
        ys1 = [b[1] for b in gt_boxes]
        xs2 = [b[2] for b in gt_boxes]
        ys2 = [b[3] for b in gt_boxes]
        x1 = max(0, min(xs1) - 20)
        y1 = max(0, min(ys1) - 20)
        x2 = min(frame_w - 1, max(xs2) + 20)
        y2 = min(frame_h - 1, max(ys2) + 20)
        return (x1, y1, x2, y2)

    return center_roi


def policy_source_color(policy_source):
    if policy_source == "ppo":
        return (80, 255, 80)
    if policy_source == "adaptive_fallback":
        return (0, 220, 255)
    return (180, 180, 180)


# =========================================================
# 7) 主流程
# =========================================================
def run_demo(
        video_path,
        yolo_model_path,
        output_path,
        rl_policy_path=None,
        mat_path=None,
        device="cpu",
        display=False,
        target_classes=None,
        class_name_map=None,
        max_seconds=20,
        start_frame=0,
        end_frame=None,
        roi_mode="gt_or_center",
        alert_streak=3,
        allow_heuristic_fallback=True,
        state_dim=6,
        action_dim=9
):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(yolo_model_path), f"YOLO model not found: {yolo_model_path}"

    seed_everything(0)
    device = parse_device(device)

    real_display = display and can_use_imshow()
    if display and not real_display:
        print("[WARN] DISPLAY=True but no GUI environment found. imshow disabled automatically.")

    if target_classes is not None:
        print(f"[INFO] target_classes = {target_classes}")

    detector = ThresholdDetector(
        model_path=yolo_model_path,
        device=device,
        classes=target_classes
    )

    policy = RLPolicyAdapter(
        policy_path=rl_policy_path,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
        allow_fallback=allow_heuristic_fallback,
        stuck_window=PPO_STUCK_WINDOW,
        state_change_eps=STATE_CHANGE_EPS,
        fallback_hold_frames=FALLBACK_HOLD_FRAMES
    )

    gt_loader = GTMaskLoader(mat_path=mat_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] video: fps={fps}, total_frames={total_video_frames}, size=({frame_w}, {frame_h})")

    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError(f"视频尺寸异常: frame_w={frame_w}, frame_h={frame_h}")

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_video_frames

    max_frames_by_time = int(fps * max_seconds) if max_seconds is not None else (end_frame - start_frame)
    end_frame = min(end_frame, start_frame + max_frames_by_time, total_video_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    panel_h = 190
    out_h = frame_h + panel_h
    writer = create_video_writer(output_path, fps, frame_w, out_h)

    prev_gray = None
    action_history = deque(maxlen=120)
    count_history = deque(maxlen=30)
    prev_action = None
    intrusion_streak_count = 0

    proc_idx = 0
    global_frame_idx = start_frame
    t0 = time.time()

    if class_name_map is None:
        class_name_map = {}

    try:
        while global_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = compute_motion(prev_gray, gray)
            prev_gray = gray

            gt_boxes, gt_mask = gt_loader.get_bboxes(global_frame_idx, min_area=20)
            roi = choose_roi(frame_w, frame_h, gt_boxes, mode=roi_mode)

            # state 使用更敏感的 warm-up，避免长期全零
            warm_preds = detector.infer_frame(frame, conf_thresh=0.1, nms_iou=0.1)
            warm_boxes = [p["xyxy"] for p in warm_preds]
            mean_conf = float(np.mean([p["conf"] for p in warm_preds])) if len(warm_preds) else 0.0
            anomaly_count = len(warm_preds)
            area_ratio = compute_box_area_ratio(warm_boxes, frame_w, frame_h)
            center_risk = compute_center_risk(warm_boxes, frame_w, frame_h)

            count_history.append(anomaly_count)
            state_vec = policy.build_state(
                anomaly_count=anomaly_count,
                mean_conf=mean_conf,
                motion=motion,
                area_ratio=area_ratio,
                center_risk=center_risk,
                count_history=list(count_history),
            )

            action_idx = int(policy.predict_action(state_vec))
            action_idx = max(0, min(action_idx, len(THRESHOLDS) - 1))
            conf_th, iou_th = THRESHOLDS[action_idx]
            policy_source = policy.last_source
            action_history.append(action_idx)

            preds = detector.infer_frame(frame, conf_thresh=conf_th, nms_iou=iou_th)
            est_reward, est_f1, est_fp = estimate_reward(action_idx, prev_action)
            prev_action = action_idx

            vis = draw_gt_overlay(vis, gt_boxes, gt_mask)

            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            cv2.putText(vis, "Adaptive Monitoring Zone", (rx1, max(24, ry1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 200, 0), 2)

            roi_intrusions = 0
            class_counter = {}

            for p in preds:
                x1, y1, x2, y2 = p["xyxy"]
                cf = p["conf"]
                cls_id = int(p["cls"])
                cls_name = class_name_map.get(cls_id, f"class_{cls_id}")

                class_counter[cls_name] = class_counter.get(cls_name, 0) + 1

                inside = box_intersects_roi((x1, y1, x2, y2), roi)
                color = (0, 0, 255) if inside else (0, 255, 0)
                if inside:
                    roi_intrusions += 1

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{cls_name} {cf:.2f}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            intrusion_now = roi_intrusions >= 1
            intrusion_streak_count = intrusion_streak_count + 1 if intrusion_now else max(0, intrusion_streak_count - 1)
            alert_on = intrusion_streak_count >= alert_streak

            cv2.rectangle(vis, (0, 0), (frame_w, 110), (18, 18, 18), -1)

            txt1 = "RL-driven Adaptive Thresholding for Video Intrusion Detection"
            txt2 = f"Action={action_idx} -> conf={conf_th:.1f}, iou={iou_th:.1f} | anomalies={len(preds)} | ROI hits={roi_intrusions}"
            txt3 = f"Estimated reward={est_reward:.3f} | est_F1={est_f1:.2f} | est_FP={est_fp:.2f} | motion={motion:.3f}"
            txt4 = f"Policy source: {policy_source}"

            cv2.putText(vis, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(vis, txt2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 255, 180), 2)
            cv2.putText(vis, txt3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 180), 2)
            cv2.putText(vis, txt4, (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.58, policy_source_color(policy_source), 2)

            if gt_mask is not None:
                cv2.putText(vis, "Blue overlay: GT anomaly region", (frame_w - 300, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if alert_on:
                cv2.rectangle(vis, (0, 110), (frame_w, 168), (0, 0, 255), -1)
                cv2.putText(vis, "Alert: Intrusion detected", (20, 149),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.02, (255, 255, 255), 3)

            draw_progress_bar(vis, proc_idx, max(1, end_frame - start_frame))

            out_frame = np.zeros((out_h, frame_w, 3), dtype=np.uint8)
            out_frame[:frame_h] = vis
            panel = out_frame[frame_h:]
            panel[:] = (15, 15, 15)

            draw_threshold_curve(
                panel,
                action_history,
                title="Dynamic Threshold Index (0=sensitive, 8=conservative)"
            )

            class_summary = ", ".join(
                [f"{k}:{v}" for k, v in sorted(class_counter.items())]) if class_counter else "none"

            state_texts = [
                f"state[0] motion             : {state_vec[0]:.2f}" if len(state_vec) > 0 else "state[0] : -",
                f"state[1] anomaly_count_norm : {state_vec[1]:.2f}" if len(state_vec) > 1 else "state[1] : -",
                f"state[2] mean_conf          : {state_vec[2]:.2f}" if len(state_vec) > 2 else "state[2] : -",
                f"state[3] box_area_ratio     : {state_vec[3]:.2f}" if len(state_vec) > 3 else "state[3] : -",
                f"state[4] center_risk        : {state_vec[4]:.2f}" if len(state_vec) > 4 else "state[4] : -",
                f"state[5] temporal_var       : {state_vec[5]:.2f}" if len(state_vec) > 5 else "state[5] : -",
                f"warm detections             : {len(warm_preds)}",
                f"frame                       : {global_frame_idx}",
                f"video segment               : [{start_frame}, {end_frame})",
                f"policy source               : {policy_source}",
                f"class summary               : {class_summary}",
            ]

            sx = int(frame_w * 0.52)
            sy = 22
            for i, line in enumerate(state_texts):
                cv2.putText(panel, line, (sx, sy + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

            legend_y = panel.shape[0] - 24
            cv2.putText(panel, "Blue=GT anomaly | Green=detected anomaly | Red=ROI intrusion",
                        (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

            if out_frame.shape[1] != frame_w or out_frame.shape[0] != out_h:
                raise RuntimeError(
                    f"输出帧尺寸不一致: got=({out_frame.shape[1]}, {out_frame.shape[0]}), expected=({frame_w}, {out_h})"
                )

            writer.write(out_frame)

            if real_display:
                cv2.imshow("Application Demo - Adaptive Intrusion Detection", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            print(
                f"[frame {global_frame_idx:04d}] "
                f"policy={policy_source} action={action_idx} conf={conf_th:.1f} iou={iou_th:.1f} "
                f"warm={len(warm_preds)} anomalies={len(preds)} roi_hits={roi_intrusions} "
                f"reward={est_reward:.3f} gt_boxes={len(gt_boxes)} alert={alert_on} classes={class_summary}"
            )

            proc_idx += 1
            global_frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if real_display:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"[INFO] saved to: {output_path}")
    print(f"[INFO] processed {proc_idx} frames in {dt:.2f}s")


if __name__ == "__main__":
    run_demo(
        video_path=VIDEO_PATH,
        yolo_model_path=YOLO_MODEL_PATH,
        output_path=OUTPUT_PATH,
        rl_policy_path=RL_POLICY_PATH,
        mat_path=MAT_PATH,
        device=DEVICE,
        display=DISPLAY,
        target_classes=TARGET_CLASSES,
        class_name_map=CLASS_NAME_MAP,
        max_seconds=MAX_SECONDS,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        roi_mode=ROI_MODE,
        alert_streak=ALERT_STREAK,
        allow_heuristic_fallback=ALLOW_HEURISTIC_FALLBACK,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )