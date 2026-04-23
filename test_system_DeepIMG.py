from __future__ import annotations

import csv
import logging
import os
import time
import threading
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from time import time as now

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

log = logging.getLogger(__name__)

# ─── DEVICE ────────────────────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

yolo = YOLO("./Object_detection/Model/yolo11n.pt")

depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if torch.backends.mps.is_available() else -1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Detection record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    track_id: int
    cx: int          # centroid x
    cy: int          # centroid y (chân)
    zone: str | None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Crowd Heatmap
# ─────────────────────────────────────────────────────────────────────────────

class CrowdHeatmap:
    """
    Tích lũy mật độ theo thời gian.
    - Decay: làm mờ dần → heatmap phản ánh hoạt động gần đây.
    - Lưu ảnh PNG định kỳ.
    """

    def __init__(self, cfg: dict, frame_size: tuple[int, int]):
        self.enabled       = cfg.get("enabled", True)
        self.decay         = float(cfg.get("decay", 0.995))
        self.save_interval = float(cfg.get("save_interval_sec", 300))
        self.out_dir       = cfg.get("output_dir", "./analytics/heatmaps")
        self.w, self.h     = frame_size

        os.makedirs(self.out_dir, exist_ok=True)

        self._map       = np.zeros((self.h, self.w), dtype=np.float32)
        self._next_save = time.time() + self.save_interval

    def update(self, detections: list[Detection]) -> None:
        if not self.enabled:
            return
        # Decay: mờ dần theo thời gian
        self._map *= self.decay
        # Gaussian splat tại centroid mỗi detection
        for det in detections:
            cx = int(np.clip(det.cx, 0, self.w - 1))
            cy = int(np.clip(det.cy, 0, self.h - 1))
            x1 = max(0, cx - 10); x2 = min(self.w, cx + 11)
            y1 = max(0, cy - 10); y2 = min(self.h, cy + 11)
            gx = np.arange(x1, x2) - cx
            gy = np.arange(y1, y2) - cy
            gxx, gyy = np.meshgrid(gx, gy)
            kernel = np.exp(-(gxx**2 + gyy**2) / (2 * 7**2)).astype(np.float32)
            self._map[y1:y2, x1:x2] += kernel

    def draw(self, frame: np.ndarray, alpha: float = 0.45) -> None:
        """Overlay heatmap lên frame (in-place)."""
        if not self.enabled or self._map.max() < 1e-3:
            return
        norm    = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        # Chỉ vẽ vùng có giá trị đáng kể
        mask = (norm > 10).astype(np.uint8)
        roi  = cv2.bitwise_and(colored, colored, mask=mask)
        cv2.addWeighted(roi, alpha, frame, 1.0, 0, frame)

    def tick(self, t_now: float) -> None:
        if not self.enabled:
            return
        if t_now >= self._next_save:
            self._save(t_now)
            self._next_save = t_now + self.save_interval

    def _save(self, t_now: float) -> None:
        if self._map.max() < 1e-3:
            return
        ts   = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_now))
        path = os.path.join(self.out_dir, f"heatmap_{ts}.png")
        norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        img  = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(path, img)
        log.info("Heatmap saved: %s", path)
        print(f"[CrowdHeatmap] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trajectory Drawer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDrawer:
    """Vẽ đuôi trajectory cho mỗi track, màu fade dần từ đậm (mới) → nhạt (cũ)."""

    def __init__(self, cfg: dict):
        self.enabled  = cfg.get("enabled", True)
        self.max_tail = int(cfg.get("max_tail", 60))
        self.min_disp = float(cfg.get("min_displacement", 30))
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.max_tail)
        )
        self._colors: dict[int, tuple] = {}

    def update(self, detections: list[Detection]) -> None:
        if not self.enabled:
            return
        seen = set()
        for det in detections:
            self._history[det.track_id].append((det.cx, det.cy))
            seen.add(det.track_id)
            if det.track_id not in self._colors:
                np.random.seed(det.track_id * 7 + 13)
                h = int(np.random.randint(0, 180))
                c = cv2.cvtColor(
                    np.array([[[h, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0][0]
                self._colors[det.track_id] = tuple(int(x) for x in c)
        # Xóa track đã mất khỏi history
        stale = [k for k in self._history if k not in seen]
        for k in stale:
            del self._history[k]
            self._colors.pop(k, None)

    def draw(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        for tid, hist in self._history.items():
            pts = list(hist)
            if len(pts) < 2:
                continue
            total_disp = sum(
                np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                for i in range(1, len(pts))
            )
            if total_disp < self.min_disp:
                continue
            color = self._colors.get(tid, (0, 255, 0))
            n = len(pts)
            for i in range(1, n):
                alpha = i / n           # 0→1: cũ→mới
                c = tuple(int(x * alpha) for x in color)
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pts[i-1], pts[i], c, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Occupancy Counter
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyCounter:
    """
    Đếm số người trong mỗi zone theo từng phút.
    Ghi log ra CSV: timestamp, zone, count, peak.
    """

    def __init__(self, cfg: dict, zone_names: list[str]):
        self.enabled      = cfg.get("enabled", True)
        self.log_interval = float(cfg.get("log_interval_sec", 30))
        self.csv_path     = cfg.get("output_csv", "./analytics/occupancy.csv")
        self.zone_names   = list(zone_names)

        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        os.makedirs(csv_dir, exist_ok=True)

        # Ghi header chỉ khi file chưa tồn tại
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            if write_header:
                csv.writer(f).writerow(["timestamp", "zone", "count", "peak"])
            f.flush()

        self._current: dict[str, set] = {z: set() for z in self.zone_names}
        self._peak:    dict[str, int] = {z: 0     for z in self.zone_names}
        self._next_log = time.time() + self.log_interval

        print(f"[OccupancyCounter] CSV: {os.path.abspath(self.csv_path)}")
        print(f"[OccupancyCounter] Interval: {self.log_interval}s | Zones: {self.zone_names}")

    def update(self, detections: list[Detection]) -> None:
        if not self.enabled:
            return
        # Reset count cho zone đã biết
        for z in self.zone_names:
            self._current[z] = set()
        for det in detections:
            if not det.zone:
                continue
            # Tự thêm zone mới nếu xuất hiện lần đầu
            if det.zone not in self._current:
                self._current[det.zone] = set()
                self._peak[det.zone]    = 0
                self.zone_names.append(det.zone)
            self._current[det.zone].add(det.track_id)
        # Cập nhật peak
        for z, ids in self._current.items():
            cnt = len(ids)
            if cnt > self._peak.get(z, 0):
                self._peak[z] = cnt

    def draw(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        y = 20
        cv2.putText(frame, "Occupancy", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        for z in self.zone_names:
            cnt  = len(self._current.get(z, set()))
            peak = self._peak.get(z, 0)
            text = f"  {z}: {cnt} (peak {peak})"
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
            y += 18

    def tick(self, t_now: float) -> None:
        if not self.enabled:
            return
        if t_now >= self._next_log:
            self._log(t_now)
            for z in list(self._peak.keys()):
                self._peak[z] = 0
            self._next_log = t_now + self.log_interval

    def _log(self, t_now: float) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_now))
        try:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                for z in self.zone_names:
                    cnt  = len(self._current.get(z, set()))
                    peak = self._peak.get(z, 0)
                    w.writerow([ts, z, cnt, peak])
                f.flush()
            msg = f"[OccupancyCounter] Logged @ {ts}"
            log.info(msg)
            print(msg)
        except Exception as e:
            err = f"[OccupancyCounter] ERROR writing CSV: {e}"
            log.error(err)
            print(err)


# ─────────────────────────────────────────────────────────────────────────────
# Facade – AnalyticsManager
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsManager:
    """Tập hợp 3 module analytics, giao tiếp qua 1 interface duy nhất."""

    def __init__(self, analytics_cfg: dict, frame_size: tuple[int, int],
                 zone_names: list[str] | None = None):
        self.heatmap    = CrowdHeatmap(analytics_cfg.get("heatmap", {}), frame_size)
        self.trajectory = TrajectoryDrawer(analytics_cfg.get("trajectory", {}))
        self.occupancy  = OccupancyCounter(
            analytics_cfg.get("occupancy", {}),
            zone_names or [],
        )

    def update(self, detections: list[Detection], t_now: float) -> None:
        self.heatmap.update(detections)
        self.trajectory.update(detections)
        self.occupancy.update(detections)

    def draw(self, frame: np.ndarray) -> None:
        self.heatmap.draw(frame)
        self.trajectory.draw(frame)
        self.occupancy.draw(frame)

    def tick(self, t_now: float) -> None:
        self.heatmap.tick(t_now)
        self.occupancy.tick(t_now)


# ─── DEPTH UTILS ────────────────────────────────────────────────────────────────

def get_depth_map(frame_rgb: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(frame_rgb)
    result  = depth_pipe(pil_img)
    depth   = np.array(result["depth"])
    depth   = cv2.resize(depth, (frame_rgb.shape[1], frame_rgb.shape[0]))
    depth   = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth   = 1.0 - depth   # gần = lớn
    return depth


def sample_depth_at_box(depth_map: np.ndarray, box: list, percentile: int = 20) -> float:
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.percentile(roi, percentile))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map)
    return (depth_map - d_min) / (d_max - d_min)


# ─── ALERT LOGIC ────────────────────────────────────────────────────────────────
DANGER_THRESHOLD  = 0.75
WARNING_THRESHOLD = 0.55

def get_alert_level(depth_score: float) -> str:
    if depth_score > DANGER_THRESHOLD:
        return "DANGER"
    elif depth_score > WARNING_THRESHOLD:
        return "WARNING"
    return "SAFE"


# ─── 3D BOUNDING BOX ──────────────────────────────────────────────────────────
def draw_3d_box(frame: np.ndarray, box, depth_score: float, color) -> None:
    x1, y1, x2, y2 = map(int, box)
    offset = max(4, int((1 - depth_score) * 20))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    bx1, by1 = x1 - offset, y1 - offset
    bx2, by2 = x2 - offset, y2 - offset
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

    for p1, p2 in [
        ((x1, y1), (bx1, by1)),
        ((x2, y1), (bx2, by1)),
        ((x2, y2), (bx2, by2)),
        ((x1, y2), (bx1, by2)),
    ]:
        cv2.line(frame, p1, p2, color, 1)


# ─── VISUALIZATION ─────────────────────────────────────────────────────────────
ALERT_COLORS = {
    "DANGER":  (0, 0, 255),
    "WARNING": (0, 165, 255),
    "SAFE":    (0, 200, 80),
}
ZONE_STYLES = {
    "restricted": {
        "fill":   (0, 0, 255),
        "border": (0, 0, 220),
        "alpha":  0.20,
    },
    "warning_zone": {
        "fill":   (0, 165, 255),
        "border": (0, 140, 220),
        "alpha":  0.15,
    },
}
DEFAULT_ZONE_STYLE = {"fill": (180, 0, 180), "border": (150, 0, 150), "alpha": 0.15}


def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ─── HOMOGRAPHY SETUP ──────────────────────────────────────────────────────────
_SRC = np.float32([
    [660,  640],
    [1460, 450],
    [1950, 650],
    [950,  800],
])
_DST = np.float32([
    [0,   0],
    [700, 0],
    [700, 500],
    [0,   500],
])

H     = cv2.getPerspectiveTransform(_SRC, _DST)
H_INV = cv2.getPerspectiveTransform(_DST, _SRC)


def world_to_image(pts_world: np.ndarray) -> np.ndarray:
    pts = pts_world.reshape(-1, 1, 2).astype(np.float32)
    pts_img = cv2.perspectiveTransform(pts, H_INV)
    return pts_img.reshape(-1, 1, 2).astype(np.int32)


def image_to_world(pt_img: tuple) -> np.ndarray:
    pt = np.array([[[float(pt_img[0]), float(pt_img[1])]]], dtype=np.float32)
    pt_world = cv2.perspectiveTransform(pt, H)
    return pt_world[0][0]


ZONES_WORLD = {
    "restricted": np.array([
        [50,  20],
        [550, 20],
        [550, 380],
        [50,  380],
    ], dtype=np.float32),
}


def draw_zones(frame: np.ndarray, zones_world: dict) -> None:
    for zone_name, poly_world in zones_world.items():
        style  = ZONE_STYLES.get(zone_name, DEFAULT_ZONE_STYLE)
        fill   = style["fill"]
        border = style["border"]
        alpha  = style["alpha"]

        poly_img = world_to_image(poly_world)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_img], fill)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [poly_img], isClosed=True, color=border, thickness=2)

        label = zone_name.upper()
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        px, py = int(poly_img[0][0][0]), int(poly_img[0][0][1])
        pad = 4
        cv2.rectangle(frame,
                      (px - pad, py - th - pad - baseline),
                      (px + tw + pad, py + baseline),
                      fill, -1)
        cv2.putText(frame, label, (px, py - baseline),
                    font, font_scale, (255, 255, 255), thickness)


def is_in_zone(cx: int, cy: int, zones_world: dict) -> str | None:
    pt_world = image_to_world((cx, cy))
    for zone_name, poly_world in zones_world.items():
        inside = cv2.pointPolygonTest(
            poly_world.reshape(-1, 1, 2).astype(np.float32),
            (float(pt_world[0]), float(pt_world[1])),
            False,
        )
        if inside >= 0:
            return zone_name
    return None


# ─── DEPTH WORKER THREAD ───────────────────────────────────────────────────────

from queue import Queue

class DepthWorker:
    """
    Chạy depth estimation bất đồng bộ trên thread riêng.
    Main loop đẩy frame vào, lấy kết quả mới nhất qua get().
    """
    def __init__(self):
        self._input_q = Queue(maxsize=1)
        self._result  = None
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            frame_rgb = self._input_q.get()
            result = get_depth_map(frame_rgb)
            with self._lock:
                self._result = result

    def submit(self, frame_rgb: np.ndarray):
        """Gửi frame mới (non-blocking, drop nếu đang bận)."""
        if not self._input_q.full():
            self._input_q.put(frame_rgb.copy())

    def get(self) -> np.ndarray | None:
        """Trả về depth map mới nhất (None nếu chưa có)."""
        with self._lock:
            return self._result


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./Data/test")
    opt = parser.parse_args()

    source = int(opt.source) if opt.source.isdigit() else opt.source
    cap = cv2.VideoCapture(source)

    ret, frame = cap.read()
    if not ret:
        print("Cannot read source")
        return

    h, w = frame.shape[:2]

    # ── Analytics ──
    analytics_cfg = {
        "heatmap": {
            "enabled":          True,
            "decay":            0.995,
            "save_interval_sec": 300,
            "output_dir":       "./analytics/heatmaps",
        },
        "trajectory": {
            "enabled":          True,
            "max_tail":         60,
            "min_displacement": 30,
        },
        "occupancy": {
            "enabled":          True,
            "log_interval_sec": 30,
            "output_csv":       "./analytics/occupancy.csv",
        },
    }
    analytics = AnalyticsManager(
        analytics_cfg,
        frame_size=(w, h),
        zone_names=list(ZONES_WORLD.keys()),
    )

    # ── Tracking storage ──
    tracks = defaultdict(lambda: {
        "history":     deque(maxlen=30),
        "zone":        None,
        "zone_time":   0,
        "last_update": None,
    })

    fps_deque   = deque(maxlen=30)
    DEPTH_EVERY_N = 3
    frame_count   = 0
    depth_worker  = DepthWorker()

    # Kick-start depth worker với frame đầu tiên
    depth_worker.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print("Start processing...")

    while True:
        t_start = now()

        ret, frame = cap.read()
        if not ret:
            break

        t_now     = now()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── YOLO tracking ──
        results = yolo.track(frame, persist=True, verbose=False, classes=[0])
        result  = results[0]

        # ── Depth (async) ──
        frame_count += 1
        if frame_count % DEPTH_EVERY_N == 0:
            depth_worker.submit(frame_rgb)

        cached_depth = depth_worker.get()
        if cached_depth is None:
            cached_depth = np.zeros((h, w), dtype=np.float32)

        depth_norm = normalize_depth(cached_depth)

        # ── Depth thumbnail (top-right) ──
        depth_vis   = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        thumb = cv2.resize(depth_color, (w // 4, h // 4))
        frame[0: h // 4, w - w // 4: w] = thumb

        # ── Zones ──
        draw_zones(frame, ZONES_WORLD)

        # ── Detections ──
        detections_for_analytics: list[Detection] = []
        boxes = result.boxes

        if boxes is not None and len(boxes):
            ids   = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
            xyxys = boxes.xyxy.cpu().numpy()

            for box, track_id in zip(xyxys, ids):
                if track_id is None:
                    continue

                track_id        = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                h_d, w_d    = depth_norm.shape
                x1_c, y1_c  = max(0, x1), max(0, y1)
                x2_c, y2_c  = min(w_d, x2), min(h_d, y2)
                roi         = depth_norm[y1_c:y2_c, x1_c:x2_c]
                z           = float(np.percentile(roi, 20)) if roi.size > 0 else 0.0

                cx = (x1 + x2) // 2
                cy = y2
                current_zone = is_in_zone(cx, cy, ZONES_WORLD)

                track = tracks[track_id]
                track["history"].append((cx, cy, z, t_now))

                dt = 0.0
                if track["last_update"] is not None:
                    dt = t_now - track["last_update"]
                track["last_update"] = t_now

                velocity, dz = 0.0, 0.0
                if len(track["history"]) >= 2:
                    x_prev, y_prev, z_prev, _ = track["history"][-2]
                    dx, dy   = cx - x_prev, cy - y_prev
                    velocity = np.sqrt(dx*dx + dy*dy) / (dt + 1e-5)
                    dz       = z - z_prev

                if track["zone"] == current_zone:
                    track["zone_time"] += dt
                else:
                    track["zone_time"] = 0
                track["zone"] = current_zone

                intrusion  = current_zone == "restricted" and z > 0.4
                running    = velocity > 20
                approaching = dz > 0.01
                loitering  = (
                    current_zone == "restricted"
                    and velocity < 2
                    and track["zone_time"] > 3
                )

                if intrusion and running:
                    alert = "CRITICAL"
                elif intrusion and approaching:
                    alert = "HIGH"
                elif intrusion or loitering:
                    alert = "MEDIUM"
                else:
                    alert = "SAFE"

                draw_color = {
                    "CRITICAL": (0, 0, 255),
                    "HIGH":     (0, 165, 255),
                    "MEDIUM":   (0, 255, 255),
                    "SAFE":     (0, 255, 0),
                }[alert]

                draw_3d_box(frame, box, z, draw_color)
                draw_label(frame, f"ID:{track_id} {alert} z:{z:.2f}", x1, y1, draw_color)

                detections_for_analytics.append(
                    Detection(track_id=track_id, cx=cx, cy=cy, zone=current_zone)
                )

        # ── Analytics update / draw / tick ──
        analytics.update(detections_for_analytics, t_now)
        analytics.draw(frame)
        analytics.tick(t_now)

        # ── FPS ──
        fps_deque.append(1.0 / (now() - t_start + 1e-6))
        fps = min(60, float(np.mean(fps_deque)))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO + Depth + Analytics", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
