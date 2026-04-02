# """
# analytics.py
# ────────────
# P2 – Crowd heatmap · Trajectory · Occupancy counter

# Cách dùng:
#     analytics = AnalyticsManager(cfg["analytics"], frame_size=(w, h))

#     # Mỗi frame, sau khi có detections:
#     analytics.update(detections, t_now)
#     analytics.draw(frame)

#     # Mỗi frame:
#     analytics.tick(t_now)
# """

# from __future__ import annotations
# import os
# import csv
# import logging
# import time
# from collections import defaultdict, deque
# from dataclasses import dataclass, field

# import cv2
# import numpy as np

# log = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # Detection record (truyền vào từ main loop)
# # ─────────────────────────────────────────────────────────────────────────────

# @dataclass
# class Detection:
#     track_id: int
#     cx: int          # centroid x
#     cy: int          # centroid y (chân)
#     zone: str | None


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Crowd Heatmap
# # ─────────────────────────────────────────────────────────────────────────────

# class CrowdHeatmap:
#     """
#     Tích lũy mật độ theo thời gian.
#     - Decay: làm mờ dần (nhân với hệ số < 1 mỗi frame) → heatmap phản ánh
#       hoạt động gần đây thay vì tích lũy vô hạn.
#     - Lưu ảnh PNG định kỳ.
#     """

#     def __init__(self, cfg: dict, frame_size: tuple[int, int]):
#         self.enabled = cfg.get("enabled", True)
#         self.decay   = float(cfg.get("decay", 0.995))
#         self.save_interval = float(cfg.get("save_interval_sec", 300))
#         self.out_dir = cfg.get("output_dir", "./analytics/heatmaps")
#         self.w, self.h = frame_size

#         os.makedirs(self.out_dir, exist_ok=True)

#         self._map  = np.zeros((self.h, self.w), dtype=np.float32)
#         self._next_save = time.time() + self.save_interval

#     def update(self, detections: list[Detection]) -> None:
#         if not self.enabled:
#             return
#         # Decay
#         self._map *= self.decay
#         # Tích lũy: Gaussian splat tại mỗi centroid
#         for det in detections:
#             cx, cy = np.clip(det.cx, 0, self.w - 1), np.clip(det.cy, 0, self.h - 1)
#             # Kernel 21×21 gaussian (sigma ≈ 7px)
#             x1 = max(0, cx - 10); x2 = min(self.w, cx + 11)
#             y1 = max(0, cy - 10); y2 = min(self.h, cy + 11)
#             gx = np.arange(x1, x2) - cx
#             gy = np.arange(y1, y2) - cy
#             gxx, gyy = np.meshgrid(gx, gy)
#             kernel = np.exp(-(gxx**2 + gyy**2) / (2 * 7**2)).astype(np.float32)
#             self._map[y1:y2, x1:x2] += kernel

#     def draw(self, frame: np.ndarray, alpha: float = 0.45) -> None:
#         """Overlay heatmap lên frame (in-place)."""
#         if not self.enabled or self._map.max() < 1e-3:
#             return
#         norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
#         colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
#         # Chỉ vẽ ở vùng có giá trị đáng kể
#         mask = (norm > 10).astype(np.uint8)
#         roi  = cv2.bitwise_and(colored, colored, mask=mask)
#         cv2.addWeighted(roi, alpha, frame, 1.0, 0, frame)

#     def tick(self, t_now: float) -> None:
#         if not self.enabled:
#             return
#         if t_now >= self._next_save:
#             self._save(t_now)
#             self._next_save = t_now + self.save_interval

#     def _save(self, t_now: float) -> None:
#         if self._map.max() < 1e-3:
#             return
#         ts   = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_now))
#         path = os.path.join(self.out_dir, f"heatmap_{ts}.png")
#         norm = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
#         img  = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
#         cv2.imwrite(path, img)
#         log.info("Heatmap saved: %s", path)


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. Trajectory
# # ─────────────────────────────────────────────────────────────────────────────

# class TrajectoryDrawer:
#     """
#     Vẽ đuôi trajectory (lịch sử điểm) cho mỗi track.
#     - Màu đuôi fade dần từ đậm (mới) → nhạt (cũ).
#     - Bỏ qua track đứng yên (displacement quá nhỏ).
#     """

#     def __init__(self, cfg: dict):
#         self.enabled       = cfg.get("enabled", True)
#         self.max_tail      = int(cfg.get("max_tail", 60))
#         self.min_disp      = float(cfg.get("min_displacement", 30))
#         # Lịch sử: {track_id: deque[(cx, cy)]}
#         self._history: dict[int, deque] = defaultdict(
#             lambda: deque(maxlen=self.max_tail)
#         )
#         # Bảng màu cố định theo track_id
#         self._colors: dict[int, tuple] = {}

#     def update(self, detections: list[Detection]) -> None:
#         if not self.enabled:
#             return
#         seen = set()
#         for det in detections:
#             self._history[det.track_id].append((det.cx, det.cy))
#             seen.add(det.track_id)
#             if det.track_id not in self._colors:
#                 # Màu ngẫu nhiên nhưng đủ sáng
#                 np.random.seed(det.track_id * 7 + 13)
#                 h = int(np.random.randint(0, 180))
#                 c = cv2.cvtColor(
#                     np.array([[[h, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
#                 )[0][0]
#                 self._colors[det.track_id] = tuple(int(x) for x in c)
#         # Xóa track đã mất
#         stale = [k for k in self._history if k not in seen]
#         for k in stale:
#             del self._history[k]
#             self._colors.pop(k, None)

#     def draw(self, frame: np.ndarray) -> None:
#         if not self.enabled:
#             return
#         for tid, hist in self._history.items():
#             pts = list(hist)
#             if len(pts) < 2:
#                 continue
#             # Kiểm tra displacement tổng
#             total_disp = sum(
#                 np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
#                 for i in range(1, len(pts))
#             )
#             if total_disp < self.min_disp:
#                 continue

#             color = self._colors.get(tid, (0, 255, 0))
#             n = len(pts)
#             for i in range(1, n):
#                 alpha = i / n           # 0→1: cũ→mới
#                 c = tuple(int(x * alpha) for x in color)
#                 thickness = max(1, int(alpha * 3))
#                 cv2.line(frame, pts[i-1], pts[i], c, thickness, cv2.LINE_AA)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. Occupancy Counter
# # ─────────────────────────────────────────────────────────────────────────────

# class OccupancyCounter:
#     """
#     Đếm số người trong mỗi zone theo từng phút.
#     Ghi log ra CSV: timestamp, zone, count, peak.
#     """

#     def __init__(self, cfg: dict, zone_names: list[str]):
#         self.enabled      = cfg.get("enabled", True)
#         self.log_interval = float(cfg.get("log_interval_sec", 60))
#         self.csv_path     = cfg.get("output_csv", "./analytics/occupancy.csv")
#         self.zone_names   = zone_names

#         os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)

#         # Khởi tạo CSV nếu chưa có header
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, "w", newline="") as f:
#                 w = csv.writer(f)
#                 w.writerow(["timestamp", "zone", "count", "peak"])

#         # Đếm hiện tại: {zone: set(track_id)}
#         self._current: dict[str, set] = defaultdict(set)
#         # Peak trong khoảng log: {zone: int}
#         self._peak: dict[str, int] = defaultdict(int)

#         self._next_log = time.time() + self.log_interval

#     def update(self, detections: list[Detection]) -> None:
#         if not self.enabled:
#             return
#         # Reset current count
#         for z in self.zone_names:
#             self._current[z] = set()

#         for det in detections:
#             if det.zone:
#                 self._current[det.zone].add(det.track_id)

#         # Cập nhật peak
#         for z in self.zone_names:
#             cnt = len(self._current[z])
#             if cnt > self._peak[z]:
#                 self._peak[z] = cnt

#     def draw(self, frame: np.ndarray) -> None:
#         """Hiển thị bảng đếm góc trái trên."""
#         if not self.enabled:
#             return
#         y = 20
#         cv2.putText(frame, "Occupancy", (10, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         y += 20
#         for z in self.zone_names:
#             cnt  = len(self._current.get(z, set()))
#             peak = self._peak.get(z, 0)
#             text = f"  {z}: {cnt} (peak {peak})"
#             cv2.putText(frame, text, (10, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
#             y += 18

#     def tick(self, t_now: float) -> None:
#         if not self.enabled:
#             return
#         if t_now >= self._next_log:
#             self._log(t_now)
#             # Reset peak
#             for z in self.zone_names:
#                 self._peak[z] = 0
#             self._next_log = t_now + self.log_interval

#     def _log(self, t_now: float) -> None:
#         ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_now))
#         with open(self.csv_path, "a", newline="") as f:
#             w = csv.writer(f)
#             for z in self.zone_names:
#                 cnt  = len(self._current.get(z, set()))
#                 peak = self._peak.get(z, 0)
#                 w.writerow([ts, z, cnt, peak])
#         log.info("Occupancy logged @ %s", ts)


# # ─────────────────────────────────────────────────────────────────────────────
# # Facade
# # ─────────────────────────────────────────────────────────────────────────────

# class AnalyticsManager:
#     """Tập hợp 3 module analytics, giao tiếp qua 1 interface duy nhất."""

#     def __init__(self, analytics_cfg: dict, frame_size: tuple[int, int],
#                  zone_names: list[str] | None = None):
#         self.heatmap    = CrowdHeatmap(analytics_cfg.get("heatmap", {}), frame_size)
#         self.trajectory = TrajectoryDrawer(analytics_cfg.get("trajectory", {}))
#         self.occupancy  = OccupancyCounter(
#             analytics_cfg.get("occupancy", {}),
#             zone_names or [],
#         )

#     def update(self, detections: list[Detection], t_now: float) -> None:
#         self.heatmap.update(detections)
#         self.trajectory.update(detections)
#         self.occupancy.update(detections)

#     def draw(self, frame: np.ndarray) -> None:
#         self.heatmap.draw(frame)
#         self.trajectory.draw(frame)
#         self.occupancy.draw(frame)

#     def tick(self, t_now: float) -> None:
#         self.heatmap.tick(t_now)
#         self.occupancy.tick(t_now)
"""
analytics.py
────────────
P2 – Crowd heatmap · Trajectory · Occupancy counter

Cách dùng:
    analytics = AnalyticsManager(cfg["analytics"], frame_size=(w, h))

    # Mỗi frame, sau khi có detections:
    analytics.update(detections, t_now)
    analytics.draw(frame)

    # Mỗi frame:
    analytics.tick(t_now)
"""

from __future__ import annotations
import os
import csv
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Detection record (truyền vào từ main loop)
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
        self._map *= self.decay
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
        if not self.enabled or self._map.max() < 1e-3:
            return
        norm    = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        mask    = (norm > 10).astype(np.uint8)
        roi     = cv2.bitwise_and(colored, colored, mask=mask)
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trajectory
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDrawer:
    """Vẽ đuôi trajectory cho mỗi track, màu fade dần."""

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
                alpha = i / n
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
        # Mặc định 30s để dễ kiểm tra (thay vì 60s)
        self.log_interval = float(cfg.get("log_interval_sec", 30))
        self.csv_path     = cfg.get("output_csv", "./analytics/occupancy.csv")
        self.zone_names   = list(zone_names)

        # Dùng abspath để đảm bảo thư mục luôn đúng
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        os.makedirs(csv_dir, exist_ok=True)

        # Ghi header nếu file chưa tồn tại
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            if write_header:
                csv.writer(f).writerow(["timestamp", "zone", "count", "peak"])
            f.flush()

        # Dùng dict thường, KHÔNG dùng defaultdict — tránh bug peak = 0 mãi
        self._current: dict[str, set] = {z: set() for z in self.zone_names}
        self._peak:    dict[str, int] = {z: 0     for z in self.zone_names}

        # _next_log = NOW + interval → ghi sau đúng 1 interval
        self._next_log = time.time() + self.log_interval

        print(f"[OccupancyCounter] CSV: {os.path.abspath(self.csv_path)}")
        print(f"[OccupancyCounter] Interval: {self.log_interval}s | Zones: {self.zone_names}")

    def update(self, detections: list[Detection]) -> None:
        if not self.enabled:
            return

        # Reset count zone đã biết
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
# Facade
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