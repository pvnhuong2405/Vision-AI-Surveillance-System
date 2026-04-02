from time import time
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from transformers import pipeline
from PIL import Image
from analysis import AnalyticsManager, Detection

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")


yolo = YOLO("./Object_detection/Model/yolo11n.pt")


depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if torch.backends.mps.is_available() else -1
)


# ─── 3. DEPTH UTILS ────────────────────────────────────────────────────────────

def get_depth_map(frame_rgb):
    # convert OpenCV -> PIL
    pil_img = Image.fromarray(frame_rgb)

    # input cho transformers pipeline
    result = depth_pipe(pil_img)

    depth = np.array(result["depth"])

    # resize về đúng size frame
    depth = cv2.resize(depth, (frame_rgb.shape[1], frame_rgb.shape[0]))

    # normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # đảo để: gần = lớn
    depth = 1.0 - depth

    return depth


def sample_depth_at_box(depth_map: np.ndarray, box: list, percentile: int = 20) -> float:
    """
    Lấy giá trị depth đại diện trong bounding box.
    Dùng percentile thấp (gần camera nhất) để tránh nhiễu background.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.percentile(roi, percentile))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Chuẩn hóa về [0,1] để hiển thị."""
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map)
    return (depth_map - d_min) / (d_max - d_min)


# ─── 4. ALERT LOGIC ────────────────────────────────────────────────────────────
# MiDaS cho giá trị TƯƠNG ĐỐI: giá trị CAO = GẦN hơn
DANGER_THRESHOLD = 0.75
WARNING_THRESHOLD = 0.55

def get_alert_level(depth_score: float) -> str:
    if depth_score > DANGER_THRESHOLD:
        return "DANGER"
    elif depth_score > WARNING_THRESHOLD:
        return "WARNING"
    return "SAFE"


# ─── 5. 3D BOUNDING BOX (pseudo-3D từ depth) ──────────────────────────────────
def draw_3d_box(frame, box, depth_score, color):
    """
    Vẽ pseudo-3D box: dịch chuyển box ra phía sau theo depth.
    depth_score cao → object gần → offset nhỏ.
    depth_score thấp → object xa → offset lớn.
    """
    x1, y1, x2, y2 = map(int, box)
    offset = max(4, int((1 - depth_score) * 20))  # 4–20px

    # Mặt trước
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Mặt sau
    bx1, by1 = x1 - offset, y1 - offset
    bx2, by2 = x2 - offset, y2 - offset
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

    # Nối 4 cạnh
    for p1, p2 in [
        ((x1, y1), (bx1, by1)),
        ((x2, y1), (bx2, by1)),
        ((x2, y2), (bx2, by2)),
        ((x1, y2), (bx1, by2)),
    ]:
        cv2.line(frame, p1, p2, color, 1)



# ─── 6. VISUALIZATION ─────────────────────────────────────────────────────────
ALERT_COLORS = {
    "DANGER":   (0, 0, 255),
    "WARNING":  (0, 165, 255),
    "SAFE":     (0, 200, 80),
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
DEFAULT_ZONE_STYLE = {
    "fill": (180, 0, 180), "border": (150, 0, 150), "alpha": 0.15
}
def draw_label(frame, text, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    
# ─── 7. HOMOGRAPHY SETUP ──────────────────────────────────────────────────────
# 4 điểm trên ảnh (pixel) tương ứng với 4 góc thực tế đã biết
# Chỉnh lại src_pts và dst_pts cho phù hợp với camera 
# kéo sang trái, kéo sang phải
_SRC = np.float32([
    [660, 640],   # top-left  
    [1460, 450],  # top-right 
    [1950, 650],  # bottom-right 
    [950, 800],   # bottom-left  
])

# 4 điểm tương ứng trong world space (cm)
_DST = np.float32([
    [0,   0],
    [700, 0],
    [700, 500],
    [0,   500],
])

H     = cv2.getPerspectiveTransform(_SRC, _DST)   # image  → world
H_INV = cv2.getPerspectiveTransform(_DST, _SRC)   # world  → image

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

        label      = zone_name.upper()
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
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
            False
        )
        if inside >= 0:
            return zone_name
    return None


# ─── 10. MAIN LOOP ──────────────────────────────────────────────────────────────
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./Data/test")
    opt = parser.parse_args()

   
    source = int(opt.source) if opt.source.isdigit() else opt.source
    cap = cv2.VideoCapture(source)

    # ===== read first frame để init analytics =====
    ret, frame = cap.read()
    if not ret:
        print("Cannot read source")
        return

    h, w = frame.shape[:2]

    # ===== INIT ANALYTICS =====
    from analysis import AnalyticsManager, Detection

    analytics_cfg = {
        "heatmap": {"enabled": True},
        "trajectory": {"enabled": True},
        "occupancy": {"enabled": True}
    }

    analytics = AnalyticsManager(
        analytics_cfg,
        frame_size=(w, h),
        zone_names=list(ZONES_WORLD.keys())
    )

    # ===== tracking storage =====
    tracks = defaultdict(lambda: {
        "history": deque(maxlen=30),
        "zone": None,
        "zone_time": 0,
        "last_update": None
    })

    fps_deque = deque(maxlen=30)

    # ===== depth cache =====
    DEPTH_EVERY_N = 5   # tăng FPS
    frame_count = 0
    cached_depth = None

    print("Start processing...")

    while True:
        t_start = time()

        ret, frame = cap.read()
        if not ret:
            break

        t_now = time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ===== YOLO tracking =====
        results = yolo.track(frame, persist=True, verbose=False, classes=[0])
        result = results[0]

        # ===== DEPTH =====
        frame_count += 1
        if frame_count % DEPTH_EVERY_N == 0 or cached_depth is None:
            cached_depth = get_depth_map(frame_rgb)

        depth_map = cached_depth
        depth_norm = normalize_depth(depth_map)

        # ===== visualize depth =====
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        thumb = cv2.resize(depth_color, (w // 4, h // 4))
        frame[0:h // 4, w - w // 4:w] = thumb

        # ===== zones =====
        draw_zones(frame, ZONES_WORLD)

        # ===== detections + analytics input =====
        detections_for_analytics = []

        boxes = result.boxes
        if boxes is not None and len(boxes):

            ids   = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
            xyxys = boxes.xyxy.cpu().numpy()

            for box, track_id in zip(xyxys, ids):
                if track_id is None:
                    continue

                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                # ===== clamp bbox =====
                h_d, w_d = depth_norm.shape
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_d, x2), min(h_d, y2)

                roi = depth_norm[y1_c:y2_c, x1_c:x2_c]

                # ===== depth robust =====
                z = float(np.percentile(roi, 20)) if roi.size > 0 else 0.0

                # ===== position =====
                cx = (x1 + x2) // 2
                cy = y2

                current_zone = is_in_zone(cx, cy, ZONES_WORLD)

                # ===== tracking =====
                track = tracks[track_id]
                track["history"].append((cx, cy, z, t_now))

                dt = 0
                if track["last_update"] is not None:
                    dt = t_now - track["last_update"]
                track["last_update"] = t_now

                velocity, dz = 0.0, 0.0
                if len(track["history"]) >= 2:
                    x_prev, y_prev, z_prev, _ = track["history"][-2]
                    dx, dy = cx - x_prev, cy - y_prev
                    velocity = np.sqrt(dx*dx + dy*dy) / (dt + 1e-5)
                    dz = z - z_prev

                # ===== zone time =====
                if track["zone"] == current_zone:
                    track["zone_time"] += dt
                else:
                    track["zone_time"] = 0
                track["zone"] = current_zone

                # ===== behavior =====
                intrusion = current_zone == "restricted" and z > 0.4
                running = velocity > 20
                approaching = dz > 0.01

                loitering = (
                    current_zone == "restricted"
                    and velocity < 2
                    and track["zone_time"] > 3
                )

                # ===== decision =====
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
                    "HIGH": (0, 165, 255),
                    "MEDIUM": (0, 255, 255),
                    "SAFE": (0, 255, 0)
                }[alert]

                draw_3d_box(frame, box, z, draw_color)
                draw_label(frame, f"ID:{track_id} {alert} z:{z:.2f}", x1, y1, draw_color)

                # ===== PUSH TO ANALYTICS =====
                detections_for_analytics.append(
                    Detection(track_id=track_id, cx=cx, cy=cy, zone=current_zone)
                )

        # ===== ANALYTICS =====
        analytics.update(detections_for_analytics, t_now)
        analytics.draw(frame)
        analytics.tick(t_now)

        # ===== FPS =====
        fps_deque.append(1.0 / (time() - t_start + 1e-6))
        fps = min(60, np.mean(fps_deque))

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO + Depth + Analytics", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()