# Vision AI Surveillance System (YOLO + Depth Anything V2)

## Giới thiệu
- Đây là một hệ thống Computer Vision thời gian thực phục vụ cho bài toán giám sát (surveillance), kết hợp giữa:

    - Nhận diện & tracking người
    - Ước lượng độ sâu từ ảnh (monocular depth)
    - Phân tích hành vi (behavior analysis)
    - Thống kê & trực quan hóa dữ liệu (analytics)


## Tính năng chính
### 1. Phát hiện & theo dõi đối tượng
- Sử dụng YOLO để detect người
- Tracking ID theo thời gian (multi-object tracking)
### 2. Ước lượng độ sâu (Depth Estimation)
- Sử dụng Depth Anything V2
- Độ sâu là tương đối (relative depth):
    - Giá trị cao → gần camera
    - Giá trị thấp → xa camera
- Có normalize & đảo chiều để dễ sử dụng
### 3. Phân tích hành vi (Behavior Analysis)
- Dựa trên:

    - vị trí
    - vận tốc
    - thay đổi độ sâu theo thời gian

- Hệ thống phát hiện:

    - Intrusion: xâm nhập vùng cấm
    - Running: di chuyển nhanh
    - Approaching: tiến gần camera
    - Loitering: đứng lâu trong vùng

- Phân mức cảnh báo:

    - SAFE
    - MEDIUM
    - HIGH
    - CRITICAL
### 4. Zone & Spatial Mapping
- Định nghĩa vùng (zone) theo tọa độ thực (world coordinates)
- Sử dụng homography để map:
    - Image → World
    - World → Image

---> Giúp hệ thống hiểu không gian giống thực tế (CCTV thật)

### 5. Analytics (Phân tích dữ liệu)
- Crowd Heatmap
    - Tích lũy mật độ người theo thời gian
    - Có decay (fade dần)
    - Có thể lưu ảnh định kỳ
- Trajectory
    - Vẽ quỹ đạo di chuyển của từng đối tượng
    - Hiển thị hướng & hành vi
- Occupancy Counter
    - Đếm số người theo từng zone
    - Log ra CSV theo thời gian
    - Theo dõi peak

### 6. Tối ưu hiệu năng (Real-time Optimization)
- Không chạy depth mỗi frame: DEPTH_EVERY_N = 5 
--> tăng FPS đáng kể

    - Cache depth map
    - Giảm tải cho model nặng

## Kiến trúc hệ thống
                    Video Input
                        ↓
                    YOLO Detection + Tracking
                        ↓
                    Depth Estimation (Depth Anything V2)
                        ↓
                    Spatial Mapping (Homography)
                        ↓
                    Behavior Analysis
                        ↓
                    Analytics (Heatmap / Trajectory / Occupancy)
                        ↓
                    Visualization (OpenCV)

## Cấu trúc project
                .
                ├── test_system_DeepIMG.py   # main pipeline
                ├── analysis.py              # analytics modules
                ├── Object_detection/
                │   └── Model/yolo11n.pt     # model YOLO
                ├── Data/
                │   └── test/                # video test
                └── analytics/
                    ├── heatmaps/            # lưu heatmap
                    └── occupancy.csv        # log occupancy



## Cách chạy
- Chạy với video:
python test_system_DeepIMG.py --source ./Data/test/test4.MOV
- Chạy webcam:
python test_system_DeepIMG.py --source 0

## Output
- Video hiển thị:
    - bounding box
    - cảnh báo
    - depth map (thumbnail)
    - trajectory
    - zone overlay
- File sinh ra:
    - analytics/heatmaps/*.png
    - analytics/occupancy.csv
      
## Demo
<video controls src="Demo.mov" title="Title"></video>

## Hạn chế hiện tại
- Depth estimation là relative, không phải metric (mét)
- Behavior detection dựa trên rule (threshold)
- Chưa hỗ trợ multi-camera
- Chưa deploy (web / API)

