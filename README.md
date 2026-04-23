# Vision AI Surveillance System (YOLO + Depth Anything V2)

- Hệ thống giám sát thông minh kết hợp Nhận diện, Ước lượng độ sâu và Phân tích hành vi thời gian thực.
## Điểm nổi bật kỹ thuật
- Hệ thống này không chỉ dừng lại ở việc phát hiện đối tượng cơ bản mà còn tích hợp các kỹ thuật xử lý thị giác máy tính chuyên sâu:
    - Pseudo-3D Perception: Kết hợp YOLO để phát hiện vật thể và Depth Anything V2 để ước lượng khoảng cách, tạo ra không gian giám sát 3D từ camera 2D đơn tiêu.
    - Spatial Intelligence: Sử dụng phép biến đổi Homography (Perspective Transform) để ánh xạ tọa độ từ hình ảnh (Pixel) sang mặt phẳng thực tế (World coordinates), giúp tính toán vận tốc và vị trí chính xác hơn.
    - Multi-threaded Pipeline: Tối ưu hóa hiệu năng bằng cách chạy các tác vụ nặng (Inference Depth) trên các luồng (Thread) riêng biệt, đảm bảo FPS của hệ thống luôn ổn định.

## Tính năng chính
### 1. Phân tích hành vi (Behavior Analysis)
- Hệ thống tự động phân loại mức độ cảnh báo (SAFE → CRITICAL) dựa trên sự kết hợp giữa vị trí trong vùng cấm và trạng thái di chuyển:
    - Intrusion: Xâm nhập vào các vùng giới hạn (Restricted Zone).
    - Running: Phát hiện đối tượng di chuyển với vận tốc bất thường.
    - Approaching: Cảnh báo khi có đối tượng tiến lại quá gần camera dựa trên sự thay đổi giá trị Depth.
    - Loitering: Phát hiện đối tượng đứng lâu (lảng vãng) trong khu vực nhạy cảm.
    - Phân mức cảnh báo:
        - SAFE
        - MEDIUM
        - HIGH
        - CRITICAL

### 2. Trực quan hóa dữ liệu (Analytics)
- Crowd Heatmap: Bản đồ nhiệt hiển thị mật độ xuất hiện của đối tượng, có cơ chế tự động mờ dần (decay) theo thời gian.
- Trajectory Tracking: Vẽ lại quỹ đạo di chuyển của từng ID đối tượng để theo dõi lộ trình.
- Occupancy Counter: Đếm số người hiện diện trong từng Zone và lưu trữ dữ liệu vào file CSV để phân tích peak hour.

### 3. Tối ưu hóa phần cứng
- Hỗ trợ chạy tăng tốc trên Apple Silicon (MPS) hoặc CPU.
- Cơ chế Depth Sampling: Chỉ lấy mẫu độ sâu tại vùng Bounding Box (percentile 20th) để loại bỏ nhiễu background.


## Kiến trúc hệ thống

                graph TD
                    Input(Video/Webcam) --> YOLO(YOLO Detection & Tracking)
                    Input --> Depth(Depth Anything V2 - Async Thread)
                    YOLO --> Spatial(Homography Mapping)
                    Depth --> Spatial
                    Spatial --> Behavior(Behavior Analysis & Alerts)
                    Behavior --> Analytics(Heatmap/Trajectory/Occupancy)
                    Analytics --> Output(Visualization & CSV Logs)
                    
## Cấu trúc project
.
├── main.py   # main pipeline
├── analysis.py              # analytics modules
├── Object_detection/
│   └── Model/yolo11n.pt     # model YOLO
├── Data/
│   └── test/                # video test
└── analytics/
    ├── heatmaps/            # lưu heatmap
    └── occupancy.csv        # log occupancy



## Hướng dẫn cài đặt & Sử dụng
### Yêu cầu hệ thống
- Python 3.9+
- PyTorch, OpenCV, Ultralytics, Transformers
### Cách chạy
- Chạy thực tế với Video: python main.py --source ./Data/test/test4.MOV

- Chạy trực tiếp từ Webcam: python main.py --source 0

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
- Link: https://drive.google.com/drive/u/0/folders/1rC_ILQNHK5di0YrAM1fTnRPsI01nJa65

## Hạn chế hiện tại
- Depth estimation là relative, không phải metric (mét)
- Behavior detection dựa trên rule (threshold)
- Chưa hỗ trợ multi-camera
- Chưa deploy (web / API)

## Lộ trình phát triển (Roadmap)
- Tích hợp nhận diện biển số xe (OCR) và thông báo qua hoặc mail.
- Chuyển đổi model sang định dạng ONNX/CoreML để tối ưu hóa thêm cho macOS.
- Phát triển giao diện Web Dashboard quản lý tập trung.
