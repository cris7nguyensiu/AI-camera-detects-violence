#  AI Violence Detection System

Hệ thống AI phát hiện bạo lực qua camera sử dụng Pose Estimation và Optical Flow.

##  Tính năng
- Phát hiện bạo lực trong video CCTV
- Sử dụng YOLOv8 + MediaPipe Pose
- Dual-Stream Architecture (Transformer + CNN)
- Giao diện Streamlit thân thiện

##  Cài đặt

1. Clone repository:
```bash
git clone https://github.com/cris7nguyensiu/AI-camera-detects-violence.git
cd AI-camera-detects-violence
2. Cài đặt dependencies:
pip install -r requirements.txt
3. Download YOLO weights:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
## Cách sử dụng
Bước 1: Trích xuất đặc trưng
python extract_data.py
Bước 2: Huấn luyện mô hình
python train_model.py
Bước 3: Chạy ứng dụng
streamlit run app.py