import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm

# --- CẤU HÌNH HỆ THỐNG ---
print("🚀 Đang khởi động hệ thống trích xuất đặc trưng (Pose + Optical Flow)...")
yolo_model = YOLO('yolov8n.pt')

mp_pose = mp.solutions.pose
# model_complexity=2 cho độ chính xác cao nhất, phù hợp cho trích xuất dữ liệu
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQUENCE_LENGTH = 30 
FEATURE_SIZE = 231  # 33 keypoints * 4 (x,y,z,vis) + 33 * 3 (dx,dy,dz) = 231
FLOW_IMG_SIZE = 64  

def extract_features(video_path, flip=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Không thể mở video: {video_path}")
        return np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE)), np.zeros((SEQUENCE_LENGTH, FLOW_IMG_SIZE, FLOW_IMG_SIZE))

    frames_pose = []
    frames_flow = []
    
    prev_landmarks = None 
    prev_gray_crop = None
    
    while cap.isOpened() and len(frames_pose) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret: break
        
        if flip:
            frame = cv2.flip(frame, 1)

        # 1. PHÁT HIỆN NGƯỜI BẰNG YOLO
        # Giảm imgsz xuống 640 để cân bằng tốc độ & độ chính xác khi trích xuất hàng loạt
        results_yolo = yolo_model(frame, classes=[0], imgsz=640, verbose=False)
        boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
        
        current_pose_data = np.zeros(FEATURE_SIZE).tolist()
        current_flow_img = np.zeros((FLOW_IMG_SIZE, FLOW_IMG_SIZE), dtype=np.uint8)

        if len(boxes) > 0:
            # 🛠️ FIX QUAN TRỌNG: Thay vì gom tất cả box thành 1 khung, ta chỉ lấy người LỚN NHẤT/CÓ CONF CAO NHẤT
            confs = results_yolo[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs) # Hoặc dùng diện tích box nếu muốn
            x1, y1, x2, y2 = map(int, boxes[best_idx])

            # Padding 10%
            h, w = frame.shape[:2]
            pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
            x1, y1 = max(0, x1-pad_x), max(0, y1-pad_y)
            x2, y2 = min(w, x2+pad_x), min(h, y2+pad_y)

            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.shape[0] >= 30 and cropped_frame.shape[1] >= 30:
                cropped_frame = cv2.resize(cropped_frame, (256, 256), interpolation=cv2.INTER_CUBIC)
                
                # --- NHÁNH 1: TRÍCH XUẤT XƯƠNG KHỚP (POSE) ---
                image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image_rgb)
                
                if pose_results.pose_landmarks:
                    coords = []
                    deltas = []
                    for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                        cx, cy, cz, cv_val = lm.x, lm.y, lm.z, lm.visibility
                        coords.extend([cx, cy, cz, cv_val])
                        
                        if prev_landmarks is not None:
                            px, py, pz = prev_landmarks[i*4], prev_landmarks[i*4+1], prev_landmarks[i*4+2]
                            deltas.extend([cx - px, cy - py, cz - pz])
                        else:
                            deltas.extend([0.0, 0.0, 0.0])
                    
                    current_pose_data = coords + deltas
                    prev_landmarks = coords[:]  # Copy giá trị, tránh tham chiếu
                
                # --- NHÁNH 2: TRÍCH XUẤT DÒNG QUANG HỌC (OPTICAL FLOW) ---
                gray_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_crop, (FLOW_IMG_SIZE, FLOW_IMG_SIZE))
                
                if prev_gray_crop is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray_crop, gray_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    # Lọc nhiễu chuyển động nhỏ (< 2.0 pixel)
                    magnitude[magnitude < 2.0] = 0 
                    
                    # Chuẩn hóa an toàn về 0-255 (tránh chia cho 0 nếu frame tĩnh)
                    if magnitude.max() > magnitude.min():
                        current_flow_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                        current_flow_img = np.zeros_like(magnitude, dtype=np.uint8)
                
                prev_gray_crop = gray_resized.copy()  # Copy để tránh tham chiếu
            
        frames_pose.append(current_pose_data)
        frames_flow.append(current_flow_img)
            
    cap.release()
    
    # Bù đủ 30 frame nếu video ngắn hoặc mất dấu giữa chừng
    while len(frames_pose) < SEQUENCE_LENGTH:
        frames_pose.append(np.zeros(FEATURE_SIZE).tolist())
        frames_flow.append(np.zeros((FLOW_IMG_SIZE, FLOW_IMG_SIZE), dtype=np.uint8))
        
    return np.array(frames_pose, dtype=np.float32), np.array(frames_flow, dtype=np.float32)

def process_dataset(dataset_path, output_path):
    classes = ['non_violence', 'violence']
    pose_dir = os.path.join(output_path, "pose_data")
    flow_dir = os.path.join(output_path, "flow_data")
    
    for class_name in classes:
        os.makedirs(os.path.join(pose_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(flow_dir, class_name), exist_ok=True)
        
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir): 
            print(f"⚠️ Không tìm thấy thư mục: {class_dir}")
            continue
            
        # Lọc chỉ lấy file video hợp lệ
        video_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_name in tqdm(video_files, desc=f"📂 Xử lý lớp: {class_name}"):
            video_path = os.path.join(class_dir, video_name)
            base_name = os.path.splitext(video_name)[0]
            
            try:
                # --- XỬ LÝ 1: VIDEO GỐC ---
                pose_norm, flow_norm = extract_features(video_path, flip=False)
                np.save(os.path.join(pose_dir, class_name, f"{base_name}_norm.npy"), pose_norm)
                np.save(os.path.join(flow_dir, class_name, f"{base_name}_norm.npy"), flow_norm)
                
                # --- XỬ LÝ 2: VIDEO LẬT (Data Augmentation) ---
                pose_flip, flow_flip = extract_features(video_path, flip=True)
                np.save(os.path.join(pose_dir, class_name, f"{base_name}_flip.npy"), pose_flip)
                np.save(os.path.join(flow_dir, class_name, f"{base_name}_flip.npy"), flow_flip)
                
            except Exception as e:
                print(f"\n❌ Lỗi xử lý {video_name}: {e}")
                continue
            
    print("\n✅ HOÀN THÀNH! Dữ liệu đã được trích xuất và lưu thành công.")

if __name__ == "__main__":
    process_dataset("../dataset", "../extracted_data")