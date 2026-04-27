import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import deque
import tempfile
import os
import shutil

# =====================================================================
# 🛠️ BẢNG CẤU HÌNH DOANH NGHIỆP (ENTERPRISE CONTROL PANEL)
# =====================================================================
CONFIDENCE_THRESHOLD = 0.85       # Ngưỡng tự tin tối thiểu của AI Keras
FLOW_VIOLENCE_MIN = 25.0          # Lực chuyển động tối thiểu để xác nhận bạo lực
FLOW_SUSPECT_MIN = 180.0          # Mức rung lắc để kích hoạt cảnh báo nghi vấn
DISTANCE_MULTIPLIER = 1.5         # Bán kính va chạm = 1.5x kích thước cơ thể
VOTING_WINDOW = 5                 # Độ dài bộ nhớ voting (số frame)
VOTES_FOR_VIOLENCE = 2            # Số phiếu "Đỏ" cần để báo động
VOTES_FOR_SUSPECT = 3             # Số phiếu "Cam" cần để cảnh báo nghi vấn
INFERENCE_INTERVAL = 5            # Chạy model mỗi N frame để tối ưu tốc độ
# =====================================================================

st.set_page_config(page_title="🛡️ AI An Ninh: Phát Hiện Bạo Lực", layout="wide", page_icon="🎥")

@st.cache_resource
def load_all_models():
    """Khởi tạo mô hình một lần duy nhất, tái sử dụng cho mọi session"""
    dual_model = load_model("models/dual_stream_model.h5")
    yolo_model = YOLO('yolov8n.pt')
    return dual_model, yolo_model

def cleanup_temp_files():
    """Dọn dẹp file tạm sau khi xử lý xong"""
    try:
        for f in tempfile.listdir():
            if f.startswith('tmp') or f.endswith('.webm'):
                os.remove(os.path.join(tempfile.gettempdir(), f))
    except:
        pass

# === KHỞI TẠO ỨNG DỤNG ===
st.title("🛡️ Hệ Thống AI Luồng Kép: Phát Hiện Bạo Lực")
st.markdown("*Chuẩn Enterprise - Production Ready*")

# Sidebar cấu hình nhanh
with st.sidebar:
    st.header("⚙️ Tinh Chỉnh Tham Số")
    CONFIDENCE_THRESHOLD = st.slider("Ngưỡng tự tin AI", 0.5, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    VOTES_FOR_VIOLENCE = st.slider("Số phiếu báo động", 1, 5, VOTES_FOR_VIOLENCE)
    st.info("💡 Mẹo: Giảm ngưỡng tự tin để phát hiện nhạy hơn, nhưng có thể tăng false positive.")

# Load model
with st.spinner("🔄 Đang tải mô hình AI..."):
    model, yolo_model = load_all_models()
st.success("✅ Mô hình đã sẵn sàng!")

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

FEATURE_SIZE = 231 
FLOW_IMG_SIZE = 64

# Upload video
uploaded_file = st.file_uploader("📂 Tải video CCTV kiểm thử", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Lưu video upload vào file tạm
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile_in.write(uploaded_file.read())
    video_path = tfile_in.name
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎬 Video Gốc")
        st.video(video_path)
    
    if st.button("🚀 Bắt Đầu Phân Tích AI", type="primary", use_container_width=True):
        
        with st.status("🔄 Đang xử lý video...", expanded=True) as status:
            
            # === KHỞI TẠO VIDEO PROCESSING ===
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 24)  # Đảm bảo fps hợp lệ
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
            out_path = tfile_out.name
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            # Progress UI
            progress_bar = st.progress(0)
            stats_placeholder = st.empty()
            
            # === KHỞI TẠO BỘ NHỚ THEO TRACK_ID ===
            track_data = {}  # Dict chứa tất cả history cho mỗi person
            
            # State machine cho alarm (tránh flash 1 frame)
            alarm_state = {"active": False, "cooldown": 0, "type": None}
            
            frame_count = 0
            stats = {"processed": 0, "violence_detected": 0, "suspect_detected": 0}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                display_frame = frame.copy()
                frame_count += 1
                
                # Cập nhật progress
                progress = min(int((frame_count / total_frames) * 100), 100)
                progress_bar.progress(progress)
                
                # === YOLO TRACKING ===
                yolo_results = yolo_model.track(frame, classes=[0], imgsz=640, 
                                               persist=True, tracker="botsort.yaml", verbose=False)
                
                current_frame_centers = {}  # Lưu vị trí & kích thước từng người trong frame này
                
                if yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    track_ids = yolo_results[0].boxes.id.int().cpu().tolist()

                    # Thu thập thông tin vị trí tất cả người trong frame
                    for box, track_id in zip(boxes, track_ids):
                        w_box = box[2] - box[0]
                        h_box = box[3] - box[1]
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        current_frame_centers[track_id] = (cx, cy, w_box, h_box)

                    # Xử lý từng người được phát hiện
                    for box, track_id in zip(boxes, track_ids):
                        # Khởi tạo bộ nhớ cho track_id mới
                        if track_id not in track_data:
                            track_data[track_id] = {
                                'pose_history': deque(maxlen=30),
                                'flow_history': deque(maxlen=30),
                                'prev_landmarks': None,
                                'prev_gray': None,
                                'voting': deque(maxlen=VOTING_WINDOW),
                                'last_decision': 0
                            }

                        x1, y1, x2, y2 = map(int, box)
                        
                        # Padding 10% quanh bounding box
                        h, w = frame.shape[:2]
                        px, py = int((x2-x1)*0.1), int((y2-y1)*0.1)
                        x1, y1 = max(0, x1-px), max(0, y1-py)
                        x2, y2 = min(w, x2+px), min(h, y2+py)

                        cropped_person = frame[y1:y2, x1:x2]
                        feature_vector = np.zeros(FEATURE_SIZE).tolist()
                        current_flow_img = np.zeros((FLOW_IMG_SIZE, FLOW_IMG_SIZE), dtype=np.uint8)

                        # === TRÍCH XUẤT ĐẶC TRƯNG ===
                        if cropped_person.shape[0] >= 30 and cropped_person.shape[1] >= 30:
                            cropped_resized = cv2.resize(cropped_person, (256, 256), interpolation=cv2.INTER_CUBIC)
                            
                            # Nhánh Pose
                            img_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
                            pose_results = pose.process(img_rgb)
                            
                            if pose_results.pose_landmarks:
                                coords, deltas = [], []
                                for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                                    cx, cy, cz, cv_val = lm.x, lm.y, lm.z, lm.visibility
                                    coords.extend([cx, cy, cz, cv_val])
                                    
                                    if track_data[track_id]['prev_landmarks'] is not None:
                                        prev = track_data[track_id]['prev_landmarks']
                                        deltas.extend([cx - prev[i*4], cy - prev[i*4+1], cz - prev[i*4+2]])
                                    else:
                                        deltas.extend([0.0, 0.0, 0.0])
                                
                                feature_vector = coords + deltas
                                # Lưu copy để tránh reference bug
                                track_data[track_id]['prev_landmarks'] = coords[:]
                                
                                # Vẽ skeleton lên frame crop
                                mp_drawing.draw_landmarks(cropped_resized, pose_results.pose_landmarks, 
                                                         mp_pose.POSE_CONNECTIONS)
                                display_frame[y1:y2, x1:x2] = cv2.resize(cropped_resized, (x2-x1, y2-y1))
                            
                            # Nhánh Optical Flow (🚀 OPTIMIZATION: Dùng absdiff thay Farneback cho tốc độ)
                            gray_crop = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2GRAY)
                            gray_resized = cv2.resize(gray_crop, (FLOW_IMG_SIZE, FLOW_IMG_SIZE))
                            
                            if track_data[track_id]['prev_gray'] is not None:
                                # 🚀 Thay Farneback bằng absdiff: nhanh hơn 5-10x, đủ cho phát hiện chuyển động mạnh
                                diff = cv2.absdiff(track_data[track_id]['prev_gray'], gray_resized)
                                current_flow_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            
                            track_data[track_id]['prev_gray'] = gray_resized.copy()
                        
                        # Lưu vào history
                        track_data[track_id]['pose_history'].append(feature_vector)
                        track_data[track_id]['flow_history'].append(current_flow_img)
                        
                        # === LUỒNG DỰ ĐOÁN & VOTING ===
                        if len(track_data[track_id]['pose_history']) == 30 and frame_count % INFERENCE_INTERVAL == 0:
                            
                            # Chuẩn bị input cho model
                            input_pose = np.expand_dims(np.array(track_data[track_id]['pose_history']), axis=0)
                            input_flow = np.array(track_data[track_id]['flow_history'], dtype=np.float32) / 255.0
                            input_flow = np.expand_dims(np.expand_dims(input_flow, axis=0), axis=-1)
                            
                            # Dự đoán
                            res = model.predict([input_pose, input_flow], verbose=0)[0]
                            action_idx = np.argmax(res)
                            confidence = res[action_idx]
                            
                            current_decision = 0
                            display_label = "Binh thuong"
                            flow_intensity = np.mean(current_flow_img)
                            
                            # Logic xác thực bạo lực đa yếu tố
                            if action_idx == 1 and confidence > CONFIDENCE_THRESHOLD:
                                # Kiểm tra không gian: có người khác trong bán kính va chạm không?
                                my_cx, my_cy, my_w, my_h = current_frame_centers[track_id]
                                safe_radius = max(my_w, my_h) * DISTANCE_MULTIPLIER
                                
                                is_valid_violence = any(
                                    np.sqrt((my_cx - ox)**2 + (my_cy - oy)**2) < safe_radius
                                    for oid, (ox, oy, _, _) in current_frame_centers.items()
                                    if oid != track_id
                                )
                                
                                if is_valid_violence and flow_intensity > FLOW_VIOLENCE_MIN:
                                    current_decision = 1  # Bạo lực xác nhận
                                    display_label = f"DANH NHAU! ({confidence:.2f})"
                                    stats["violence_detected"] += 1
                                else:
                                    display_label = "Loai bo (khong gian/flow)"
                            
                            elif flow_intensity > FLOW_SUSPECT_MIN:
                                current_decision = 2  # Nghi vấn
                                display_label = f"NGHI VAN ({flow_intensity:.0f})"
                                stats["suspect_detected"] += 1
                            
                            # Lưu decision vào voting history
                            track_data[track_id]['voting'].append(current_decision)
                            track_data[track_id]['last_decision'] = current_decision
                        
                        # === HIỂN THỊ KẾT QUẢ CUỐI CÙNG (VOTING) ===
                        color = (0, 255, 0)  # Xanh lá = an toàn
                        final_label = "Dang phan tich..."
                        
                        votes = track_data[track_id]['voting']
                        if len(votes) > 0:
                            violence_votes = sum(1 for v in votes if v == 1)
                            suspect_votes = sum(1 for v in votes if v == 2)
                            
                            if violence_votes >= VOTES_FOR_VIOLENCE:
                                color = (0, 0, 255)  # Đỏ = bạo lực
                                final_label = "⚠️ BAO LUC!"
                            elif suspect_votes >= VOTES_FOR_SUSPECT:
                                color = (0, 165, 255)  # Cam = nghi vấn
                                final_label = "❓ NGHI VAN"
                            else:
                                final_label = track_data[track_id].get('last_display', "An toan")
                        
                        track_data[track_id]['last_display'] = final_label
                        
                        # Vẽ bounding box + label
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"ID:{track_id} {final_label}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # === THANH TRẠNG THÁI TỔNG QUÁT (State Machine) ===
                # Vẽ nền header
                cv2.rectangle(display_frame, (0, 0), (min(650, width), 55), (20, 20, 20), -1)
                
                # Quản lý alarm state với cooldown
                has_violence = any(
                    sum(1 for v in td['voting'] if v == 1) >= VOTES_FOR_VIOLENCE 
                    for td in track_data.values() if len(td['voting']) > 0
                )
                
                if has_violence and alarm_state["cooldown"] == 0:
                    alarm_state = {"active": True, "cooldown": 60, "type": "VIOLENCE"}  # 60 frames ~ 2-3s
                elif alarm_state["cooldown"] > 0:
                    alarm_state["cooldown"] -= 1
                
                # Hiển thị status banner
                if alarm_state["active"] and alarm_state["type"] == "VIOLENCE":
                    cv2.putText(display_frame, "🚨 CANH BAO: PHAT HIEN BAO LUC!", (15, 38), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "✅ TRANG THAI: AN TOAN", (15, 38), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Ghi frame vào output video
                out.write(display_frame)
                stats["processed"] += 1
                
                # Cập nhật stats UI mỗi 30 frame
                if frame_count % 30 == 0:
                    stats_placeholder.metric("📊 Tiến độ", f"{frame_count}/{total_frames} frames", 
                                           f"🔴 {stats['violence_detected']} bạo lực | 🟠 {stats['suspect_detected']} nghi vấn")
            
            # === HOÀN TẤT XỬ LÝ ===
            cap.release()
            out.release()
            
            # Dọn pose resource
            pose.close()
            
            status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
            
            # Hiển thị kết quả
            st.markdown("### 🔍 Video Kết Quả")
            video_bytes = open(out_path, 'rb').read()
            st.video(video_bytes, format="video/webm")
            
            # Nút download
            st.download_button(
                label="📥 Tải Video Đã Xử Lý",
                data=video_bytes,
                file_name="analyzed_video.webm",
                mime="video/webm"
            )
            
            # Thống kê cuối
            st.markdown("### 📈 Báo Cáo Phân Tích")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🎬 Frames xử lý", stats["processed"])
            col_b.metric("⚠️ Cảnh báo bạo lực", stats["violence_detected"])
            col_c.metric("❓ Cảnh báo nghi vấn", stats["suspect_detected"])
            
            # Cleanup temp files
            try:
                os.unlink(video_path)
                # Giữ out_path để user download, sẽ cleanup ở session next
            except:
                pass

# Footer
st.markdown("---")
st.caption("🛡️ AI Violence Detection System | Dual-Stream Architecture | Pose + Optical Flow")