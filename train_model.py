import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import random

# --- THIẾT LẬP RANDOM SEED CHO TÍNH TÁI LẬP ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# --- CẤU HÌNH ---
DATA_PATH = "../extracted_data"
MODEL_SAVE_PATH = "../models/dual_stream_model.h5"
EVAL_REPORT_PATH = "../models/evaluation_report.txt"
PLOT_PATH = "../models/training_history.png"

FEATURE_SIZE = 231
FLOW_IMG_SIZE = 64
SEQUENCE_LENGTH = 30
label_map = {'non_violence': 0, 'violence': 1}

# 1. TẢI DỮ LIỆU KÉP (Cả Pose và Flow)
print("📂 Đang nạp dữ liệu Luồng kép từ ổ cứng...")
X_pose, X_flow, y_labels = [], [], []

for action in label_map.keys():
    action_pose_dir = os.path.join(DATA_PATH, "pose_data", action)
    action_flow_dir = os.path.join(DATA_PATH, "flow_data", action)

    if not os.path.exists(action_pose_dir):
        print(f"⚠️ Bỏ qua thư mục không tồn tại: {action_pose_dir}")
        continue

    for file_name in os.listdir(action_pose_dir):
        pose_path = os.path.join(action_pose_dir, file_name)
        flow_path = os.path.join(action_flow_dir, file_name)

        if not os.path.exists(flow_path):
            print(f"⚠️ Không tìm thấy file flow tương ứng: {flow_path}")
            continue

        try:
            pose_data = np.load(pose_path)
            flow_data = np.load(flow_path)

            # Kiểm tra shape để đảm bảo khớp với mô hình
            if pose_data.shape[0] == SEQUENCE_LENGTH and flow_data.shape[0] == SEQUENCE_LENGTH:
                X_pose.append(pose_data)
                X_flow.append(flow_data)
                y_labels.append(label_map[action])
            else:
                print(f"⚠️ Shape không khớp cho {file_name}: Pose {pose_data.shape}, Flow {flow_data.shape}")
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {file_name}: {e}")

X_pose = np.array(X_pose, dtype=np.float32)
X_flow = np.array(X_flow, dtype=np.float32) / 255.0
X_flow = np.expand_dims(X_flow, axis=-1)
y = to_categorical(np.array(y_labels), num_classes=2)

print(f"✅ Đã nạp thành công {len(X_pose)} mẫu dữ liệu.")
print(f"📊 Phân bố nhãn: non_violence={sum(y_labels==0)}, violence={sum(y_labels==1)}")

# 2. CHIA DỮ LIỆU (CÂN BẰNG NHÃN VỚI STRATIFY)
print("\n⚖️ Đang chia tập Train/Test (Stratified)...")
X_pose_train, X_pose_test, X_flow_train, X_flow_test, y_train, y_test = train_test_split(
    X_pose, X_flow, y, test_size=0.2, random_state=SEED, stratify=y_labels)

print(f"🚂 Dữ liệu Huấn luyện: {X_pose_train.shape[0]} video | 🧪 Kiểm thử: {X_pose_test.shape[0]} video")

# TÍNH TRỌNG SỐ LỚP (CLASS WEIGHT) XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_labels)
class_weight_dict = {i: class_weights[i] for i in range(2)}
print(f"⚖️ Class Weights (Tự động cân bằng): {class_weight_dict}")

# 3. KIẾN TRÚC MẠNG LUỒNG KÉP (DUAL-STREAM)
print("\n🏗️ Đang xây dựng kiến trúc Dual-Stream...")

def build_pose_branch(input_shape):
    inputs = Input(shape=input_shape, name="pose_input")
    x = LayerNormalization(epsilon=1e-6)(inputs)
    
    # 🛠️ FIX QUAN TRỌNG: Chiếu lên 256 chiều để khớp key_dim=64 * num_heads=4
    x = Dense(256, activation='linear')(x)
    x = MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.3)(x, x)
    x = Dropout(0.3)(x)
    # Residual Connection an toàn
    x = Add()([x, Dense(256, activation='linear')(inputs)])
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    return inputs, x

def build_flow_branch(input_shape):
    inputs = Input(shape=input_shape, name="flow_input")
    
    x = TimeDistributed(Conv2D(16, (3,3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    
    x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x) 
    
    x = GlobalAveragePooling1D()(x) 
    x = Dense(64, activation="relu")(x)
    return inputs, x

pose_input, pose_features = build_pose_branch((SEQUENCE_LENGTH, FEATURE_SIZE))
flow_input, flow_features = build_flow_branch((SEQUENCE_LENGTH, FLOW_IMG_SIZE, FLOW_IMG_SIZE, 1))

merged = concatenate([pose_features, flow_features])
x = Dense(64, activation="relu")(merged)
x = Dropout(0.4)(x)
outputs = Dense(2, activation="softmax", name="final_output")(x)

model = Model(inputs=[pose_input, flow_input], outputs=outputs)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. HUẤN LUYỆN
print("\n🔥 Bắt đầu huấn luyện...")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1)

history = model.fit(
    [X_pose_train, X_flow_train], y_train,
    epochs=100,
    batch_size=8,
    validation_data=([X_pose_test, X_flow_test], y_test),
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weight_dict,
    verbose=1
)

# 5. ĐÁNH GIÁ & LƯU KẾT QUẢ
print("\n📊 Đang đánh giá mô hình trên tập Test...")
y_pred_probs = model.predict([X_pose_test, X_flow_test])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred, target_names=['Non-Violence', 'Violence'])
cm = confusion_matrix(y_true, y_pred)

print("\n📋 BÁO CÁO ĐÁNH GIÁ:")
print(report)

# Lưu báo cáo ra file
os.makedirs(os.path.dirname(EVAL_REPORT_PATH), exist_ok=True)
with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("=== CLASSIFICATION REPORT ===\n")
    f.write(report)
    f.write(f"\n\n=== CONFUSION MATRIX ===\n{cm}")

# Vẽ & lưu biểu đồ Loss & Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📈 Đã lưu biểu đồ huấn luyện tại: {PLOT_PATH}")

model.save(MODEL_SAVE_PATH)
print(f"\n✅ HOÀN THÀNH! Đã lưu mô hình tại: {MODEL_SAVE_PATH}")
print(f"📄 Báo cáo chi tiết: {EVAL_REPORT_PATH}")