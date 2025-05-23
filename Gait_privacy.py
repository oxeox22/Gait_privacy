import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 데이터 경로 설정
data_path = r'C:\Users\user\Desktop\dataset\preprocessed'

# 데이터 로드 함수
def load_data(data_path):
    data = []
    labels = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            # Accelerometer와 Gyroscope 데이터를 읽음
            acc_path = os.path.join(folder_path, 'Accelerometer.csv')
            gyro_path = os.path.join(folder_path, 'Gyroscope.csv')

            if os.path.exists(acc_path) and os.path.exists(gyro_path):
                acc_data = pd.read_csv(acc_path)[['x', 'y', 'z']].values
                gyro_data = pd.read_csv(gyro_path)[['x', 'y', 'z']].values

                # 두 센서 데이터의 길이를 맞춤
                min_length = min(len(acc_data), len(gyro_data))
                acc_data = acc_data[:min_length]
                gyro_data = gyro_data[:min_length]

                # 두 데이터를 병합
                combined_data = np.hstack((acc_data, gyro_data))

                # 슬라이딩 윈도우로 샘플 생성
                window_size = 100
                step_size = 20  # 데이터 증강을 위해 step_size 감소
                for start in range(0, len(combined_data) - window_size + 1, step_size):
                    window_data = combined_data[start:start + window_size]
                    data.append(window_data)
                    labels.append(folder.split('_')[0])  # 폴더 이름으로 라벨 추출

    return data, labels

# 데이터 준비
data, labels = load_data(data_path)

# 데이터를 배열로 변환
data = np.array(data)
labels = np.array(labels)

# 라벨 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = X_train.reshape(-1, X_train.shape[2])  # (samples, features)
X_test = X_test.reshape(-1, X_test.shape[2])
X_train = scaler.fit_transform(X_train).reshape(-1, 100, X_train.shape[1])  # (samples, time steps, features)
X_test = scaler.transform(X_test).reshape(-1, 100, X_test.shape[1])

# CNN + LSTM 모델 정의 (개선 버전)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),  # 과적합 방지
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

# 모델 평가
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix 시각화 수정
# 글씨체를 Times New Roman으로 변경하고, 라벨 이름과 순서를 수정
conf_matrix = confusion_matrix(y_test, y_pred_classes, normalize='true')

# 클래스 이름 숫자 형태로 변경 및 정렬
class_labels = [str(i) for i in range(1, len(label_encoder.classes_) + 1)]

plt.figure(figsize=(12, 10))
# Times New Roman 폰트 설정
plt.rc('font', family='Times New Roman')
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, fmt='.2f', annot_kws={"size": 14})
plt.xlabel('Predicted label', fontsize=16)
plt.ylabel('Actual label', fontsize=16)
plt.title('Confusion Matrix', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 이미지 저장 (로컬 경로 지정)
save_path = r'C:\Users\user\Desktop\confusion_matrix.png'  # 원하는 경로로 변경
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution save

# Show the plot
plt.show()

# 분류 리포트 출력
print("Classification Report - Improved CNN + LSTM")
print(classification_report(y_test, y_pred_classes, target_names=class_labels))

# 모델 요약 출력
model.summary()
