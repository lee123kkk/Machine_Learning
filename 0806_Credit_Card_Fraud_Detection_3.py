# 0806_Credit_Card_Fraud_Detection_3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE  # ★ 추가된 SMOTE 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===========================================================
# 1. 데이터 준비 및 전처리
# ===========================================================
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, 
                           weights=[0.99, 0.01], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================================================
# ★ 2. SMOTE를 이용한 데이터 오버샘플링
# ===========================================================
print("--- SMOTE 적용 전 학습 데이터 분포 ---")
print(f"정상(0): {sum(y_train==0)}개, 사기(1): {sum(y_train==1)}개")

# 사기 데이터(1)를 가상으로 생성하여 정상 데이터(0)의 수와 1:1로 맞춤
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- SMOTE 적용 후 학습 데이터 분포 ---")
print(f"정상(0): {sum(y_train_smote==0)}개, 사기(1): {sum(y_train_smote==1)}개\n")

# ===========================================================
# 3. 딥러닝 모델 구축 및 학습
# ===========================================================
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_smote.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- 딥러닝 모델 학습 시작 (SMOTE 데이터 사용) ---")
# 클래스 가중치(class_weight)는 SMOTE로 이미 비율을 맞췄으므로 제거합니다.
model.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32, 
          validation_data=(X_test, y_test), verbose=0)
print("--- 모델 학습 완료 ---")

# ===========================================================
# ★ 4. 임계값(Threshold) 조정 및 예측
# ===========================================================
y_pred_probs = model.predict(X_test).flatten()

# 기존 0.5 대신, 더 엄격한 기준인 0.85로 임계값을 끌어올립니다.
# "확률이 85% 이상일 때만 확실한 사기로 간주하겠다!"는 의미입니다.
CUSTOM_THRESHOLD = 0.85  
y_pred = (y_pred_probs > CUSTOM_THRESHOLD).astype(int)

# ===========================================================
# 5. 결과 평가 및 시각화
# ===========================================================
print(f"\n--- Classification Report (Threshold: {CUSTOM_THRESHOLD}) ---")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Fraud(1)']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal(0)', 'Fraud(1)'], 
            yticklabels=['Normal(0)', 'Fraud(1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'SMOTE + Threshold({CUSTOM_THRESHOLD}) Matrix')
plt.show()
#=====================================================================
# 임계값 최적화 + 데이터 샘프링 기법 적용(SMOTE)
# 전 모델은 사기일 확률이 50%만 넘어도 무조건 사기라고 판단해했는데 PB 곡선을 그려서 완벽한 비율의 임계값을 탐지하도록 수정
# 오버샘플링을 통해서 기존 사기 데이터들의 특징을 섞어서 가짜 사기 데이터를 만들어 정상 데이터와 비율을 비슷하게 맞춘다.

# 가짜 사기 데이터를 7900개 생성해서 정상 데이터와 사기 데이터를 1:1의 비율로 학습한다.
# 임계값을 0.85로 올려서 정상을 사기라고 판단한 비율을 낮췄다.

# 실제 사기 거래 26건 중 7건만 잡아내서 사기에 대한 재현율은 0.27%로 폭락했다.
# 525건이었던 오경보를 36건까지 낮워서 정확도를 97%, 정상에 대한 재현율을 0.98까지 올렸다.

# 이번에는 다시 오경보를 줄이려다 사기를 놓치는 과소탐지 상태에 빠져 버렸다.