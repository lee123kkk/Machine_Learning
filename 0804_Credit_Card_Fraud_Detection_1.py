# 0804_Credit_Card_Fraud_Detection

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===========================================================
# 1. 데이터 생성 및 전처리 (불균형 데이터 시뮬레이션)
# ===========================================================
# 정상 거래 99%, 사기 거래 1% 비율의 가상 데이터 1만 개 생성
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, 
                           weights=[0.99, 0.01], random_state=42)

# 학습용(80%)과 테스트용(20%) 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 신경망 학습의 안정성을 위해 데이터 스케일링(정규화) 진행
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================================================
# 2. 딥러닝 모델 구축 및 학습 (신용카드 사기 탐지용)
# ===========================================================
# 이진 분류에 적합한 심층 신경망 모델 구성
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3), # 과적합(Overfitting) 방지
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # 사기일 확률(0~1)을 출력
])

# 모델 컴파일 (이진 분류이므로 binary_crossentropy 사용)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- 딥러닝 모델 학습 시작 ---")
# 모델 학습 진행 (출력을 간결하게 하기 위해 verbose=0 설정)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
print("--- 딥러닝 모델 학습 완료 ---")

# ===========================================================
# 3. 예측 및 시각화 (제공해주신 코드 적용)
# ===========================================================
# 테스트 데이터에 대한 실제 정답
y_true = y_test

# 모델이 예측한 확률값 추출 및 1차원 배열로 변환
y_pred_probs = model.predict(X_test).flatten()

# 임계값(Threshold) 적용: 확률이 0.5보다 크면 사기(1), 아니면 정상(0)으로 판단
y_pred = (y_pred_probs > 0.5).astype(int)

# 주요 지표 계산 및 출력
print("\n--- Classification Report ---")
# 0: 정상 거래(Negative), 1: 사기 거래(Positive)
print(classification_report(y_true, y_pred, target_names=['Normal(0)', 'Fraud(1)']))

# 혼동 행렬(Confusion Matrix) 시각화
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal(0)', 'Fraud(1)'], 
            yticklabels=['Normal(0)', 'Fraud(1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Credit Card Fraud Detection - Confusion Matrix')
plt.show()