# 0805_Credit_Card_Fraud_Detection_2

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

# 1. 데이터 생성 및 전처리 (정상 99%, 사기 1%)
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, 
                           weights=[0.99, 0.01], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 딥러닝 모델 구축
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===========================================================
# ★ 핵심 수정 포인트: 클래스 가중치(Class Weights) 설정
# ===========================================================
# 사기 데이터가 정상 데이터보다 100배 적으므로, 사기 데이터(1)에 100배의 가중치를 줍니다.
class_weights = {0: 1.0, 1: 100.0}

print("--- 가중치가 부여된 모델 학습 시작 ---")
# fit 함수에 class_weight 파라미터를 추가하여 학습합니다.
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, 
          class_weight=class_weights, verbose=0)
print("--- 모델 학습 완료 ---")

# 3. 예측 및 시각화
y_true = y_test
y_pred_probs = model.predict(X_test).flatten()

# 임계값을 0.5로 유지한 상태에서 결과 확인
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Normal(0)', 'Fraud(1)']))

# 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', # 눈에 띄게 색상을 붉은색 톤으로 변경
            xticklabels=['Normal(0)', 'Fraud(1)'], 
            yticklabels=['Normal(0)', 'Fraud(1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Improved Fraud Detection - Confusion Matrix')
plt.show()
#=========================================================
# 사기 탐지 프로그램에 가중치 부여(class Weight)
# 사기 거래를 놓치는 것이 정상 거래를 틀리는 것보다 100배 더 치명적인 실수로 설정

# 실제 사기 거래 26건 중 19건을 사기로 정확히 탐지했다.
# 재현률이 이전 모델의 0.00에서 0.73(73%)로 상승했다.
# 사기를 놓치지 않기 위해서 정상 거래 1974건 중 525건을 오해했다.
# 전체 정확도는 73%로 떨어졌고 정밀도는 0.03으로 매우 낮게 나왔다.

