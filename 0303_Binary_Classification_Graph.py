# Binary_Classification_Graph

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 결과의 일관성을 위해 시드 고정
tf.random.set_seed(777)

# 1. 데이터 설정
# x_data: [공부 시간, 출석 일수]
x_train = np.array([[10, 5],
                    [9, 5],
                    [3, 2],
                    [2, 4],
                    [11, 1]], dtype=np.float32)

# y_data: [1:합격, 0:불합격]
y_train = np.array([[1], [1], [0], [0], [0]], dtype=np.float32)

# 2. 모델 구성
model = tf.keras.Sequential()
# 입력 변수가 2개(공부, 출석), 출력은 1개(합격여부)
# 이진 분류이므로 활성화 함수는 시그모이드(sigmoid)
model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))

# 3. 컴파일
# 이진 분류이므로 binary_crossentropy 사용
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# 4. 학습
print("📉 학습을 시작합니다...")
history = model.fit(x_train, y_train, epochs=2000, verbose=0)
print("✅ 학습 완료!")

# ==========================================================
# 5. [시각화 준비] 학습된 가중치(W)와 편향(b) 꺼내기
# ==========================================================
# model.get_weights()를 하면 [W행렬, b배열] 리스트를 줍니다.
weights = model.get_weights()
w1 = weights[0][0][0] # 공부 시간(x1)에 대한 가중치
w2 = weights[0][1][0] # 출석 일수(x2)에 대한 가중치
b  = weights[1][0]    # 편향(bias)

print(f"\n학습된 파라미터 -> W1: {w1:.4f}, W2: {w2:.4f}, Bias: {b:.4f}")

# 그래프를 그릴 바둑판(Grid) 좌표 만들기
# x축(공부시간): 0~15, y축(출석): 0~10 범위를 잘게 쪼갭니다.
x1_vals = np.linspace(0, 15, 50)
x2_vals = np.linspace(0, 10, 50)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# 결정 경계 및 가설(예측값) 계산 공식: Sigmoid(W1*x1 + W2*x2 + b)
# 우리가 학습시킨 모델의 수식을 그대로 옮겨온 것입니다.
Z = w1 * X1 + w2 * X2 + b
Hypothesis = 1 / (1 + np.exp(-Z)) # 시그모이드 공식 적용

# ==========================================================
# 6. [2D 그래프] 위에서 내려다본 결정 경계 (Decision Boundary)
# ==========================================================
plt.figure(figsize=(10, 5))

# 등고선 그리기 (높이가 0~0.5인 구간과 0.5~1인 구간을 색칠)
plt.contourf(X1, X2, Hypothesis, levels=[0, 0.5, 1], colors=["lightblue", "lightcoral"], alpha=0.5)

# 실제 데이터 점 찍기 (파란색: 불합격(0), 빨간색: 합격(1))
# c=y_train.flatten()은 0과 1에 따라 색을 다르게 칠하라는 뜻입니다.
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.flatten(), cmap="bwr", edgecolors="k", s=100)

# 결정 경계선 그리기 (정확히 확률이 0.5가 되는 지점에 빨간 선 긋기)
plt.contour(X1, X2, Hypothesis, levels=[0.5], colors='red', linewidths=3)

plt.xlabel('Study Time (x1)')
plt.ylabel('Attendance (x2)')
plt.title('2D Decision Boundary (Red Line = 50% Probability)')
plt.show()

# ==========================================================
# 7. [3D 그래프] 입체적으로 본 시그모이드 곡면
# ==========================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D 곡면(Surface) 그리기
ax.plot_surface(X1, X2, Hypothesis, cmap="viridis", alpha=0.6)

# 실제 데이터 점 찍기
ax.scatter(x_train[:, 0], x_train[:, 1], y_train.flatten(), c='r', edgecolors="k", s=50, label='Data Points')

# 결정 경계선 그리기 (높이 0.5 지점에 빨간 선)
ax.contour(X1, X2, Hypothesis, levels=[0.5], colors='red', linestyles="solid", linewidths=3, offset=0.5)

ax.set_xlabel('Study Time')
ax.set_ylabel('Attendance')
ax.set_zlabel('Probability (Pass=1)')
ax.set_title('3D Sigmoid Surface')
plt.legend()
plt.show()
#===============================================================
# 결정 경계 시각화 

# 로지틱스 회귀 모델을 시각적으로 보여준다.
# 2D 그래프: 
#   빨간 실선: 결정 경계
#   영역 색깔: 붉은 영역:합격(1), 푸른 영역:불합격(0)
# 3D 그래프:
#   데이터가 바닥과 천장에 붙어 있음

# 로지틱스 회귀는 데이터 공간을 칼로 자르듯 나누는 결정 경계를 찾아내는 기하학적 과정이다.
