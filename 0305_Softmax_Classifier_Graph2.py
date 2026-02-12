# Lab 6 Softmax Classifier

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 그래프를 그리기 위한 도구

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비]
# x_data: [공부 시간, 출석 수]
x_data = [[10, 5],
          [9, 5],
          [3, 2],
          [2, 4],
          [11, 1]]

# y_data: [A등급, B등급, C등급] (원-핫 인코딩)
# [1, 0, 0] -> Class 0 (A)
# [0, 1, 0] -> Class 1 (B)
# [0, 0, 1] -> Class 2 (C)
y_data = [[1, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# 클래스 개수 (3개)
nb_classes = 3

# 2. [모델 구성]
model = tf.keras.Sequential()
# 입력 2개 -> 출력 3개 (Softmax)
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=2, activation='softmax'))

# 3. [컴파일]
# 원-핫 인코딩이므로 categorical_crossentropy 사용
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# 4. [학습]
print("🏗️ 인공지능이 3D 성적 계단을 쌓고 있습니다...")
history = model.fit(x_data, y_data, epochs=2000, verbose=0)
print("✅ 학습 완료!")

# ==========================================================
# 5. [3D 시각화] 결정 경계(Decision Boundary) 그리기
# ==========================================================

# (1) 그래프의 바닥 면적(Grid) 만들기
x1_min, x1_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
x2_min, x2_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# 0.1 간격으로 촘촘하게 좌표 생성 (바둑판 만들기)
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))

# (2) 바둑판 위의 모든 점에 대해 예측하기
# 2차원 바둑판을 1줄로 쭉 펴서(ravel) 모델에 입력
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
pred_probs = model.predict(grid_points, verbose=0)

# 가장 높은 확률을 가진 클래스 번호(0, 1, 2)를 가져옴 -> 이것이 Z축(높이)이 됨
pred_labels = np.argmax(pred_probs, axis=1)

# 예측 결과를 다시 바둑판 모양(2D)으로 복구
Z = pred_labels.reshape(xx1.shape)

# (3) 3D 그래프 그리기
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # 3D 축 생성

# 결정 경계면(Surface) 그리기
# Z값(0, 1, 2)에 따라 높이가 다른 계단식 지형이 그려집니다.
# cmap='coolwarm': 파란색(0) ~ 빨간색(2)으로 색상 표현
ax.plot_surface(xx1, xx2, Z, alpha=0.3, cmap='coolwarm', edgecolor='none')

# (4) 실제 데이터 점 찍기
# 정답(y_data)을 숫자로 변환 (0, 1, 2) -> 점의 높이(z)로 사용
y_label = np.argmax(y_data, axis=1)
label_names = {0:'Class A', 1:'Class B', 2:'Class C'}
colors = ['blue', 'green', 'red'] # 0:파랑, 1:초록, 2:빨강

for i in range(nb_classes):
    # 해당 클래스인 데이터만 골라내기
    idx = (y_label == i)
    # 3D 산점도 그리기 (xs, ys, zs)
    ax.scatter(x_data[idx, 0], x_data[idx, 1], y_label[idx],
               c=colors[i], 
               s=100, 
               edgecolors='k', 
               label=label_names[i])

ax.set_xlabel('Study Hours (x1)')
ax.set_ylabel('Attendance (x2)')
ax.set_zlabel('Class (0, 1, 2)')
ax.set_title('3D Softmax Decision Boundary')
ax.legend()

plt.show()
#=============================================================================
# 3D 입체 공간에서 소프트 맥스 분류기의 결정 경계를 시각화하는 예제
# 높이를 클래스로 설정하여 입력 데이터에 따라 어떤 등급을 선택해야 하는지 계단 모형의 지형으로 보여준다.
# 소프트맥스 분류기는 입력 데이터의 특징을 바타탕으로 각 데이터가 속해야 할 최적의 클래스를 입체적으로 결정한다.

