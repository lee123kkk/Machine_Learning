# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비]
# x_data: [공부 시간, 출석 수]
x_data = [[10, 5],
          [9, 5],
          [3, 2],
          [2, 4],
          [11, 1]]

# y_data: [A등급, B등급, C등급] (원-핫 인코딩 됨)
# [1, 0, 0] -> A
# [0, 1, 0] -> B
# [0, 0, 1] -> C
y_data = [[1, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# 클래스 개수 (3개: A, B, C)
nb_classes = 3

# 2. [모델 구성]
model = tf.keras.Sequential()
# 입력 2개 -> 출력 3개 (Softmax)
# 편향(bias)도 사용하겠다고 명시 (use_bias=True)
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=2, activation='softmax'))

# 3. [컴파일]
# 원-핫 인코딩 데이터이므로 categorical_crossentropy 사용
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# 4. [학습]
print("🎨 인공지능이 땅따먹기 지도를 그리고 있습니다...")
history = model.fit(x_data, y_data, epochs=2000, verbose=0)
print("✅ 학습 완료!")

# ==========================================================
# 5. [시각화] 결정 경계(Decision Boundary) 그리기
# ==========================================================

# (1) 그래프의 범위 설정 (데이터보다 조금 더 넓게 잡음)
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

# (2) 바둑판(Meshgrid) 만들기: 0.1 간격으로 촘촘하게 좌표 생성
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# (3) 바둑판 위의 모든 점에 대해 예측하기
# 촘촘한 점들을 모델에 넣어서 A인지, B인지, C인지 물어봅니다.
# ravel()은 2차원 행렬을 1줄로 펴주는 함수입니다.
grid_points = np.c_[xx.ravel(), yy.ravel()]
pred_probs = model.predict(grid_points, verbose=0)

# 가장 높은 확률을 가진 클래스 번호(0, 1, 2)를 가져옵니다.
pred_labels = np.argmax(pred_probs, axis=1)

# 예측 결과를 다시 바둑판 모양으로 복구합니다.
Z = pred_labels.reshape(xx.shape)

# (4) 등고선(Contour) 그리기
plt.figure(figsize=(10, 6))

# 영역 칠하기 (배경색)
# cmap='coolwarm' 등 다양한 컬러맵 사용 가능
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)

# (5) 실제 데이터 점 찍기
class_names = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
colors = ['red', 'blue', 'green']

# 정답(y_data)을 숫자로 변환 (원-핫 -> 0, 1, 2)
y_label = np.argmax(y_data, axis=1)

for i in range(nb_classes):
    # 해당 클래스인 데이터만 골라내기
    idx = (y_label == i)
    plt.scatter(x_data[idx, 0], x_data[idx, 1], 
                c=colors[i], 
                s=100, 
                edgecolors='k', 
                label=class_names[i])

# (6) 학습된 가중치(W)와 편향(b) 수식 표시하기
# 모델 내부의 파라미터를 꺼내옵니다.
weights = model.layers[0].get_weights()
W = weights[0] # 가중치 (2x3 행렬)
b = weights[1] # 편향 (3개)

print("\n--- 학습된 수식 파라미터 ---")
for i in range(nb_classes):
    # 각 클래스별로 학습된 직선의 방정식 계수
    w1 = W[0, i]
    w2 = W[1, i]
    bias = b[i]
    print(f"[{class_names[i]}] Score = ({w1:.2f})*x1 + ({w2:.2f})*x2 + ({bias:.2f})")
    
    # 범례에 수식 추가 (그래프 구석에 표시)
    plt.plot([], [], ' ', label=f'{class_names[i]}: {w1:.2f}x1 + {w2:.2f}x2 + {bias:.2f}')

plt.xlabel('Study Hours (x1)')
plt.ylabel('Attendance (x2)')
plt.title('Softmax Decision Boundaries (3 Classes)')
plt.legend(loc='lower right')
plt.show()
    

    # # 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.contour3D(xx1, xx2, h, 50, cmap='binary')
    # for i in range(3):
    #     ax.scatter(x_data[np.argmax(y_data, axis=1) == i, 0], x_data[np.argmax(y_data, axis=1) == i, 1], np.argmax(y_data, axis=1)[np.argmax(y_data, axis=1) == i], label=f'Class: {label[i]}', s=100, edgecolors='r', alpha=0.5)
    # ax.set_xlabel('X1(hour)')
    # ax.set_ylabel('X2(attendance)')
    # ax.set_zlabel('Class')
    # ax.set_title('Decision Boundaries')
    # ax.legend()          
    
    # plt.show()
#================================================================
# 소프트맥스 분류기의 결정경계 시각화
# 소프트맥스 분류기는 여려 개의 직선 방정식을 동시에 학습하여 데이터 공간을 각 클래스가 점유하는 구역으로 깔끔하게 분할한다.
# 