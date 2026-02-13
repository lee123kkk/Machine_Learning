import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비] XOR
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. [모델 구성] (성공하는 Adam 모델 사용)
input_layer = tf.keras.Input(shape=(2,))
# 은닉층 (Hidden Layer): 10개의 뉴런
hidden_layer = tf.keras.layers.Dense(units=10, activation='sigmoid')(input_layer)
# 출력층 (Output Layer): 1개의 뉴런
output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(hidden_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# 3. [컴파일] Optimizer를 Adam으로 설정 (학습 성공의 열쇠!)
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), 
              metrics=['accuracy'])

# 4. [학습 수행]
print("🧠 XOR 패턴을 학습하고 있습니다...")
history = model.fit(x_data, y_data, epochs=3000, verbose=0)
print("✅ 학습 완료!")

# ==========================================================
# 5. [시각화] 4분할 그래프 그리기
# ==========================================================
fig = plt.figure(figsize=(14, 10))

# --- (1) 좌측 상단: 시그모이드 함수 모양 ---
ax1 = fig.add_subplot(2, 2, 1)
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))
ax1.plot(z, sigmoid, 'b-')
ax1.set_title("Sigmoid Activation Function")
ax1.set_xlabel("Input (z)")
ax1.set_ylabel("Output (0~1)")
ax1.grid(True)

# --- (2) 우측 상단: 학습 Loss 그래프 ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(history.history['loss'], 'b-')
ax2.set_title("Training Loss (Binary Crossentropy)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True)

# --- (3) 좌측 하단: 최종 출력(Output)의 3D 지형 ---
# 바둑판 좌표 만들기 (입력 공간 전체를 훑어보기 위함)
xx, yy = np.meshgrid(np.arange(-0.5, 1.5, 0.1), np.arange(-0.5, 1.5, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 모델 전체를 통과한 최종 예측값
final_preds = model.predict(grid_points, verbose=0)
Z_final = final_preds.reshape(xx.shape)

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
# XOR 문제가 풀렸다면 (0,1), (1,0) 부분만 솟아오른 모양이 됩니다.
ax3.plot_surface(xx, yy, Z_final, cmap='Blues', alpha=0.8, edgecolor='none')
ax3.set_title("3D Output Surface (Final Decision)")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("Probability")

# --- (4) 우측 하단: 은닉층 첫 번째 뉴런의 3D 지형 ---
# [핵심] 모델의 중간(은닉층) 결과만 뽑아내는 부분 모델 만들기
hidden_layer_model = tf.keras.Model(inputs=model.input, outputs=hidden_layer)
hidden_preds = hidden_layer_model.predict(grid_points, verbose=0)

# 은닉층에는 10개의 뉴런이 있는데, 그 중 첫 번째(0번) 뉴런의 생각만 엿봅니다.
Z_hidden = hidden_preds[:, 0].reshape(xx.shape)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
# 단일 뉴런은 직선(평면) 하나로 세상을 나누므로 '절벽' 같은 모양이 나옵니다.
ax4.plot_surface(xx, yy, Z_hidden, cmap='viridis', alpha=0.8, edgecolor='none')
ax4.set_title("Hidden Neuron #1 Activation Surface")
ax4.set_xlabel("x1")
ax4.set_ylabel("x2")
ax4.set_zlabel("Activation")

plt.tight_layout()
plt.show()
#=====================================================================
# sigmoid 형태, loss변화, 최종 출력의 3D, 은닉층 뉴런의 3D
# sigmoid: 숫자를 0과 1 사이로 압축
# 우상단(loss): loss가 바닥으로 떨어짐. 학습이 완벽하게 되었다.
# 좌하단(최종 결과): XOR의 해답
# 우하단(은닉층 뉴런): 비스듬한 절벽 모양 (1개만 사용해서는 XOR문제 해결 불가능)

# 하나의 뉴런은 선형 분리밖에 못 만들지만, 여러개를 모아서 은닉층을 만들면 비선형 문제도 해결할 수 있다.
