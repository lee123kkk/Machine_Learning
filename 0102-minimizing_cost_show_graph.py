import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # [NEW] 그래프를 그리기 위한 라이브러리

# 1. 학습 데이터 정의 (y = x - 1 의 관계)
x_train = np.array([1, 2, 3, 4], dtype=np.float32)
y_train = np.array([0, 1, 2, 3], dtype=np.float32)

# 2. 선형 회귀 모델 정의
# (tf.model에 직접 할당하는 방식은 권장되지는 않으나 예제 실행에는 문제없음)
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 3. 컴파일 및 학습
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
tf.model.compile(loss='mse', optimizer=sgd)
tf.model.summary()

# [NEW] history 변수: 학습 과정에서 발생한 모든 기록(Loss 변화 등)을 저장함
# verbose=0: 학습 진행 상황을 화면에 출력하지 않음 (그래프로 볼 거니까 생략)
history = tf.model.fit(x_train, y_train, epochs=200, verbose=0)

# 4. 시각화 [NEW]
# history.history['loss']: 매 Epoch마다 기록된 오차(Loss) 값들의 리스트
plt.plot(history.history['loss']) 
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show() # 그래프 창 띄우기

# 5. 학습된 weight(W)와 bias(b) 출력 [NEW]
# get_weights(): 모델이 학습을 통해 찾아낸 최적의 W와 b 값을 리스트로 반환
W, b = tf.model.layers[0].get_weights()
print(f"학습된 Weight(W): {W[0][0]:.4f}") # 이론상 1.0에 가까워야 함
print(f"학습된 Bias(b): {b[0]:.4f}")     # 이론상 -1.0에 가까워야 함

# 6. 예측 function definition [NEW]
# 텐서플로우 없이, 추출한 W와 b만으로 직접 계산해보는 함수
def predict_value(x):
    # Wx + b 공식을 직접 구현
    return float(W[0][0] * x + b[0])

# 7. 예측 function use
print("예측 결과:")
for x in [5, 4, 2.5]:
    # 우리가 직접 계산한 값 vs 텐서플로우 모델이 예측한 값 비교
    # reshape(1, 1): 스칼라 값을 (1, 1) 형태의 2차원 배열로 변환 (모델 입력 형식 맞춤)
    print(f"x = {x} -> y(수동) = {predict_value(x):.4f}")
    print(f"x = {x} -> y(모델) = {tf.model.predict(np.array(x).reshape(1, 1), verbose=0)[0][0]:.4f}")
#=============================================================
# 선형 회귀 모델의 학습 과정과 결과물을 시각적으로 확인하고 검증하는 예제
# 그래프를 통해서 학습 횟수(Epoch)가 늘어날수록 오차가 줄어드는 것을 확인할 수 있다.
# 수식을 최적화하는 과정을 시각화하고, 검증할 수 있다.
