# tf2-07-3-linear_regression_min_max.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 결과의 일관성을 위해 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비]
# 정규화 함수 (Min-Max Scaler): 데이터를 0~1 사이로 압축
def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누는 에러 방지를 위한 노이즈(1e-7) 추가
    return numerator / (denominator + 1e-7)

# 원본 데이터 (시가, 고가, 거래량, 저가, 종가)
xy = np.array([
    [828.659973, 833.450012, 908100, 828.349976, 831.659973],
    [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
    [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
    [816, 820.958984, 1008100, 815.48999, 819.23999],
    [819.359985, 823, 1188100, 818.469971, 818.97998],
    [819, 823, 1198100, 816, 820.450012],
    [811.700012, 815.25, 1098100, 809.780029, 813.669983],
    [809.51001, 816.659973, 1398100, 804.539978, 809.559998]
])

# [매우 중요] 데이터를 0~1로 정규화합니다.
# 이걸 안 하면 거래량(100만 단위) 때문에 학습이 터져버립니다.
xy = min_max_scaler(xy)

x_data = xy[:, 0:-1] # 입력: 시가, 고가, 거래량, 저가
y_data = xy[:, [-1]] # 정답: 종가 (Close Price)

# 2. [모델 구성]
tf.model = tf.keras.Sequential()
# 입력 4개 -> 출력 1개 (값 예측)
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=4))
# 숫자 그대로 출력하므로 선형(Linear) 활성화 함수 사용
tf.model.add(tf.keras.layers.Activation('linear'))

# 3. [컴파일]
# 정규화된 데이터이므로 학습률을 0.1로 올려서 빠르게 학습시킵니다.
# (원본 코드의 1e-5는 정규화 안 된 데이터용이라 너무 느립니다)
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
tf.model.summary()

# 4. [학습 수행]
print("📉 주식 가격 패턴 학습 중...")
history = tf.model.fit(x_data, y_data, epochs=1000, verbose=0)

# 5. [결과 예측 및 평가]
predictions = tf.model.predict(x_data)
score = tf.model.evaluate(x_data, y_data, verbose=0)

print(f"Prediction (0~1 Scale):\n {predictions}")
print(f"Final Cost (MSE): {score}")

# ==========================================================
# 6. [시각화] 결과 그래프 그리기
# ==========================================================
plt.figure(figsize=(12, 5))

# (1) 학습 오차(Loss) 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r-')
plt.title('Model Loss (Error)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)

# (2) 예측 결과 비교 (실제값 vs 예측값)
# 파란선(실제)과 빨간 점선(예측)이 겹칠수록 잘 맞춘 것입니다.
plt.subplot(1, 2, 2)
plt.plot(y_data, 'b-', label='True Price')      # 실제 종가
plt.plot(predictions, 'r--', label='Prediction') # AI 예측값
plt.title('True Price vs Prediction (Normalized)')
plt.xlabel('Day')
plt.ylabel('Price (Scaled 0~1)')
plt.legend()
plt.grid(True)

plt.show()
#=========================================================
# 정규화를 하고 데이터를 얼마나 잘 나타냈는지 분석하는 예제
# 입력 데이터의 단위가 서로 다를 때 정규화를 거쳐야 정확하게 학습할 수 있다.
#
#

