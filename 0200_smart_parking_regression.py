
# 스마트 주차장 수요 예측 시스템

import tensorflow as tf
import numpy as np

# 1. 파일에서 데이터 불러오기 (경로 주의: 같은 폴더에 있어야 함)
print("📂 주차장 데이터를 불러오는 중...")
xy = np.loadtxt('parking_data.csv', delimiter=',', dtype=np.float32)

# 2. [도마 위에서 데이터 썰기: Slicing]
# x_data (원인): 모든 행(:)을 가져오되, 열은 0번째부터 마지막 열의 '직전'까지 자름
x_data = xy[:, 0:-1] 

# y_data (결과): 모든 행(:)을 가져오되, 열은 딱 마지막 열([-1]) 하나만 가져옴
y_data = xy[:, [-1]]

print(f" - 원인 데이터(X) 형태: {x_data.shape} (시간, 유동인구, 강수량)")
print(f" - 결과 데이터(Y) 형태: {y_data.shape} (주차량)\n")

# 3. 모델 구성 (정규화 포함)
model = tf.keras.Sequential()

# 변수가 3개(시간, 유동인구, 비)이므로 input_shape=[3,]
normalizer = tf.keras.layers.Normalization(input_shape=[3,])
normalizer.adapt(x_data)
model.add(normalizer)

# 최종 차량 수를 예측하는 층
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# 4. 컴파일 및 학습
# 정규화를 했기 때문에 학습률을 0.1로 시원하게 올려도 아주 잘 학습됩니다!
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

print("🚙 인공지능이 주차장 점유율 패턴을 학습하고 있습니다...")
# 데이터가 준비되었으니 500번 넉넉하게 학습 (verbose=0으로 로그 생략)
model.fit(x_data, y_data, epochs=500, verbose=0)
print("✅ 학습 완료!\n")

# 5. 새로운 상황 예측해 보기
print("="*45)
print("🔍 내일의 스마트 주차장 수요 예측")
print("="*45)

# 상황 A: 맑은 날(비 0mm) 오후 6시(18시), 퇴근길 유동인구 폭발(6500명 -> 65)
scenario_A = np.array([[18., 65., 0.]], dtype=np.float32)
pred_A = model.predict(scenario_A, verbose=0)
print(f"상황 A (18시, 맑음, 사람 많음) -> 예상 주차량: 약 {pred_A[0][0]:.0f}대")

# 상황 B: 비가 엄청 오는(30mm) 밤 10시(22시), 유동인구 적음(1000명 -> 10)
scenario_B = np.array([[22., 10., 30.]], dtype=np.float32)
pred_B = model.predict(scenario_B, verbose=0)
print(f"상황 B (22시, 폭우, 사람 적음) -> 예상 주차량: 약 {pred_B[0][0]:.0f}대")

#==============================================
#
#
# 수집된 대규모 외부 파일 데이터를 분리하고 모델에 주입할 수 있다.
