import datetime
import numpy as np
import os
import tensorflow as tf

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비] XOR 문제
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. [모델 구성]
model = tf.keras.Sequential()

# [수정 1] 은닉층 뉴런 개수 증가 (units=2 -> units=10)
# 뉴런이 2개일 때는 '겨우겨우' 풀거나 실패하지만, 10개면 아주 여유롭게 풀어냅니다.
# 활성화 함수는 시그모이드(sigmoid)를 그대로 사용했습니다.
model.add(tf.keras.layers.Dense(units=10, input_dim=2, activation='sigmoid'))

# 출력층 (결과 1개)
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# 3. [컴파일]
# [수정 2] 최적화 도구 변경 (SGD -> Adam)
# SGD는 단순해서 XOR 문제에서 헤매는 경우가 많습니다. 
# Adam은 방향과 보폭을 자동으로 조절해주어 훨씬 강력합니다.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), 
              metrics=['accuracy'])

model.summary()

# 4. [TensorBoard 설정] (로그 경로가 겹치지 않게 시간 사용)
log_dir = os.path.join(".", "logs", "xor_fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 5. [학습 수행]
print("🧠 XOR 학습 시작 (Adam Optimizer)...")
history = model.fit(x_data, y_data, epochs=1000, 
                    verbose=0, # 로그 너무 길게 나오는 것 방지
                    callbacks=[tensorboard_callback])
print("✅ 학습 완료!")

# 6. [예측 및 평가]
predictions = model.predict(x_data)
print('Prediction: \n', predictions)

score = model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])
#===================================================================
# TensorBoard 활용
# 딥러닝 모델이 학습하는 과정이 오래걸릴때, 
# 학습이 진행되는 동안 모델의 상태를 실시간으로 감시하는 대시보드를 구축하기 위해서 
# 텐서보드를 사용한다.

# 콘솔창을 통해서 정확도가 1.0이 나온것을 확인할 수 있다.
# 텐서보드의 epoch_accuracy에는 accuracy 0.75로 보인느데 학습이 덜 된 상태의 모습이다.
# 히스토그램 그래프는 산들이 점점 퍼져가는 것을 볼 수 있는데, 뉴런들이 자시느이 위치를 찾아가고 있는 것이다.
# epoch_learning_rate는 공부하는 속도로 여기에서는 0.1로 고정되어 있어서 일직선으로 표현된 것을 확인할 수 있다.
# epoch_loss는 틀린 개수로 그래프가 미끄럼틀처럼 우하향하고 있는 것을 확인할 수 있다.
# kernal/histogram은 가중치들의 분포 변화이다. 

# TensorBoard 우측 패널에서 다양한 설정들을 건드릴 수 있다.
# Horizontal Axis는 가로축 설정으로 step은 학습 횟수 기준, relative는 학습을 시작한 시간 기준, Wall은 실제 현재 시간 기준이다.
# Smoothing 기능은 그래프의 선을 부르럽게 펴주는 기능이다. 0이면 날것 그대로의 데이터를 보여주고 1에 가까워지면 자잘한 진동을 무시하고 부드럽게 보여준다.
# Ignore outliers in chart scaling: 이상치들을 무시하고 그래프를 예쁘게 그린다.

# 눈에 보이지 않는 학습과정을 TensorBoard로 시각화하면, 모델의 상태를 실시간으로 확인할 수 있다.
