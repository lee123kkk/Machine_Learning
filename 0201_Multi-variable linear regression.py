# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

# [데이터 준비] 2차원 배열(행렬) 사용
# x_data: 5명의 학생이 각각 치른 3번의 시험 점수 (5행 3열)
x_data = np.array([[73., 80., 75.],
                   [93., 88., 93.],
                   [89., 91., 90.],
                   [96., 98., 100.],
                   [73., 66., 70.]], dtype=np.float32)

# y_data: 5명 학생의 최종 기말고사 점수 정답 (5행 1열)
y_data = np.array([[152.],
                   [185.],
                   [180.],
                   [196.],
                   [142.]], dtype=np.float32)

tf.model = tf.keras.Sequential()

# [핵심 변경점] input_dim=3
# 입력되는 변수(x)가 3개이므로 input_dim을 3으로 설정합니다. 
# 이제 모델은 3개의 가중치(W1, W2, W3)와 1개의 편향(b)을 학습하게 됩니다.
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))

# [활성화 함수 명시]
# Activation('linear'): 입력값을 그대로 통과시키는 '선형 활성화 함수'입니다.
# 회귀(Regression) 문제에서는 기본값이므로 생략해도 똑같이 동작하지만, 
# 학습 목적상 명시적으로 적어둔 것입니다.
tf.model.add(tf.keras.layers.Activation('linear')) 

# [컴파일 및 학습률 조정]
# learning_rate=1e-5 (즉, 0.00001)
# 이전 예제들(0.1, 0.01)에 비해 보폭을 극단적으로 작게 줄였습니다.
# 이유: 입력 데이터(x_data)의 값들이 70~100 단위로 크기 때문에, 
# 보폭이 조금만 커도 오차가 폭발(발산)해버리기 때문입니다.

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))

tf.model.summary()

# 100번 반복 학습 진행
history = tf.model.fit(x_data, y_data, epochs=100)

# [새로운 데이터 예측]
# 3번의 시험에서 각각 72, 93, 90점을 맞은 새로운 학생의 최종 점수는 몇 점일지 예측합니다.
y_predict = tf.model.predict(np.array([[72., 93., 90.]], dtype=np.float32))
print(y_predict)

#==================================================================================================
# 다중 선형 회귀
# 여러 가지 원인으로 하나의 결과를 예측
# 데이터 구조가 1차원 배열에서 2차원 행렬 형태로 변경되었다.

# 학습결과 오차가 있어서 최적의 값을 정확히 찾지 못해서 학습률을 크게 바꿨더니, 발산해버렸다.
# 그 후 다시 원래대로 수정했더니, 이번에는 훨신 더 좋은 결과가 나왔다.
# 내부의 가중치가 랜덤 값이기 때문에 첫번째 값의 오차가 정답과 얼마나 가까웠는지에 따라 다른 결과가 나와 버렸다.

# 그 후 검색을 해보니 시드를 고정하는 방법이 있는데 시드를 고정하면 재현성이 좋아서 디버깅과 실험 비교에 좋고,
# 시드를 비고정하면 평균적으로 어떠한 결과가 나오는지를 알 수 있다.

# 단일 원인에서 다중 원인으로 입력 차원을 확장함으로써 여러 복합적인 변수가 작용하는 실제 현상의 문제들을 AI로 예측할 수 있다.

