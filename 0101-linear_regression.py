# 02-1-linear_regression.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 데이터 준비 (x와 y의 관계를 학습시키기 위한 데이터)
x_train = np.array([1, 2, 3, 4], dtype=np.float32)
y_train = np.array([0, -1, -2, -3], dtype=np.float32)

# [모델 구성]
# Sequential: 레고 블록을 쌓듯이 층(Layer)을 순서대로 쌓겠다고 선언
model = Sequential()

# Dense: 가장 기본적인 신경망 층 (모든 뉴런이 서로 연결됨)
# units=1: 뉴런(출력)을 1개만 사용 (답이 숫자 하나니까)
# input_dim=1: 입력 데이터가 1개 (x값 하나만 들어가니까)
model.add(Dense(units=1, input_dim=1))

# [컴파일] 모델을 학습시킬 방법 설정
# SGD: 확률적 경사 하강법 (오차를 줄이기 위해 가중치를 업데이트하는 수학적 방법)
# learning_rate=0.1: 학습률 (한 번 학습할 때 얼마나 크게 수정할지 결정)
sgd = SGD(learning_rate=0.1)

# loss='mse': 평균 제곱 오차 (정답과 예측값의 차이를 제곱해서 평균 낸 것, 작을수록 좋음)
model.compile(loss='mse', optimizer=sgd)

model.summary() # 모델의 구조를 요약해서 보여줌

# [학습]
# fit: 모의고사 풀기 (데이터를 넣어 학습 시작)
# epochs=200: 전체 데이터를 200번 반복해서 공부하라는 뜻
model.fit(x_train, y_train, epochs=200)

# [예측]
# predict: 실전 테스트 (학습한 내용을 바탕으로 새로운 값 예측)
y_predict = model.predict(np.array([5, 4]))
print(y_predict)
#==============================================================
# 딥러닝의 기초가 되는 선형 회귀를 텐서플로우로 구현하는 예제
# 데이터를 바탕으로 숫자들 사이의 관계를 알아내는 과정이다.

# 데이터를 정확히 예측하기 위해서 비용(오차)이 최소가 되는 최적의 값을 찾아낼 수 있다.
