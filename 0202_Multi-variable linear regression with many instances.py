# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

# [새로운 개념: 외부 데이터 로드]
# 'data-01-test-score.csv' 파일을 읽어옵니다. (구분자는 쉼표 ',')
# 파일 안에는 3번의 쪽지 시험 점수와 1번의 기말고사 점수가 쭉 나열되어 있을 것입니다.
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

# [새로운 개념: 배열 슬라이싱 (도마 위에서 데이터 썰기)]
# 전체 데이터(xy)에서 행(: = 처음부터 끝까지)은 다 가져오고, 
# 열은 0번째부터 마지막 열의 바로 앞(0:-1)까지만 잘라냅니다. -> 문제(X)
x_data = xy[:, 0:-1]

# 전체 데이터(xy)에서 행은 다 가져오고,
# 열은 맨 마지막 열([-1]) 하나만 쏙 빼서 가져옵니다. -> 정답(Y)
y_data = xy[:, [-1]]

# 데이터가 제대로 잘렸는지 형태(Shape)를 확인하는 디버깅 과정
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

tf.model = tf.keras.Sequential()

# [새로운 개념: 활성화 함수 내장]
# 이전에는 model.add(Activation('linear'))를 따로 썼지만,
# Dense 층의 괄호 안에 activation='linear'를 옵션으로 넣어 한 줄로 깔끔하게 처리했습니다.
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))
tf.model.summary()

# (참고: tf 2.x 최신 버전에서는 lr 대신 learning_rate 사용을 권장하지만, 옛날 표기법인 lr=1e-5도 동작합니다)
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))

# 데이터가 많아졌으므로 2000번(epochs=2000) 넉넉하게 학습시킵니다.
history = tf.model.fit(x_data, y_data, epochs=2000)

# [예측 및 활용]
# 학습된 모델에 완전히 새로운 학생의 점수 [100, 70, 101]을 넣고 결과를 봅니다.
print("Your score will be ", tf.model.predict(np.array([[100, 70, 101]], dtype=np.float32)))

# 여러 명의 학생 점수를 한꺼번에 배열로 묶어서 물어볼 수도 있습니다.
print("Other scores will be ", tf.model.predict(np.array([[60, 70, 110], [90, 100, 80]], dtype=np.float32)))

#==================================================
# 다중 선형 회귀 
# 외부 데이터 파일 불러오기 (현실의 데이터는 용량이 훨신 크기 때문에 펼도의 파일에서 읽오는 과정이 필수적)

# 파일 읽어오기: np.loadtxt()를 사용해 외부 파일의 숫자 데이터를 파이썬으로 가져옵니다.

# 데이터 슬라이싱(Slicing): 가져온 거대한 표(행렬)를 도마 위에 올려놓고 칼로 썰듯이, 
# 앞부분의 열(Columns)들은 시험 점수 데이터(x_data)로, 
# 맨 마지막 열은 최종 정답 데이터(y_data)로 깔끔하게 분리합니다.

# 코드의 간결화: 이전 예제에서 두 줄로 나누어 썼던 Dense 층과 Activation 층을 
# 한 줄로 합쳐서 코드를 더 세련되게 작성하는 방법을 보여줍니다.

# 데이터를 코드와 분리하여 파일로 관리하고 슬라이싱 기술을 적용함으로써 대용량의 실무 데이터를 처리할 수 있다.
