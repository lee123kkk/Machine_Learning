
# tf2-10-1-mnist_softmax.py
import tensorflow as tf

# [하이퍼파라미터 설정]
learning_rate = 0.001
batch_size = 100       # 한 번에 학습할 이미지 개수
training_epochs = 15   # 전체 데이터를 반복 학습할 횟수
nb_classes = 10        # 분류할 숫자의 개수 (0~9)

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# [데이터 전처리 1] 정규화 (Normalization)
# 0~255 픽셀 값을 0.0~1.0 사이로 변환하여 학습 안정성 확보
x_train, x_test = x_train / 255.0, x_test / 255.0

# [데이터 전처리 2] 평탄화 (Flattening)
# (60000, 28, 28) -> (60000, 784)
# 2차원 이미지를 1차원 벡터로 변환 (28 * 28 = 784)
print(x_train.shape)  
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# [데이터 전처리 3] 원-핫 인코딩 (One-hot Encoding)
# 숫자 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 형태로 변환
# Softmax를 쓰기 위해서는 정답이 이런 확률 벡터 형태여야 오차 계산이 가능함
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# [모델 구성]
tf.model = tf.keras.Sequential()
# input_dim=784: 입력 픽셀 수
# units=10: 출력 결과 (0~9까지의 확률)
# activation='softmax': 다중 분류를 위한 활성화 함수 (출력의 합 = 1)
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=784, activation='softmax'))

# [컴파일]
# loss='categorical_crossentropy': 원-핫 인코딩된 라벨과 softmax 출력 간의 오차 계산
# optimizer='Adam': SGD보다 빠르고 안정적인 최적화 도구
tf.model.compile(loss='categorical_crossentropy', 
                 optimizer=tf.optimizers.Adam(learning_rate), 
                 metrics=['accuracy'])
tf.model.summary()

# [학습 수행]
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# [예측]
predictions = tf.model.predict(x_test)
print('Prediction: \n', predictions)

# [평가]
# 학습된 데이터(train)로 정확도 평가 (보통은 test set으로 평가하는 것이 더 정확함)
score = tf.model.evaluate(x_train, y_train)
print('Accuracy: ', score[1])
#====================================================================
# MNIST 손글씨 순자 분류
# 0부터 9까지의 10개의 숫자 중 하나를 맞추는 문제

# 데이터 전처리: 정규화, 평탄화
# 원-핫 인코딩: 정답을 기계가 이해하기 쉬운 백터 형태로 변환(to_categorical()함수)
# softmax 분류기: 출력값을 모두 합하면 1.0이 되도록 확률값 변환 
# 손실 함수: softmax출력과 one hot 인코딩된 정답 사이의 오차를 계산

# 결과 분석
# 학습진행: 
# 초기 상태: 정확도83%, 손실 0.64, 학습 완료: 정확도 93.1% 손실 0.24

# 셋 이상의 클래스를 분류할 때는 정답을 원-핫 인코딩하고, 출력층에  softmax함수를 사용하면
# 각 클래스에 속할 확률을 계산하여 분류할 수 있다.