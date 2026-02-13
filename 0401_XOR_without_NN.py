# Lab 9 XOR
import tensorflow as tf
import numpy as np

# 1. [데이터 준비] XOR 문제
# (0,0) -> 0
# (0,1) -> 1
# (1,0) -> 1
# (1,1) -> 0
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. [모델 구성] 단층 퍼셉트론 (Single Layer Perceptron)
tf.model = tf.keras.Sequential()

# [실패 원인]
# units=1: 출력(뉴런)이 1개라는 뜻은, 판단 기준선(Decision Boundary)을 1개만 긋겠다는 의미입니다.
# 하지만 XOR 문제는 선 1개로는 절대로 나눌 수 없습니다.
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))

# 3. [컴파일] 이진 분류 문제이므로 binary_crossentropy 사용
tf.model.compile(loss='binary_crossentropy', 
                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                 metrics=['accuracy'])

tf.model.summary()

# 4. [학습 수행]
# 아무리 학습(Epoch)을 많이 시켜도 손실(Loss)이 줄어들지 않습니다.
history = tf.model.fit(x_data, y_data, epochs=1000, verbose=0) 

# 5. [결과 확인]
predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

# [정확도 확인]
score = tf.model.evaluate(x_data, y_data)
# 결과는 무조건 0.5 근처가 나옵니다. (4문제 중 2문제만 맞춤 = 찍기 수준)
print('Accuracy: ', score[1])
#=====================================================================
# 단층 퍼셉트론의 한계
# XOR 문제는 직선 하나만으로 해결할 수 없기 때문에, 단층 퍼셉트론으로는 해결할 수 없다.
# 이를 선형 분리 불가능 문제라고 한다.
# 뉴런(layer)이 하나뿐인 단순한 인공지능은 XOR과 같은 비선형 문제를 해결할 수 없으며,
# 이를 극복하기 위해서 다층 퍼셉트론(Deep learning)이 필요하다.
