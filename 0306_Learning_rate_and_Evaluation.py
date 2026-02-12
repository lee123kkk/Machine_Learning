# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import numpy as np

# 1. [학습용 데이터] (교과서)
# 모델이 패턴을 익히기 위해 사용하는 데이터입니다.
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# 2. [테스트용 데이터] (시험지)
# 모델이 학습할 때는 절대 보여주지 않고, 마지막 검증 때만 사용합니다.
# "학습 데이터에 없는 새로운 케이스"들입니다.
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# 리스트를 넘파이 배열로 변환 (텐서플로우가 이해할 수 있는 형태)
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# 3. [하이퍼파라미터 설정] 학습률
# 이 값을 너무 크게 하면(예: 10.0) loss가 줄어들지 않고 폭발(NaN)할 수 있습니다.
# 반대로 너무 작게 하면(예: 1e-10) 학습이 거의 진행되지 않습니다.
learning_rate = 0.1

# 4. [모델 구성]
tf.model = tf.keras.Sequential()
# 입력은 3개, 출력은 3개 (클래스 3개 분류)
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))

# 5. [컴파일]
tf.model.compile(loss='categorical_crossentropy', 
                 # 여기서 학습률(lr)을 직접 지정해 줍니다.
                 optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
                 metrics=['accuracy'])

# 6. [학습 수행]
# 오직 x_data(교과서)로만 공부합니다! x_test는 보여주지 않습니다.
tf.model.fit(x_data, y_data, epochs=1000, verbose=0) # 로그 출력 생략

# 7. [최종 평가]
print("==========================================")
# predict: x_test를 주고 예측값을 받아옵니다. (확률로 나옴)
# argmax: 확률 중 가장 높은 것의 인덱스를 뽑아 실제 정답 클래스를 확인합니다.
predicted_probs = tf.model.predict(x_test)
print("예측값(확률):\n", predicted_probs)
print("예측된 클래스:", np.argmax(predicted_probs, axis=1))

# evaluate: x_test와 y_test를 주고 채점을 합니다.
# evaluate[0]은 loss, evaluate[1]은 accuracy입니다.
score = tf.model.evaluate(x_test, y_test, verbose=0)
print("시험 점수(Accuracy): {:.2f}%".format(score[1] * 100))
print("==========================================")

#=================================================================
# 학습용 데이터와 테스트용 데이터를 분리했다.
# 인공지능이 한 번도 본 적 없는 새로운 데이터를 주고 일반화 능력을 검증한다.
# 학습률이 너무 크면 발산(overshooting)해 버리고 , 너무 작으면 학습 속도가 느려진다.

# 이미 외워버린 학습 데이터가 아니라 본 적 없는 테스트 데이터를 얼마나 잘 맞추는 지로 검증해야 하며, 학습률 조절이 학습의 성패를 좌우한다.



