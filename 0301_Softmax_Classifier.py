# Lab 6 Softmax Classifier

import tensorflow as tf
import numpy as np

# 1. 데이터 준비
x_raw = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
         [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_raw = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
         [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)
nb_classes = 3

# 2. 모델 구성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# ==========================================================
# [핵심] 100번마다 생존 신고를 하는 감시자(Callback) 클래스
# ==========================================================
class MyPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d}/2000 | Loss: {logs['loss']:.6f} | Acc: {logs['accuracy']:.2f}")
# ==========================================================

print("🚀 학습을 시작합니다! (100번마다 로그가 출력됩니다)")

# 3. 모델 학습 (verbose=0으로 기본 출력은 끄고, 우리가 만든 감시자를 투입!)
history = model.fit(x_data, y_data, epochs=2000, 
                    verbose=0, 
                    callbacks=[MyPrinter()])

print("✅ 학습 완료!\n")

# 4. 전체 예측 테스트 (원래 코드에 있던 모든 테스트 부활)
print('-------------- [Test A] --------------')
a = model.predict(np.array([[1, 11, 7, 9]], dtype=np.float32))
print(f"예측 확률: {a}")
print(f"선택된 클래스: {np.argmax(a, axis=1)}")

print('-------------- [Test B] --------------')
b = model.predict(np.array([[1, 3, 4, 3]], dtype=np.float32))
print(f"예측 확률: {b}")
print(f"선택된 클래스: {np.argmax(b, axis=1)}")

print('-------------- [Test C] --------------')
c = model.predict(np.array([[1, 1, 0, 1]], dtype=np.float32))
c_onehot = np.argmax(c, axis=-1)
print(f"예측 확률: {c}")
print(f"선택된 클래스: {c_onehot}")

print('-------------- [Test All] --------------')
all_data = np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]], dtype=np.float32)
all_predict = model.predict(all_data)
all_onehot = np.argmax(all_predict, axis=1)
print(f"전체 예측 확률:\n{all_predict}")
print(f"전체 선택 결과: {all_onehot}")

#============================================================
# 3개 이상의 선택지 중 하나를 고르는 다중 클래스 분류 예제

# 원-핫 인코딩:정답이 숫자 하나가 아니라 [0,0,1]처럼 되어 있다. 이 경우에 3번째로 분류한다.
# 소프트맥스: 모델의 출력값을 모두 더하면 1.0이 되도록 만들어준다.
# 아그맥스: 확률이 가장 높은 곳의 위치를 찾아낸다.

# softmax함수를 통해 출력값을 확률 분포로 변환하고 원-핫 인코딩을 사용하여 
# 세 가지 이상의 선택지가 있는 복잡한 분류 문제를 해결할 수 있다.
