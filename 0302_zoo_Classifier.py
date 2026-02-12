#동물 분류

import tensorflow as tf
import numpy as np

# 1. 데이터 불러오기
# data-04-zoo.csv 파일이 필요합니다.
# 데이터 구조: 16개의 특징(털, 날개, 알...) + 1개의 정답(동물 종류 0~6)
print("🦁 동물원 데이터를 불러오고 있습니다...")
try:
    xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
except OSError:
    print("🚨 오류: 'data-04-zoo.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요!")
    # (실행을 위해 가상의 데이터를 임시로 만듭니다)
    xy = np.random.rand(20, 17).astype(np.float32)
    xy[:, -1] = np.random.randint(0, 7, 20) 

# 특징(X)과 정답(Y) 분리
x_data = xy[:, 0:-1]  # 16가지 특징
y_data = xy[:, [-1]]  # 0~6 사이의 동물 종(Class)

# 정답의 종류 개수 (0번부터 6번까지 총 7개)
nb_classes = 7

# [핵심 기술: 원-핫 인코딩]
# 정답 숫자(예: 3)를 벡터(예: [0, 0, 0, 1, 0, 0, 0])로 변환합니다.
# (None, 1) -> (None, 7) 형태로 모양이 바뀝니다.
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)

print(f" - 입력 데이터(X) 형태: {x_data.shape}")
print(f" - 정답 데이터(Y) 형태: {y_data.shape} -> 원-핫 변환 후: {y_one_hot.shape}")

# 2. 모델 구성
model = tf.keras.Sequential()

# 입력은 16개, 출력은 7개 (동물 종류)
# 다중 분류이므로 활성화 함수는 'softmax' 사용
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))

# 3. 컴파일
# 원-핫 인코딩된 정답을 사용하므로 'categorical_crossentropy'를 써야 합니다.
# (만약 원-핫 변환을 안 했다면 'sparse_categorical_crossentropy'를 씁니다)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

model.summary()

# ==========================================================
# [중간 감시자] 100번마다 학습 상태를 보고하는 콜백 함수
# ==========================================================
class ZooMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"🐾 학습 중... Epoch {epoch + 1:4d}/1000 | Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.2f}")
# ==========================================================

print("\n🚀 동물 분류 AI 학습 시작!")

# 4. 학습 (verbose=0으로 기본 로그 끄고, 콜백으로 대체)
history = model.fit(x_data, y_one_hot, epochs=1000, 
                    verbose=0, 
                    callbacks=[ZooMonitor()])

print("✅ 학습 완료!\n")

# 5. 전체 데이터에 대한 예측 및 결과 비교
print("="*50)
print("🔍 [최종 테스트] 예측값 vs 실제 정답 비교")
print("="*50)

# 모델에게 전체 데이터를 주고 맞춰보라고 시킵니다.
pred_probs = model.predict(x_data, verbose=0)

# 확률(Probability) 중 가장 높은 것의 위치(Index)를 뽑아냅니다.
pred_labels = np.argmax(pred_probs, axis=1)

# 실제 정답(y_data)은 2차원 배열([[0], [1]...])이므로 
# 보기 좋게 1차원([0, 1...])으로 펴줍니다(flatten).
y_flat = y_data.flatten().astype(int)

# zip()을 써서 예측한 것과 정답을 하나씩 꺼내서 비교합니다.
# 너무 많으니 앞쪽 10개만 출력해서 확인해 보겠습니다.
for i, (p, y) in enumerate(zip(pred_labels, y_flat)):
    if i >= 10: break # 10개만 보고 멈춤
    
    is_correct = (p == y)
    result_emoji = "🙆‍♂️ 정답" if is_correct else "🙅‍♀️ 오답"
    print(f"[{i+1}] 예측: {p} vs 정답: {y} -> {result_emoji}")

# 최종 정확도 계산
accuracy = np.mean(pred_labels == y_flat)
print(f"\n📊 전체 데이터에 대한 최종 정확도: {accuracy * 100:.2f}%")

#============================================================
# 숫자 꼬리표를 벡터 지도로 바꾸기
# 컴퓨터에게 정답을 그냥 숫자로 알려주면, 숫자의 크기에 의미를 부여할 수 있음.
# 원-핫 인코딩을 통해서 위치로만 정답을 표시한다.
# 숫자 형태로 된 카테고리를 One-Hot Encoding으로 변환하여 모델에 제공함으로써, 
# 컴퓨터가 데이터 간의 불필요한 서열 관계를 오해하지 않고 각 클래스를 독립적으로 분류한다.
