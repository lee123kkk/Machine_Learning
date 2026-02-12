# Brainwave_measurement_data_classifier

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)
np.random.seed(777)

# 1. [데이터셋 생성] 가상의 뇌파 데이터 만들기
# 입력: [알파파(휴식), 베타파(활동), 감마파(흥분/스트레스)] (0~1 사이 값)
x_data = np.random.rand(500, 3).astype(np.float32)

# 정답 생성 규칙 (가상의 생체 로직)
# - 알파파(0번)가 가장 높으면 -> [1, 0, 0] (안정)
# - 베타파(1번)가 가장 높으면 -> [0, 1, 0] (보통)
# - 감마파(2번)가 가장 높으면 -> [0, 0, 1] (스트레스)
y_data = []
for row in x_data:
    max_index = np.argmax(row) # 가장 높은 수치의 인덱스 찾기
    one_hot = [0, 0, 0]
    one_hot[max_index] = 1
    y_data.append(one_hot)

y_data = np.array(y_data, dtype=np.float32)

# 2. [데이터 분리] 훈련용(80%) vs 테스트용(20%)
# 총 500개 중 400개는 공부용, 100개는 시험용으로 나눕니다.
train_size = int(len(x_data) * 0.8)

x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

print(f"🧬 데이터 준비 완료: 훈련용 {x_train.shape}, 테스트용 {x_test.shape}")

# 클래스 정의 (3가지 상태)
class_names = ['😌 안정(Stable)', '😐 보통(Normal)', '😫 스트레스(Stress)']
nb_classes = 3

# 3. [모델 구성]
model = tf.keras.Sequential()
# 입력 3개(뇌파) -> 출력 3개(상태), 활성화 함수는 Softmax
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=3, activation='softmax'))

# 4. [컴파일]
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# ==========================================================
# 5. [모니터링] 학습 상황을 감시하는 콜백(Callback)
# ==========================================================
class StressMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:4d}/500 | Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.2f}")

# 6. [학습 수행]
print("\n🧠 AI가 뇌파 패턴을 분석하고 있습니다...")
# validation_data=(x_test, y_test)를 추가하면, 
# 학습 도중에도 틈틈이 시험을 쳐서 성적을 기록합니다. (그래프 그릴 때 유용!)
history = model.fit(x_train, y_train, 
                    epochs=500, 
                    batch_size=10,
                    validation_data=(x_test, y_test),
                    verbose=0,
                    callbacks=[StressMonitor()])
print("✅ 학습 완료!")

# 7. [결과 시각화]
plt.figure(figsize=(10, 4))

# (1) Loss 그래프 (오차)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='Train Loss')       # 공부할 때 오차
plt.plot(history.history['val_loss'], 'r--', label='Test Loss')   # 시험 볼 때 오차
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# (2) Accuracy 그래프 (정확도)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='Train Acc')     # 공부할 때 성적
plt.plot(history.history['val_accuracy'], 'k--', label='Test Acc') # 시험 볼 때 성적
plt.title('Accuracy Evolution')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show(block=False) # 그래프 띄우고 코드 계속 진행
plt.pause(2) # 2초 대기

# 8. [실전 테스트] 웨어러블 기기 시뮬레이션
print("\n" + "="*50)
print("⌚ [Smart Watch] 실시간 스트레스 측정 모드")
print("="*50)

# 가상의 실시간 측정 데이터 (알파, 베타, 감마)
real_time_data = np.array([
    [0.9, 0.1, 0.1], # 상황 1: 알파파가 아주 높음 (명상 중?)
    [0.2, 0.3, 0.8], # 상황 2: 감마파가 치솟음 (화가 난 상태?)
    [0.4, 0.5, 0.2]  # 상황 3: 베타파가 높음 (업무 중?)
], dtype=np.float32)

preds = model.predict(real_time_data, verbose=0)

for i, pred in enumerate(preds):
    status_index = np.argmax(pred)
    prob = np.max(pred) * 100
    
    print(f"⏱️ 측정 {i+1}초: {real_time_data[i]}")
    print(f"   ㄴ 분석 결과: {class_names[status_index]} (확신: {prob:.1f}%)")
    
    # 80% 이상 스트레스면 경고 알람!
    if status_index == 2 and prob >= 80:
        print("   🚨 [경고] 스트레스 수치 위험! 심호흡이 필요합니다.")
    print("-" * 30)

input("엔터를 누르면 종료합니다...")
#==========================================================
# 학습 데이터와 검증 
# np.random 코드는 완전한 무작위 데이터라 학습이 되지 않으므로, 
# 학습 효과를 그래프로 보여드리기 위해 '알파파가 높으면 안정, 
# 감마파가 높으면 스트레스'라는 가상의 규칙을 심어서 데이터를 생성
#
# 그래프의 점선이 실선을 따라간다면 AI가 실전에도 강하다

# 보이지 않는 생체 신호 데이터를 학습시키고 검증 데이터로 성능을 모니터링함으로써 
# 사용자의 감정 상태를 실시간으로 진단하고 위험상황에 개입하는 헬스케어 AI의 기본 원리를 구현했다.
