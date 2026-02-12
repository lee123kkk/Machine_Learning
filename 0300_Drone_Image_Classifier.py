# 드론 이미지 분류기

# 다중 분류(softmax), 학습 그래프 시각화, callback, threshold 사용
# 촬영한 물체의 4가지 특징을 바탕으로 사람인지 차량인지 나무인지 판별

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. [데이터셋 준비] 드론이 수집한 특징 데이터
# 특징(Feature) 4개: [물체 크기, 이동 속도, 붉은색(R), 초록색(G)] (0~1 정규화값)
# 정답(Label) 3개: [1,0,0]=사람, [0,1,0]=차량, [0,0,1]=나무

# (1) 사람 데이터: 크기 작음, 속도 느림
human_x = [[0.1, 0.2, 0.6, 0.1], [0.2, 0.1, 0.5, 0.2], [0.1, 0.3, 0.7, 0.1], [0.2, 0.2, 0.6, 0.2]]
human_y = [[1, 0, 0]] * 4

# (2) 차량 데이터: 크기 큼, 속도 빠름, 색상 다양(여기선 붉은 계열 가정)
car_x = [[0.8, 0.9, 0.8, 0.1], [0.9, 0.8, 0.9, 0.1], [0.7, 0.9, 0.7, 0.2], [0.8, 0.8, 0.8, 0.1]]
car_y = [[0, 1, 0]] * 4

# (3) 나무 데이터: 크기 다양, 속도 0(고정), 초록색 높음
tree_x = [[0.5, 0.0, 0.1, 0.9], [0.6, 0.0, 0.2, 0.8], [0.4, 0.0, 0.1, 0.9], [0.7, 0.0, 0.1, 0.8]]
tree_y = [[0, 0, 1]] * 4

# 데이터를 하나로 합칩니다.
x_data = np.array(human_x + car_x + tree_x, dtype=np.float32)
y_data = np.array(human_y + car_y + tree_y, dtype=np.float32)

# 클래스 이름 정의 (사람, 차량, 나무)
class_names = ['사람(Human)', '차량(Car)', '나무(Tree)']
nb_classes = 3

# 2. [모델 구성]
model = tf.keras.Sequential()
# 입력 특징 4개 -> 출력 클래스 3개 (Softmax로 확률 변환)
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=4, activation='softmax'))

# 3. [컴파일] 다중 분류이므로 'categorical_crossentropy' 사용
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=['accuracy'])

# ==========================================================
# 4. [콜백(Callback)] 200번마다 생존 신고를 하는 감시자
# ==========================================================
class DroneMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 200 == 0:
            print(f"🚁 학습 중... Epoch {epoch + 1:4d}/2000 | Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.2f}")

# ==========================================================

print("📡 드론 AI가 물체 식별 패턴을 학습합니다...")
history = model.fit(x_data, y_data, epochs=2000, verbose=0, callbacks=[DroneMonitor()])
print("✅ 학습 완료!\n")

# 5. [결과 시각화] 학습 진행 그래프 그리기
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r')
plt.title('Loss (Error)')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b')
plt.title('Accuracy')
plt.tight_layout()
plt.show()

# 6. [실전 예측 테스트] 새로운 미확인 물체 식별
print("="*50)
print("🔍 [실전 테스트] 드론 카메라에 포착된 미확인 물체 분석")
print("="*50)

# 새로운 관측 데이터 (특징: 크기, 속도, R, G)
unknown_objects = np.array([
    [0.15, 0.2, 0.6, 0.1],  # Case A: 작고 느림 (사람 추정)
    [0.85, 0.9, 0.9, 0.1],  # Case B: 크고 빠름 (차량 추정)
    [0.55, 0.0, 0.1, 0.85], # Case C: 멈춰있고 초록색 (나무 추정)
    [0.4, 0.4, 0.4, 0.4]    # Case D: 아주 애매한 데이터 (???)
], dtype=np.float32)

predictions = model.predict(unknown_objects, verbose=0)

# 결과 해석 로직 (임계값 70% 적용)
THRESHOLD = 0.7

for i, pred in enumerate(predictions):
    max_prob = np.max(pred)        # 가장 높은 확률
    max_index = np.argmax(pred)    # 그 클래스의 번호
    
    print(f"\n물체 #{i+1} 분석 결과:")
    print(f" - 예측 확률 분포: {pred}")
    
    if max_prob >= THRESHOLD:
        print(f" -> 🎯 식별 성공! [{class_names[max_index]}] 입니다. (확신: {max_prob*100:.1f}%)")
    else:
        print(f" -> ⚠️ 식별 불가! (가장 높은게 {class_names[max_index]} 같긴 한데, 확신이 {max_prob*100:.1f}% 뿐이라 위험함)")

#=====================================================================
