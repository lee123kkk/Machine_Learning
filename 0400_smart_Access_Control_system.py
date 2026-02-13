# File Name: Smart_Access_Control_Complete.py
import tensorflow as tf
from tensorflow import keras  # Pylint 에러 방지용 임포트
import numpy as np
import datetime
import os

# 결과 재현을 위한 시드 고정
tf.random.set_seed(777)

# 1. [데이터 준비]
# 입력: [생체인식, 신분증, 관리자승인]
x_data = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
# 출력: [출입 허용 여부]
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

# 2. [모델 구성]
model = keras.Sequential()
model.add(keras.layers.Dense(units=8, input_dim=3, activation='sigmoid', name='Hidden_Layer'))
model.add(keras.layers.Dense(units=1, activation='sigmoid', name='Output_Layer'))

model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(learning_rate=0.1), 
              metrics=['accuracy'])

# =================================================================
# [⭐ 핵심 수정] 프로파일러 끄고 순수 그래프만 저장하기
# =================================================================
# (1) 로그 경로 설정 (logs_final 폴더 사용)
log_dir = os.path.join(".", "logs_final", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(log_dir)

# (2) 추적 함수 정의
@tf.function
def my_model_tracer(x):
    return model(x)

# (3) 추적 시작 (profiler=False로 설정하여 에러 원천 차단!)
tf.summary.trace_on(graph=True, profiler=False)

# (4) 데이터 흘려보내기
my_model_tracer(tf.constant(x_data))

# (5) 그래프 저장 (profiler_outdir 제거!)
with file_writer.as_default():
    tf.summary.trace_export(
        name="my_model_graph",
        step=0,
        profiler_outdir=None 
    )
print("✅ 그래프 저장 성공! (logs_final 폴더)")
# =================================================================

# 3. [학습 수행]
# 자동 그래프 저장 끄기 (write_graph=False)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1,
    write_graph=False, 
    write_images=True
)

print("\n🔒 학습 시작...")
model.fit(x_data, y_data, epochs=1000, verbose=0, callbacks=[tensorboard_callback])
print("✅ 학습 완료!")

# =================================================================
# 4. [결과 확인] 터미널 출력 부분 (다시 추가됨!)
# =================================================================
print("\n" + "="*60)
print("🏢 [보안 시스템] 출입 시도 로그 분석")
print("="*60)

predictions = model.predict(x_data)
score = model.evaluate(x_data, y_data, verbose=0)

# 결과 이쁘게 출력하기 위한 시나리오 리스트
scenarios = [
    "빈손 방문객", 
    "승인만 받은 외부인", 
    "등록 안 된 직원(지문만)", 
    "완벽한 직원(풀세트)", 
    "임시 출입증+승인", 
    "신분증+지문(승인X)"
]

for i, pred in enumerate(predictions):
    prob = pred[0]
    # 확률이 0.5보다 크면 문 열림(Green), 작으면 차단(Red)
    result = "🟢 문 열림" if prob > 0.5 else "🔴 차단됨"
    print(f"상황: {scenarios[i]:<18} {x_data[i]} -> AI 판단: {prob:.4f} ({result})")

print("-" * 60)
print(f"최종 정확도(Accuracy): {score[1]*100:.2f}%")

#=============================================================
# 스마트 출입 통제 시스템
# 복합 논리(XOR, OR, AND)를 해결하는 보안 AI 구축
# 생체인식, 신분증, 관리자 승인의 3가지 조건이 복잡하기 얽힌 상태

# Main Graph: 끊어진 곳 없이 입력부터 출력까지 화살표가 매끄럽게 연결됨
# 정확도: 1.0도달
# Loss: 0.7에서 시작해 0으로 수렴
# Histograms(뉴런의 역할 분담): 초기에는 중앙(0)에 뭉쳐 있으나, 학습이 진행될 수록 양쪽 끝으로 넓게 퍼짐

# 텐서플로우 keras의 자동 그래프 저장 기능이 윈도우 환경에서 충돌을 일으켜 파일이 깨져서 수동 추적으로 바꿨다.
# 프로파일러 에러가 발생해서 프로파일러를 비활성화했다.

# 복잡한 규칙이 섞인 문제라도 충분한 뉴런과 좋은 학습도구를 사용하면 완벽한 ai를 만들 수 있다.
