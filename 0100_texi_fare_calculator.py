#_0100_texi_fare_calculator

'''
시나리오: "택시비가 얼마 나올까?"
우리가 택시를 타면 기본요금이 있고, 거리에 따라 요금이 올라갑니다. 
이것이 정확히 일차함수(y = ax + b)의 구조입니다.
x (입력): 이동 거리 (km)
y (정답): 총 택시 요금 (1,000원 단위)
W (가중치): km당 주행 요금 (기울기)
b (편향): 택시 기본요금 (절편)

[가상의 택시 요금 규칙]
기본요금: 3,000원 (b=3)
km당 요금: 1,000원 (W=1)
예: 2km 이동 시 -> 3,000 + (1,000 × 2) = 5,000원
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re 

# 1. 학습 데이터 정의
x_train = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
y_train = np.array([4, 5, 6, 7, 8, 9], dtype=np.float32)

# 2. 모델 정의
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# 3. 컴파일
sgd = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(loss='mse', optimizer=sgd)

print("🚕 택시 요금 규칙을 학습 중입니다... (잠시만 기다려주세요)")

# [수정 1] 학습 횟수(epochs)를 500 -> 1000으로 늘림 (더 정확해짐!)
history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# 5. 학습 결과 확인
W, b = model.layers[0].get_weights()
print(f"\n[학습 완료]")
print(f" - AI가 예측한 km당 요금(Weight): {W[0][0] * 1000:.0f}원 (실제 정답: 1000원)")
print(f" - AI가 예측한 기본 요금(Bias)  : {b[0] * 1000:.0f}원 (실제 정답: 3000원)")

# 6. 사용자 입력 및 예측 시스템
print("\n" + "="*30)
print("🚖 AI 택시 요금 예측기 가동")
print("="*30)

while True:
    try:
        raw_input = input("\n이동할 거리는 몇 km인가요? (종료하려면 'q' 입력): ")
        
        if 'q' in raw_input.lower():
            print("프로그램을 종료합니다.")
            break
            
        clean_input = re.sub(r'[^0-9.]', '', raw_input)
        
        if not clean_input:
            print(f"숫자가 인식되지 않았습니다. -> 숫자만 입력해주세요.")
            continue

        distance = float(clean_input)
        
        # [수정 2] 입력값을 np.array([...])로 감싸서 넘파이 배열로 변환!
        # 이제 에러가 나지 않습니다.
        prediction = model.predict(np.array([distance]), verbose=0)
        
        fare = prediction[0][0] * 1000
        
        print(f"---------------------------------")
        print(f"거리: {distance}km")
        print(f"예상 요금: 약 {fare:.0f}원 입니다.")
        print(f"---------------------------------")
        
    except Exception as e:
        print(f"에러가 발생했습니다: {e}")
        print("다시 입력해주세요.")
        
#===========================================================
# 택시의 기본 요금과 km당 요금을 구하는 예제
# 딥러닝할때 숫자가 너무 크면 오차가 기하급수적으로 증가하므로 데이터를 작게 스케일링하고, 나중에 1000을 곱해주었다.
# 반복 횟수가 기존의 예제보다 많았는데, 더 부정확한 결과가 나왔다.
# 이는 학습률의 크기 차이가 나고, 목표 지점이 차이가 나기 때문이다.
# 학습 횟수를 1000회로 설정하고, 학습률을 0.05로 조정하니 정확한 답이 나왔다.

# 더 정확한 데이터를 얻으려면 학습횟수, 학습률, 데이터를 더 많이 주면 된다.
