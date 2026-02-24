# 0808_Webcam_Animal_Detector
import cv2
import numpy as np
import tensorflow as tf
# 미리 학습된 초경량 모델 MobileNetV2와 전처리 함수들을 가져옵니다.
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

print("--- 🌍 인터넷에서 미리 학습된 MobileNetV2 모델을 불러옵니다... ---")
# weights='imagenet': 1000가지 사물/동물이 학습된 가중치 사용
model = MobileNetV2(weights='imagenet')
print("--- 모델 로딩 완료! ---")

# =========================================================
# 웹캠 연결 (외장 웹캠을 위해 인덱스 1, 2, 0 순으로 시도)
# =========================================================
cap = None
for i in [1, 2, 0]: # 외장(1,2)부터 찾고 없으면 내장(0)을 찾습니다.
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"--- 📷 카메라 인덱스 {i}번(웹캠) 연결 성공! ---")
        break

if not cap or not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다. WSL 환경이라면 윈도우 기본 CMD/PowerShell에서 실행해주세요.")
    exit()

print("--- 실시간 탐지를 시작합니다. 종료하려면 창을 클릭하고 'q'를 누르세요. ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) MobileNetV2 모델은 224x224 크기의 이미지를 입력으로 받습니다.
    img_resized = cv2.resize(frame, (224, 224))
    
    # 2) 차원 확장: (224, 224, 3) -> (1, 224, 224, 3)
    x = np.expand_dims(img_resized, axis=0)
    
    # 3) MobileNetV2 전용 전처리 (픽셀값을 모델이 좋아하는 형태로 자동 변환)
    x = preprocess_input(x)

    # 4) 실시간 예측
    preds = model.predict(x, verbose=0)
    
    # 5) decode_predictions: 1000개 클래스 중 가장 확률이 높은 1개(top=1)의 이름과 확률을 사람이 읽기 쉽게 변환해줌
    results = decode_predictions(preds, top=1)[0][0]
    
    # results 구조: (코드명, 사람이름, 확률) -> ex: ('n02123045', 'tabby', 0.85)
    label = results[1]
    confidence = results[2] * 100

    # 6) 화면에 결과 텍스트 오버레이 (영어 동물 이름이 출력됩니다)
    text = f"{label.upper()}: {confidence:.1f}%"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('AI Animal & Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- 프로그램이 종료되었습니다. ---")