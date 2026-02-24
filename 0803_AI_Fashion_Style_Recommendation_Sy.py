#0803_AI_Fashion_Style_Recommendation_System

# 사용자가 업로드한 의류 이미지를 자동 분류하고 해당 스타일과 어울리는 다른 제품을 AI가 제안하는 시스템

'''
사용 기술 요약

    전이학습
    ResNet50 기반 이미지 분류

    분류 라벨
    상의, 하의, 원피스, 신발, 액세서리

    추천 시스템
    사전 정의된 스타일 조합 룰 or 간단한 ML

    인터페이스
    Gradio or Streamlit 기반 웹 UI

    추가 기능
    실시간 이미지 업로드, 추천 결과 시각화

'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
import gradio as gr
from PIL import Image

# 환경 설정
IMG_SIZE = 64 # Step 5 제안 반영: 32x32 대신 64x64로 확대하여 특징 추출 성능 향상
BATCH_SIZE = 32
CATEGORIES = ['top', 'bottom', 'dress', 'shoes', 'accessory']
DATA_DIR = 'dummy_dataset'

# ===========================================================
# [Step 0] 테스트용 더미(Dummy) 데이터셋 생성 (에러 방지용)
# ===========================================================
print("--- [Step 0] 폴더 및 테스트 데이터 준비 ---")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(DATA_DIR, cat), exist_ok=True)
        # 각 카테고리별로 단색의 임시 이미지 10장씩 생성
        for i in range(10):
            color = tuple(np.random.randint(0, 255, 3))
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color)
            img.save(os.path.join(DATA_DIR, cat, f'dummy_{i}.jpg'))
print("더미 데이터셋 준비 완료!\n")

# ===========================================================
# [Step 1] 데이터 준비 및 전처리 (Data Augmentation 추가)
# ===========================================================
print("--- [Step 1] 데이터 로더 구축 ---")
# Step 5 제안 반영: 데이터 증강(회전, 이동, 뒤집기) 추가
datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ===========================================================
# [Step 2] 전이학습 모델 구성 (ResNet50)
# ===========================================================
print("\n--- [Step 2] ResNet50 기반 모델 구축 및 학습 ---")
base_model = ResNet50(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False # 가중치 동결

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# ResNet50에 맞는 입력 전처리 적용
x = tf.keras.applications.resnet50.preprocess_input(inputs) 
x = base_model(x, training=False)
x = Flatten()(x)
x = Dropout(0.5)(x) # 과적합 방지
outputs = Dense(len(CATEGORIES), activation='softmax')(x) 

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 에포크는 빠른 테스트를 위해 3으로 설정했습니다. 실제 데이터 사용 시 늘려주세요.
model.fit(train_data, epochs=3, validation_data=val_data)

# ===========================================================
# [Step 3] 추천 시스템 로직 구성 (룰 기반)
# ===========================================================
# 분류된 아이템과 어울리는 다른 카테고리를 추천하는 딕셔너리
style_dict = {
    'top': ['bottom', 'shoes'],
    'bottom': ['top', 'shoes'],
    'dress': ['shoes', 'accessory'],
    'shoes': ['top', 'bottom', 'dress'],
    'accessory': ['top', 'dress']
}

# ===========================================================
# [Step 4] 웹 서비스화 (Gradio 활용)
# ===========================================================
print("\n--- [Step 4] Gradio 웹 서버 실행 ---")

def predict_and_recommend(img):
    if img is None:
        return "이미지를 업로드해주세요."
    
    # 1. 이미지 리사이징 및 차원 확장
    img_resized = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img_expanded = tf.expand_dims(img_resized, axis=0)
    
    # 2. 모델 예측
    pred = model.predict(img_expanded, verbose=0)
    
    # flow_from_directory가 알파벳 순으로 클래스 인덱스를 매기므로, 매핑 정보 가져오기
    class_indices = train_data.class_indices 
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    pred_idx = np.argmax(pred)
    category = idx_to_class[pred_idx] # 예측된 폴더(카테고리) 이름
    
    # 3. 룰 기반 추천 매칭
    recommendation = style_dict.get(category, [])
    
    # 4. 결과 텍스트 포맷팅
    result_text = f"👕 분석 결과: 업로드하신 이미지는 '{category.upper()}' 카테고리로 분류되었습니다.\n\n"
    result_text += f"✨ AI 스타일 제안: 이 아이템과 어울리는 [{', '.join(recommendation).upper()}] 카테고리의 제품을 함께 매치해 보세요!"
    return result_text

# Gradio 인터페이스 구성
interface = gr.Interface(
    fn=predict_and_recommend,
    inputs=gr.Image(type="numpy", label="옷 이미지를 업로드하세요"),
    outputs=gr.Textbox(label="AI 분석 및 추천 결과"),
    title="👗 AI 패션 스타일 추천 시스템",
    description="옷 이미지를 올리면 AI가 종류를 분류하고 어울리는 스타일을 제안합니다.",

)

# 서버 실행
interface.launch(share=True) # share=True를 하면 외부 접속 가능한 퍼블릭 링크도 생성됩니다.


#=============================================================
# 사용자가 업로드한 이미지를 분류하는 AI
# 실제 데이터 대신에 더미 파일을 사용했다. 실제 옷 사진들 폴더로 교체하면 변수의 경로를 바꿔야 한다.
# class를 동적으로 매핑한다.
# 캐글등의 사이트를 통해서 실제 자료를 다운받아서 실행할 수 있다.
# 전이학습과 간단한 ML을 통해서 추천 사이트를 만들 수 있다.
