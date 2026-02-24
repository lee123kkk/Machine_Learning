#0807_Aspect_Aware_Transfer

from imutils import paths
import os 
import numpy as np
import cv2
from aspectawarepreprocessor import AspectAwarePreprocessor
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 폴더명 기반 라벨(정답) 추출
imagePaths = list(paths.list_images("./images_pokemon"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)] # 중복 제거된 클래스명

# 2. 비율 유지 전처리기 객체 생성 (224x224 크기로 변환)
aap = AspectAwarePreprocessor(224, 224)

data = []
labels = []

# 3. 이미지 읽기 및 전처리 수행
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    # 이미지가 찌그러지지 않게 비율을 유지하며 224x224로 자름
    image = aap.preprocess(image) 
    
    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

# 픽셀값을 0~1 사이로 정규화 (딥러닝 학습 효율 상승)
data = data.astype("float") / 255.0

# 4. 학습용 / 테스트용 데이터 분리 (75% : 25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# ★ 중요 수정: Train 데이터로만 fit을 하고, Test 데이터는 transform만 해야 정답 라벨이 꼬이지 않습니다.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# =======================================================
# [1단계 학습] 특징 추출기(VGG16) 동결 + 새로운 분류기 학습
# =======================================================
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# VGG16 꼬리 부분에 새로운 분류용 층을 붙임
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(255, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel) # 3개 클래스 분류

model = Model(inputs=baseModel.input, outputs=headModel)

# 기존 VGG16의 지식이 망가지지 않게 모든 레이어를 잠금(Trainable = False)
for layer in baseModel.layers:
    layer.trainable = False
      
# RMSprop 옵티마이저로 빠르게 분류기만 1차 학습
opt = RMSprop(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 데이터 증강 (회전, 이동, 뒤집기 등) 적용
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

print("--- 1차 학습 시작 ---")
model.fit(aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY), epochs=1, verbose=1)

# 1차 평가
predictions = model.predict(testX, batch_size=32)
print("--- 1차 평가 결과 ---")
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# =======================================================
# [2단계 학습] 미세조정(Fine-tuning)
# =======================================================
# VGG16의 15번째 레이어부터는 잠금을 해제하여 우리 데이터에 맞게 추가 학습 허용
for layer in baseModel.layers[15:]:
    layer.trainable = True
    
# 미세조정 시에는 기존 지식이 파괴되지 않도록 SGD와 아주 작은 학습률 사용
opt = SGD(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("--- 2차 미세조정 학습 시작 ---")
model.fit(aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY), epochs=1, verbose=1)

# 2차(최종) 평가
predictions = model.predict(testX, batch_size=32)
print("--- 2차 평가 결과 ---")
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

model.save('uga.keras') # 최신 Keras 포맷으로 저장 권장
#==========================================================================
# 2단계 전이 학습 + 비율 유지 전처리
# 1차 학습: VGG16모델의 가중치를 동결하고 새로 얹은 분류기만 학습시킨다.
# 2차 학습: VGG16의 깊은 층의 동결을 풀고 작은 학습률로 조심스럽게 데이터에 맞게 깍아나간다.

# 3종류의 포켓몬 (브케인, 잠만보, 꼬부기)사진을 보여주고 분류를 하는 문제
# 1차 조절 결과에서는 잠맘봏는 다 찾아낸지만 나머지는 찾지 못했다. 
# 데이턱 적어서 일단 전부 잠만보라고 선택한것이다.

# 2차 미세 조정에서는 정확도가 1차의 0.45에서 0.64로 올랐다. 
# 2차에서는 꼬부기의 정밀도가 1.00으로 올랐다. 꼬부기의 특징을 정확히 파악했다는 뜻이다.
# 브케인은 여전히 0점이다. 테스트에 쓰인 사진이 딱 한장밖에 없었어서 학습을 하지 못했다.

# 하지만 데이터가 15장씩 밖에 없는 열악한 상황에서도 미세 조정을 거치면 성능이 향상되는 것을 확인했다.
