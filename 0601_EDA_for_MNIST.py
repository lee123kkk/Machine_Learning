# EDA for MNIST

'''
MNIST 손글씨 데이터셋을 대상으로 탐색적 자료 분석(EDA, Exploratory Data Analysis)을 수행하여 모델 설계의 근거를 마련하는 코드

데이터의 분포, 특성, 왜곡, 클래스 불균형 여부, 픽셀 위치별 평균 및 분산 등을 파악하고, 
이후 모델 레이어의 개수나 필터 수, 활성화 함수 선택 등에 대한 근거로 활용할 수 있다.
'''

#[TensorFlow2 + matplotlib + seaborn 기반 EDA 코드]

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# [환경 설정] 폰트 깨짐 방지
# 다양한 운영체제에서 한글 폰트가 네모로 깨지는 것을 막아줍니다.
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False 

# 1️⃣ [데이터 로드 및 병합]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 전체적인 통계를 보기 위해 훈련용(6만)과 테스트용(1만)을 하나로 합칩니다(총 7만 개).
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# ====================================================================
# 2️⃣ [분석 1] 라벨 분포 시각화 (막대 그래프)
# 목적: 0부터 9까지의 숫자가 골고루 있는지 확인합니다. (클래스 불균형 확인)
plt.figure(figsize=(8, 4))
sns.countplot(x=y_all)
plt.title("MNIST 숫자 라벨 분포")
plt.xlabel("Digit Label")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# ====================================================================
# 3️⃣ [분석 2] 샘플 이미지 시각화
# 목적: 실제로 컴퓨터가 봐야 할 이미지가 어떻게 생겼는지 육안으로 확인합니다.
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_all[i], cmap='gray')
    plt.axis('off')
    plt.title(f'{y_all[i]}')
plt.suptitle("10 sample images")
plt.tight_layout()
plt.show()

# ====================================================================
# 4️⃣ [분석 3] 픽셀 값 분포 확인 (히스토그램)
# 목적: 픽셀 값(0~255)들이 주로 어디에 몰려 있는지 파악하여 정규화 필요성을 봅니다.
x_flat = x_all.reshape(-1, 28*28) # 1차원으로 쫙 폅니다.

print("\nTotal pixel value statistics")
print("min:", np.min(x_flat))
print("max:", np.max(x_flat))
print("mean:", np.mean(x_flat))
print("std:", np.std(x_flat))

plt.figure(figsize=(6, 4))
plt.hist(x_flat.flatten(), bins=50, color='purple')
plt.title("Total pixel value distribution (0~255)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ====================================================================
# 5️⃣ [분석 4] 전체 평균 이미지 히트맵
# 목적: 사람들이 글씨를 주로 어느 위치에 쓰는지 공간적 특징을 파악합니다.
mean_image = np.mean(x_all, axis=0) # 7만 장의 이미지를 하나로 겹쳐서 평균을 냅니다.
plt.figure(figsize=(5, 5))
sns.heatmap(mean_image, cmap='viridis')
plt.title("Heatmap of average values by pixel location")
plt.axis('off')
plt.show()

# ====================================================================
# 6️⃣ [분석 5] 클래스별 평균 이미지
# 목적: 각 숫자(0~9)들의 평균적인 형태(가장 대표적인 뼈대)를 확인합니다.
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for digit in range(10):
    # 해당 숫자만 골라내서 겹친 뒤 평균을 구합니다.
    mean_digit = np.mean(x_all[y_all == digit], axis=0)
    axes[digit].imshow(mean_digit, cmap='gray')
    axes[digit].set_title(f"Digit {digit}")
    axes[digit].axis('off')
plt.suptitle("Average image for each number")
plt.show()

#===================================================================================
# EDA (탐색적 데이터 분석)
# 내가 다룰 데이터가 어떻게 생겼는지 살펴보는 과정
# 데이터가 쏠려 있을 때 평가지표를 바꿀지를 결정한다.
# 숫자 단위가 너무 큰지 확인하고 정규화 여부를 결정한다.
# 배경 색 등을 확인하고 불필요한 연산을 줄일 방법을 고민한다.

# 막대그래프를 보면 데이터가 골구루 분포하고 있다. -> 클래스 불균형 문제는 없다.
# 픽셀 값 분포는 0(검은색)에 몰려 있고, 흰색에 조금 몰려있고 중간값은 거의 없다.
# -> 값의 편차가 매우 크므로 데이터를 반드시 정규화하여 사용해야 한다.
# 픽셀 위치별 평균 히트맵을 보면 가운데 부분만 빛나고 가장자리는 거의 0이다.
# -> 테두리 픽셀들은 학습할때 정보가 거의 없으므로 튜닝할때 가장자리를 잘라내도 무방하다.
# 클래스별 평균 이미지: 각 숫자별 평균을 낸 이미지가 대표 템플릿이 된다.

# 모델을 돌리기 전 EDA를 수행함으로써, 클래스 불균형이 없고 중앙에 정보가 집중된 
# 극단적 픽셀 분포를 가졌다는 사실을 파악해 정규화의 근거를 마련했다.