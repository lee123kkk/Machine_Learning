# lab-07-2-linear_regression_plot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. [데이터 준비]
# 정규화 함수 (필수!)
def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

xy = np.array([
    [828.659973, 833.450012, 908100, 828.349976, 831.659973],
    [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
    [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
    [816, 820.958984, 1008100, 815.48999, 819.23999],
    [819.359985, 823, 1188100, 818.469971, 818.97998],
    [819, 823, 1198100, 816, 820.450012],
    [811.700012, 815.25, 1098100, 809.780029, 813.669983],
    [809.51001, 816.659973, 1398100, 804.539978, 809.559998]
])

# 데이터 정규화 적용 (이걸 해야 예쁜 밥그릇 모양이 나옵니다)
xy = min_max_scaler(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 2. [비용 함수 정의]
# 모델을 학습시키는 게 아니라, 가중치를 강제로 바꿔가며 오차만 계산하는 함수입니다.
def calculate_cost(W_val):
    # W_val: 우리가 테스트해볼 가중치 조합
    hypothesis = tf.matmul(x_data, W_val) # 예측값 계산
    cost = tf.reduce_mean(tf.square(hypothesis - y_data)) # 오차(MSE) 계산
    return cost.numpy()

# 3. [시각화 데이터 준비]
# 가중치 w1, w2, w3를 -3에서 5 사이로 움직여 봅니다.
w_range = np.linspace(-3, 5, 50) 
w1_vals, w2_vals = np.meshgrid(w_range, w_range)

# 결과 저장용 배열 (0으로 초기화)
cost_vals_w1_w2 = np.zeros((50, 50))
cost_vals_w2_w3 = np.zeros((50, 50))

print("🎨 비용 함수 지형을 계산하고 있습니다...")

# (A) w1(시가)과 w2(고가)의 변화에 따른 오차 지형 계산
for i in range(len(w_range)):
    for j in range(len(w_range)):
        # w1, w2는 변하고 나머지는 0으로 고정
        W_temp = np.array([[w_range[i]], [w_range[j]], [0.0], [0.0]], dtype=np.float32)
        cost_vals_w1_w2[j, i] = calculate_cost(W_temp)

# (B) w2(고가)와 w3(거래량)의 변화에 따른 오차 지형 계산
for i in range(len(w_range)):
    for j in range(len(w_range)):
        # w2, w3는 변하고 나머지는 0으로 고정
        W_temp = np.array([[0.0], [w_range[i]], [w_range[j]], [0.0]], dtype=np.float32)
        cost_vals_w2_w3[j, i] = calculate_cost(W_temp)

# 4. [그래프 그리기]
fig = plt.figure(figsize=(12, 10))

# 2D 등고선 (w1 vs w2)
ax1 = fig.add_subplot(2, 2, 1)
c1 = ax1.contourf(w1_vals, w2_vals, cost_vals_w1_w2, cmap="viridis", levels=20)
fig.colorbar(c1, ax=ax1)
ax1.set_title("Cost Landscape (w1 vs w2)")
ax1.set_xlabel("Weight 1 (Open)")
ax1.set_ylabel("Weight 2 (High)")

# 2D 등고선 (w2 vs w3)
ax2 = fig.add_subplot(2, 2, 2)
c2 = ax2.contourf(w1_vals, w2_vals, cost_vals_w2_w3, cmap="jet", levels=20)
fig.colorbar(c2, ax=ax2)
ax2.set_title("Cost Landscape (w2 vs w3)")
ax2.set_xlabel("Weight 2 (High)")
ax2.set_ylabel("Weight 3 (Volume)")

# 3D 지형도 (w1 vs w2)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(w1_vals, w2_vals, cost_vals_w1_w2, cmap='viridis', edgecolor='none')
ax3.set_title("3D Surface (w1 vs w2)")
ax3.set_zlabel("Cost")

# 3D 지형도 (w2 vs w3)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(w1_vals, w2_vals, cost_vals_w2_w3, cmap='jet', edgecolor='none')
ax4.set_title("3D Surface (w2 vs w3)")
ax4.set_zlabel("Cost")

plt.tight_layout()
plt.show()
#=================================================================
# 정규화가 왜 필요한가를 보여주는 시각화 예제
# 가중치가 변할 때 오차가 어떠헥 변하는지 등고선과 3D 산으로 그려준다.
# 정규화를 하지 않았을 때는 W3쪽은 절벽처럼 가파르고 W1(가격)쪽은 평지처럼 보인다.
# 정규화를 하면 밥그릇 모양이 나온다.

# 빨간색: 오차가 큰 곳
# 파란색/남색: 오차가 작은 곳

# 3D 그래프: 오차의 지형을 보여준다. 
# 매끄러운 곡면이 나오면 공을 어디에 떨어뜨려도 중력(경사 하강법)에 의해 자연스럽게 가장 깊은 곳으로 떨어진다.

# 데이터 정규화는 울퉁불퉁하고 왜곡된 오차 지형을 매끄럽고 둥그 밥그릇(convex)모양으로 만들어 주어, 
# 인공지능이 최적의 해를 쉽고 빠르게 찾을 수 있게 해준다.
