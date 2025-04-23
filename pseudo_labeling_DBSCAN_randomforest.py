from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
클러스터링 실험 코드의 정확도는 높았지만, 
시각화에서 분류 성능이 낮아 보였기 때문에, 이를 개선한 DBSCAN 버전 코드를 작성하였습니다.
"""

# ---------------------------------------------------------
# 1. 소융 데이터 생성 (10,000명, 3개의 특성, 3개 클래스)
# ---------------------------------------------------------
X_soyung, y_soyung = make_classification(n_samples=10000, n_features=3, 
                                         n_informative=3, n_redundant=0,
                                         n_classes=3, random_state=42)

df_soyung = pd.DataFrame(X_soyung, columns=["연간_구매", "구매_빈도", "반품률"])
df_soyung["등급"] = y_soyung

# ---------------------------------------------------------
# 2. 사근 데이터 준비 (소융 데이터의 마지막 2,000명 일부 복사)
#    - 라벨 제거 (지도학습에 사용 불가)
# ---------------------------------------------------------
X_sagun = X_soyung[8000:]
df_sagun = pd.DataFrame(X_sagun[:, :2], columns=["연간_구매", "구매_빈도"])

# ---------------------------------------------------------
# 3. DBSCAN 클러스터링 적용
# ---------------------------------------------------------
# 데이터를 정규화 (StandardScaler 사용)
scaler = StandardScaler()
X_sagun_scaled = scaler.fit_transform(df_sagun[["연간_구매", "구매_빈도"]])

# DBSCAN 클러스터링 적용
dbscan = DBSCAN(eps=0.3, min_samples=10)  # eps와 min_samples 파라미터 조정
clusters_dbscan = dbscan.fit_predict(X_sagun_scaled)

# 클러스터 결과를 새로운 컬럼으로 추가
df_sagun["클러스터"] = clusters_dbscan

# ---------------------------------------------------------
# 4. 반품률 결측 보정 (평균 반품률로 채움)
# ---------------------------------------------------------
avg_return = df_soyung.iloc[:8000]["반품률"].mean()
df_sagun["반품률"] = avg_return

# ---------------------------------------------------------
# 5. 클러스터 번호를 임시 등급으로 간주 (pseudo-label 대체)
# ---------------------------------------------------------
df_sagun["등급"] = df_sagun["클러스터"]

# ---------------------------------------------------------
# 6. 학습용 데이터 구성 (소융 8,000명 + 사근 2,000명)
# ---------------------------------------------------------
df_train = df_soyung.iloc[:8000]
df_total = pd.concat([df_train, df_sagun], ignore_index=True)

# ---------------------------------------------------------
# 7. 최종 모델 학습 (Random Forest)
# ---------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(df_total[["연간_구매", "구매_빈도", "반품률"]], df_total["등급"])

# ---------------------------------------------------------
# 8. 검증: 소융의 마지막 2,000명 사용 (진짜 등급 있음)
# ---------------------------------------------------------
df_test = df_soyung.iloc[8000:]
X_test = df_test[["연간_구매", "구매_빈도", "반품률"]]
y_test = df_test["등급"]
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# 9. 성능 평가
# ---------------------------------------------------------
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
# plt.rcParams['font.family'] = 'Malgun Gothic'
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
            xticklabels=[f'예측 {i}' for i in range(3)],
            yticklabels=[f'실제 {i}' for i in range(3)])
plt.title("고객 등급 분류 결과 (Confusion Matrix)", fontsize=14)
plt.xlabel("예측된 등급")
plt.ylabel("실제 등급")
plt.tight_layout()
plt.show()

# scatter 추가
df_test_vis = df_test.copy()
df_test_vis["예측등급"] = y_pred

# 산점도 시각화: 연간_구매 vs 구매_빈도, 색은 예측 등급으로
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_test_vis, x="연간_구매", y="구매_빈도", hue="예측등급", palette="Set2", s=50)
plt.title("예측 등급에 따른 고객 분포 (산점도)", fontsize=14)
plt.xlabel("연간 구매 금액")
plt.ylabel("구매 빈도")
plt.legend(title="예측 등급")
plt.tight_layout()
plt.show()

# 3d 시각화
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도: 예측 등급 기준
ax.scatter(df_test_vis["연간_구매"], df_test_vis["구매_빈도"], df_test_vis["반품률"],
           c=df_test_vis["예측등급"], cmap="Set2", s=40)

ax.set_xlabel("연간 구매")
ax.set_ylabel("구매 빈도")
ax.set_zlabel("반품률")
ax.set_title("예측 등급 기반 3D 시각화")
plt.show()

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
