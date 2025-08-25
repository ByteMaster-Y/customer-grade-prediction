import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

"""
pseudo_labeling_clustering_randomforest 실험에서는 정확도는 높았지만, 분류된 이미지에서 데이터 값들이 한데 뭉쳐 제대로 분리되지 않는 단점이 발견되었습니다.
정확도와 시각적으로 잘 분리된 이미지 사이에서 균형을 고민하게 되었고, 이에 따라 코드를 한 번 더 개선해 보았습니다.
그 결과, 정확도는 0.74로 다소 낮아졌지만, 시각화된 분류 결과는 훨씬 더 잘 분리된 모습으로 개선되었습니다.
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
# 3. 사근 데이터에 반품률 컬럼 추가 (0으로 채움)
# ---------------------------------------------------------
df_sagun["반품률"] = 0

# ---------------------------------------------------------
# 4. 데이터 스케일링
#    - 소융 데이터로 fit 후, 소융과 사근 둘 다 동일하게 transform
# ---------------------------------------------------------
scaler = StandardScaler()
df_soyung[["연간_구매", "구매_빈도", "반품률"]] = scaler.fit_transform(df_soyung[["연간_구매", "구매_빈도", "반품률"]])
df_sagun[["연간_구매", "구매_빈도", "반품률"]] = scaler.transform(df_sagun[["연간_구매", "구매_빈도", "반품률"]])

# ---------------------------------------------------------
# 5. 사근 데이터에 클러스터링 적용 (Gaussian Mixture)
# ---------------------------------------------------------
gmm = GaussianMixture(n_components=3, random_state=42)
sagun_cluster = gmm.fit_predict(df_sagun[["연간_구매", "구매_빈도"]])
df_sagun["예측등급"] = sagun_cluster

# ---------------------------------------------------------
# 6. 소융 데이터를 가지고 임시 모델 학습 후, 사근 데이터에 대한 confidence 계산
# ---------------------------------------------------------
X_temp = df_soyung[["연간_구매", "구매_빈도", "반품률"]]
y_temp = df_soyung["등급"]

temp_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
temp_model.fit(X_temp, y_temp)

# 사근 데이터 전체 피처 추출
X_sagun_full = df_sagun[["연간_구매", "구매_빈도", "반품률"]]
sagun_probs = temp_model.predict_proba(X_sagun_full)
sagun_confidence = np.max(sagun_probs, axis=1)
df_sagun["confidence"] = sagun_confidence

# ---------------------------------------------------------
# 7. confidence 높은 pseudo-label만 추출 (임계치: 0.9)
# ---------------------------------------------------------
df_sagun_filtered = df_sagun[df_sagun["confidence"] > 0.9].copy()
df_sagun_filtered["등급"] = df_sagun_filtered["예측등급"]

# ---------------------------------------------------------
# 8. 소융 + 사근(pseudo) 데이터 결합 후 학습용 데이터 구성
# ---------------------------------------------------------
df_combined = pd.concat([df_soyung, df_sagun_filtered], ignore_index=True)

X = df_combined[["연간_구매", "구매_빈도", "반품률"]]
y = df_combined["등급"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 9. 최종 모델 학습 (Logistic Regression)
# ---------------------------------------------------------
final_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 10. PCA 시각화 (2D)
# ---------------------------------------------------------
# plt.rcParams['font.family'] = 'Malgun Gothic'
X_test_pca = PCA(n_components=2).fit_transform(X_test)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
df_vis = pd.DataFrame(X_test_pca, columns=["PCA1", "PCA2"])
df_vis["예측등급"] = y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="예측등급", palette="Set2", s=80)
plt.title("PCA 기반 예측 등급 분포 시각화", fontsize=14)
plt.tight_layout()
plt.show()
