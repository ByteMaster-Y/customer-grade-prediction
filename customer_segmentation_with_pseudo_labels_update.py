from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Pseudo-labeling 방식 코드

소융 + 사근 데이터 통합 학습 (Pseudo-labeling 방식, 10000명 데이터)
- 소융 데이터로 학습
- 사근 데이터는 반품률 평균으로 보정 + pseudo-label 생성
- 통합하여 학습, 소융 일부로 평가
"""

# 1. 소융 쇼핑몰 데이터 생성 (10000명, 특성 3개)
X_soyung, y_soyung = make_classification(n_samples=10000, n_features=3, 
                                         n_informative=3, n_redundant=0,
                                         n_classes=3, random_state=42)

df_soyung = pd.DataFrame(X_soyung, columns=["연간_구매", "구매_빈도", "반품률"])
df_soyung["등급"] = y_soyung

# 2. 사근 데이터 생성 (소융 데이터 일부 복사, 반품률/라벨 없음)
X_sagun = X_soyung[8000:]  # 마지막 2000명 사용

# 반품률 제외, 라벨 없음
df_sagun = pd.DataFrame(X_sagun[:, :2], columns=["연간_구매", "구매_빈도"])

# 3. 소융 데이터로 초기 지도 학습 모델 학습 (상위 8000명)
df_train = df_soyung.iloc[:8000]
clf = RandomForestClassifier(random_state=42)
clf.fit(df_train[["연간_구매", "구매_빈도", "반품률"]], df_train["등급"])

# 4. 사근 데이터에 평균 반품률 채우기 + pseudo-label 예측
avg_return = df_train["반품률"].mean()
df_sagun["반품률"] = avg_return
pseudo_labels = clf.predict(df_sagun)
df_sagun["등급"] = pseudo_labels

# 5. 소융 + 사근 통합하여 다시 학습
# 전체 데이터셋 구성
df_total = pd.concat([df_train, df_sagun], ignore_index=True)

final_model = RandomForestClassifier(random_state=42)
final_model.fit(df_total[["연간_구매", "구매_빈도", "반품률"]], df_total["등급"])

# 6. 소융 검증셋으로 평가 (나머지 2000명)
df_test = df_soyung.iloc[8000:]
X_test = df_test[["연간_구매", "구매_빈도", "반품률"]]
y_test = df_test["등급"]
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred))

# 혼동행렬 시각화
# plt.rcParams['font.family'] = 'Malgun Gothic'
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# Confusion matrix 시각화
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("예측된 등급")
plt.ylabel("실제 등급")
plt.title("고객 등급 분류 결과 (Confusion Matrix)")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
