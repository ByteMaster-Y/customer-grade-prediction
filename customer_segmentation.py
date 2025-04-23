from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


"""
첫번째 파일에 정확도가 너무 높아 사근 데이터 없이 소융 데이터 일부만 이용해 학습하고 나머지로 평가.
이 파일 간략 설명: 데이터 생성 → 학습/검증 데이터 분리 → baseline 모델 학습 → 평가
"""

# 1. 소융 쇼핑몰 데이터 생성 (5000명, 특성 3개)
# - make_classification을 사용해서 연간 구매, 구매 빈도, 반품률 3개의 특성
# - 고객 등급(0, 1, 2) 라벨 포함한 데이터 생성
# X_soyung = 입력 특성 (Features)
# y_soyung = 정답 라벨 (Target)
X_soyung, y_soyung = make_classification(n_samples=5000, n_features=3, 
                                         n_informative=3, n_redundant=0,
                                         n_classes=3, random_state=42)

# 생성한 데이터를 DataFrame으로 변환하고 컬럼 이름 지정
df_soyung = pd.DataFrame(X_soyung, columns=["연간_구매", "구매_빈도", "반품률"])
df_soyung["등급"] = y_soyung  # 고객 등급 라벨 추가

# 2. baseline 모델: 소융 데이터만 사용하여 학습 및 평가
# - 처음 4500명으로 학습, 마지막 500명으로 평가
df_train = df_soyung.iloc[:4500]
df_test = df_soyung.iloc[4500:]

# - 학습
clf = RandomForestClassifier(random_state=42)
clf.fit(df_train[["연간_구매", "구매_빈도", "반품률"]], df_train["등급"])

# - 예측
X_test = df_test[["연간_구매", "구매_빈도", "반품률"]]
y_test = df_test["등급"]
y_pred = clf.predict(X_test)

# 분류 성능 지표 출력 (precision, recall, f1-score 등)
print(classification_report(y_test, y_pred))

# Confusion matrix 계산
cm = confusion_matrix(y_test, y_pred)

# 시각화
# 한글 폰트 설정
# plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용
# 한글 폰트 설정 (맥북용)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# Confusion matrix 시각화
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("예측된 등급")
plt.ylabel("실제 등급")
plt.title("고객 등급 분류 결과 (Confusion Matrix)")
plt.show()

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
