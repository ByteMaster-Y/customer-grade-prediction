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
이 파일 간략 설명: 데이터 생성 → 사근 데이터 구성 → 학습 → pseudo-labeling → 통합 학습 → 평가
현재 정확도가 1이 나와서 오버피팅이 의심됨 추가 작업 다음 파일에서
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
print(df_soyung)
df_soyung["등급"] = y_soyung  # 고객 등급 라벨 추가

# 2. 사근 쇼핑몰 데이터 생성 (반품률은 존재하지 않음, 라벨도 없음)
# - 실제로는 반품률 데이터가 없으므로 해당 컬럼 제외
# 소융 전체 데이터는 5000명
# 뒤에서 1000명을 사근 데이터로 사용
X_sagun = X_soyung[1000:]  # 소융 데이터를 복제해 일부만 사용
df_sagun = pd.DataFrame(X_sagun[:, :2], columns=["연간_구매", "구매_빈도"])  # 반품률 제외, 행은 전부 가져오고 열은 2개만.


# 3. 소융 쇼핑몰 데이터를 이용하여 지도학습 모델 학습
# - RandomForestClassifier 사용
# - 세 가지 특성으로 고객 등급(0~2)을 예측하도록 학습
clf = RandomForestClassifier(random_state=42) # 무작위성을 고정해서 매번 같은 결과 나오게 하기
# 소융 쇼핑몰 고객의 행동(특성 3개)을 바탕으로 등급을 예측하도록 랜덤포레스트에게 학습
clf.fit(df_soyung[["연간_구매", "구매_빈도", "반품률"]], df_soyung["등급"])

# 4. 사근 쇼핑몰 데이터에 pseudo-label 생성
# - 반품률이 없으므로 소융 데이터의 평균 반품률로 채움 (임의 보정)
avg_return = df_soyung["반품률"].mean()
df_sagun["반품률"] = avg_return  # 평균값으로 결측 특성 채움

# - 훈련된 모델을 사용해 사근 고객의 등급을 예측 (pseudo-labeling)
pseudo_labels = clf.predict(df_sagun)

# 5. 소융 + 사근 데이터를 통합한 후 다시 학습
# - 사근 데이터에 예측된 등급 추가
# - 두 데이터를 통합하여 데이터셋 확장
df_sagun["등급"] = pseudo_labels
df_total = pd.concat([df_soyung, df_sagun], ignore_index=True)

# - 통합 데이터를 이용한 최종 모델 학습
final_model = RandomForestClassifier(random_state=42)
final_model.fit(df_total[["연간_구매", "구매_빈도", "반품률"]], df_total["등급"])

# 6. 모델 성능 평가 (소융 쇼핑몰의 일부 데이터로 테스트)
# - 원래 라벨이 존재하는 소융 전체 데이터중 마지막 500명으로 평가
X_test = df_soyung[["연간_구매", "구매_빈도", "반품률"]].iloc[-500:] 
y_test = df_soyung["등급"].iloc[-500:]
y_pred = final_model.predict(X_test)

# 분류 성능 지표 출력 (precision, recall, f1-score 등)
print(classification_report(y_test, y_pred))

# Confusion matrix 계산
cm = confusion_matrix(y_test, y_pred)

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

