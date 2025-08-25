# 고객 등급 예측 프로젝트 (소융 & 사근 쇼핑몰 통합)

![Header](https://img.shields.io/badge/AI-Python-blue) ![ML](https://img.shields.io/badge/ML-Scikit--learn-green)

## 프로젝트 개요
본 프로젝트는 두 쇼핑몰의 고객 데이터를 활용하여 고객 등급을 예측하는 모델을 개발하는 것을 목표로 합니다.

- **소융 쇼핑몰**: 연간 구매 금액, 구매 빈도, 반품률 + 고객 등급 존재 → **지도 학습 가능**
- **사근 쇼핑몰**: 연간 구매 금액, 구매 빈도만 존재 → **비지도 학습 필요**
- 라벨이 있는 소융 데이터와 라벨이 없는 사근 데이터를 모두 활용하여, **고객 등급(0~2)을 예측**합니다.

---

## 프로젝트 배경
최근 인수한 사근 쇼핑몰은 일부 특성만 존재하고 고객 등급 레이블이 없어, 기존 소융 쇼핑몰 데이터를 활용해 모델을 학습하고 사근 데이터를 통합하는 방법이 필요했습니다.  

**목표**:
1. 소융 데이터만으로 안정적인 baseline 성능 확보
2. 라벨 없는 사근 데이터 활용
3. 모델 일반화 성능 향상

---

## 실험 방법

### 1. Baseline (소융 데이터만 활용)
- **데이터**: 5,000명 (학습 4,500 / 테스트 500)
- **모델**: RandomForestClassifier
- **목적**: 일반화 성능 확인, 오버피팅 점검
- **결과**
  - 정확도: **0.87**
  - Precision/Recall/F1: 0.83~0.92
  - 장점: 안정적, 구현 간단
  - 단점: 사근 데이터 활용 불가  

**Baseline 예시 시각화**
<img width="1583" height="590" alt="Image" src="https://github.com/user-attachments/assets/9a9d8350-8edf-40f7-8236-40f2f221152d" />

---

### 2. Pseudo-labeling 기반 통합 학습
- **방법**
  1. 소융 데이터로 초기 모델 학습
  2. 사근 데이터에 평균 반품률 삽입 → pseudo-label 생성
  3. 소융 + 사근 통합 데이터로 모델 재학습
  4. 소융 검증셋으로 성능 평가
- **데이터**
  - 소융: 8,000명
  - 사근: 2,000명
  - 통합 후: 10,000명
- **결과**
  - 정확도: **0.92**
  - 클래스별 F1: 0.90~0.93
- **장점**: 라벨 없는 데이터 활용 가능, 성능 향상
- **단점**: pseudo-label 품질 의존, noise 존재 가능

**Pseudo-label 예시 시각화**
<img width="1082" height="390" alt="Image" src="https://github.com/user-attachments/assets/4ec26cee-e5b0-422e-a610-22341102df96" />


---

### 3. 클러스터링 기반 학습
- **방법**
  1. 소융 데이터 학습 (8,000명)
  2. 사근 데이터에 KMeans(K=3) 클러스터링 적용
  3. 각 클러스터를 pseudo-label처럼 활용
  4. 통합 데이터로 최종 모델 학습
  5. 소융 검증셋으로 평가
- **결과**
  - 정확도: **0.91**
  - 클래스별 F1: 0.90~0.93
- **장점**: 데이터 분포 반영, 비지도 데이터 활용 가능
- **단점**: 초기화, 하이퍼파라미터 의존, 구현 복잡

**클러스터링 시각화 예시**
<img width="1082" height="391" alt="Image" src="https://github.com/user-attachments/assets/a9e57044-79fe-40d1-894c-03eaf1efd088" />
<img width="1085" height="380" alt="Image" src="https://github.com/user-attachments/assets/6457cab4-d434-4ab6-a956-d065decf3db3" />
<img width="1466" height="735" alt="Image" src="https://github.com/user-attachments/assets/5ed519fe-06c2-407f-81f5-90c1664a7e5a" />
---

## 실험 비교

| 실험 방법 | 정확도 | 장점 | 단점 |
|-----------|--------|------|------|
| Baseline | 0.87 | 안정적, 구현 간단 | 라벨 없는 데이터 활용 불가 |
| Pseudo-labeling | 0.92 | 라벨 없는 데이터 활용 가능, 성능 향상 | pseudo-label noise 존재, 품질 의존 |
| 클러스터링 기반 | 0.91 | 데이터 분포 반영, 라벨 다양성 확보 | 초기화·하이퍼파라미터 의존, 구현 복잡 |

## 실험 장단점 비교

| 구분 | 장점 | 단점 |
|------|------|------|
| **첫 번째 실험 (Pseudo-labeling)** | - 간단한 구현<br>- 보정된 데이터 활용 가능<br>- 효율적 학습과 평가 | - 정확성 문제 (평균 값 사용)<br>- 성능 향상 제한 (잘못된 pseudo-label 생성)<br>- 라벨 신뢰성 문제 (pseudo-label 불확실성) |
| **두 번째 실험 (클러스터링 기반)** | - 데이터 분포 반영<br>- 라벨 다양성 증가<br>- 성능 향상 가능성 | - 클러스터링 한계 (군집화 성능 의존)<br>- 복잡성 증가 (추가 하이퍼파라미터 튜닝 필요)<br>- 초기화 문제 (KMeans 초기화 의존) |

---

## 결론
- **빠른 구현 & 간단 실험** → Pseudo-labeling  
- **데이터 분포 반영 & 일반화 성능 향상** → 클러스터링 기반  
- 향후 개선 방향:
  - pseudo-label 품질 향상
  - 더 정교한 클러스터링 기법 적용
  - 정확도와 시각화 간 균형 최적화

---

## 기술 스택
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn
- RandomForestClassifier, KMeans

---

## GitHub
[프로젝트 코드 바로가기](#)  <!-- 실제 URL로 교체 -->

