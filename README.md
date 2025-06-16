# 태양광 발전량 예측 대시보드

## 프로젝트 개요
이 프로젝트는 태양광 발전량을 예측하고 시각화하는 대시보드 애플리케이션입니다.

## 환경 설정

### 필수 라이브러리
```bash
pip install streamlit
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install streamlit-echarts
```

### Python 버전
- Python 3.8 이상

## Dashboard.py 주요 기능

### 1. 데이터 전처리
- 결측값 처리
- 이상치 처리 (lower/upper bound 기반)
- 시간 데이터 변환

### 2. 이상치 처리 과정
- 각 feature별 lower/upper bound 확인
- 범위를 벗어나는 값 탐지
- 이상치를 평균값으로 대체
- 이상치 발생 정보 출력

### 3. 결측값 처리 과정
- 결측값 위치 탐지
- 결측값 개수 확인
- 결측값을 평균값으로 대체
- 결측값 발생 정보 출력

### 4. 모델 예측
- Prophet 모델 사용
- 특징 정규화
- 예측값 역변환
- MAPE 성능 평가

### 5. 시각화
- 실제값 vs 예측값 비교 그래프
- ECharts를 활용한 인터랙티브 차트
- 이상치/결측값 정보 표시

## 실행 방법
1. 필요한 라이브러리 설치
2. 터미널에서 다음 명령어 실행:
```bash
streamlit run Dashboard.py
```

## 파일 구조
```
BMT/
├── Dashboard.py          # 메인 대시보드 애플리케이션
├── feature_scaler/       # 특징 정규화 모델
├── target_scaler/        # 타겟 정규화 모델
├── prophet/             # Prophet 모델
└── iqr_df_2.csv        # 이상치 처리 기준값
```

## 주의사항
- 대용량 데이터 파일은 별도 관리
- 모델 파일(.pkl)은 git에서 제외
- 환경 변수 설정 필요

## 성능 지표
- MAPE (Mean Absolute Percentage Error)
- 0이 아닌 값 기준으로 평가

## 업데이트 내역
- 2024.03: 이상치/결측값 처리 기능 추가
- 2024.03: 실행 시간 측정 기능 추가
- 2024.03: UI 개선 및 결과 저장 기능 추가
