# 태양광 발전량 예측 대시보드

태양광 발전소의 발전량을 예측하고 시각화하는 대시보드 애플리케이션입니다.

## 기능

- CSV 파일 업로드 및 데이터 미리보기
- 결측값 및 이상치 자동 감지 및 처리
- Prophet 모델을 활용한 발전량 예측
- 예측 결과 시각화 및 성능 지표 확인
- 예측 결과 CSV 파일 저장

## 설치 방법

1. Python 3.8 이상 설치

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 필요한 디렉토리 구조 확인
```
.
├── Dashboard.py              # 대쉬보드 실행 스크립트 
├── requirements.txt          # 환경 패키지 
├── IQR_data                  # 이상치 기준 데이터
├── feature_scaler/           # 피쳐 스케일링 파일 
├── target_scaler/            # 타겟 스케일링 파일 
├── prophet/                  # 예측 모델 (prophet)
└── output/                   # 예측 후 저장 경로 
```

## 사용 방법

1. 대시보드 실행
```bash
cd Inference Dashboard
streamlit run Dashboard.py
```

2. 웹 브라우저에서 대시보드 접속
   - 기본 주소: http://localhost:8501

3. 데이터 업로드
   - 왼쪽 사이드바에서 CSV 파일 업로드
   - 파일 형식: CSV (인코딩: cp949)
   - 필수 컬럼: ymdhms, id, generate_gap, Ampe_R, Ampe_S, Ampe_T, 일사(MJ/m2), 기온(°C)

4. ID 선택
   - 사이드바에서 예측할 발전소 ID 선택

5. 결과 확인
   - 데이터 미리보기
   - 결측값/이상치 감지 결과
   - 예측 결과 그래프
   - MAPE 성능 지표
   - 예측 결과는 자동으로 output 디렉토리에 저장

## 입력 데이터 형식

CSV 파일은 다음 컬럼들을 포함해야 합니다:
- ymdhms: 날짜 및 시간 (YYYY-MM-DD HH:mm:ss)
- id: 발전소 ID
- generate_gap: 발전량
- Ampe_R: R상 전류
- Ampe_S: S상 전류
- Ampe_T: T상 전류
- 일사(MJ/m2): 일사량
- 기온(°C): 기온

## 출력 결과

1. 예측 결과 파일
   - 저장 위치: output 디렉토리
   - 파일명: [원본파일명]_pred.csv
   - 컬럼: ymdhms, GT(실제값), Pred(예측값)

2. 성능 지표
   - MAPE (Mean Absolute Percentage Error)

## 주의사항

1. 필요한 모델 파일들이 올바른 위치에 있어야 합니다:
   - feature_scaler: 특성 스케일링 모델
   - target_scaler: 타겟 스케일링 모델
   - prophet: 예측 모델

2. 입력 데이터는 반드시 지정된 형식을 따라야 합니다.

3. 대용량 데이터 처리 시 메모리 사용량에 주의하세요.

## 문제 해결

1. 파일 업로드 오류
   - CSV 파일 인코딩이 cp949인지 확인
   - 필수 컬럼이 모두 있는지 확인

2. 예측 오류
   - 모델 파일들이 올바른 위치에 있는지 확인
   - 입력 데이터 형식이 올바른지 확인

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 
