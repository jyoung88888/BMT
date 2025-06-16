import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from streamlit_echarts import st_echarts
import time
from datetime import datetime
import os

# 페이지 설정
st.set_page_config(page_title="태양광 예측 모델", layout="wide")

# CSS 스타일 적용
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.title("☀️ 태양광 발전량 예측")

# 시작 시간 기록
start_time = time.time()
start_datetime = datetime.now()

# 데이터 업로드
st.sidebar.header("데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

features = ['Ampe_R', 'Ampe_S','Ampe_T','일사(MJ/m2)', '기온(°C)']
iqr_features = ['Ampe_R', 'Ampe_S','Ampe_T','일사', '기온']

# output_dir 생성
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def data_eda(df,l,u):
    df['ymdhms'] = pd.to_datetime(df['ymdhms'])
    threshold = 30  # 연속으로 유지되는 최소 횟수
    is_zero = df['generate_gap'] == 0  # 값이 0인 위치를 True로 표시
    zero_group = is_zero.astype(int).groupby(is_zero.ne(is_zero.shift()).cumsum()).cumsum()  # 연속 카운트
    df['Hour'] = df.ymdhms.dt.hour

    # 조건 만족하는 구간 추출
    long_zero_indices = df.index[zero_group >= threshold]
    df = df.drop(long_zero_indices,axis=0).reset_index(drop=True)
    df.loc[df.Hour == 0 , 'generate_gap'] = 0
    
    #df_filled = df.fillna(method='bfill')
    
    return df

if uploaded_file is not None:
    # 데이터 로드
    df = pd.read_csv(uploaded_file,encoding='cp949')
    # 업로드된 파일 이름 저장
    uploaded_filename = uploaded_file.name
    base_filename = os.path.splitext(uploaded_filename)[0]  # 확장자 제외한 파일명
    
    # 데이터 미리보기 섹션
    st.markdown("### 📊 데이터 미리보기")
    st.markdown("#### 데이터 기본 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("기간", f"{df['ymdhms'].iloc[0]} ~ {df['ymdhms'].iloc[-1]}")
    with col2:
        st.metric("행 수", f"{len(df):,}행")
    with col3:
        st.metric("열 수", f"{len(df.columns):,}열")
    
    # 데이터 미리보기 표시
    st.dataframe(df, use_container_width=True)

    # 특징 및 타겟 선택
    st.sidebar.subheader("특징 및 타겟 선택")
    target_col = 'generate_gap'
    ids =  st.sidebar.selectbox("ID 선택", df.id.unique())
    

    if target_col and ids:
        iqr_df = pd.read_csv('./IQR_DATA/iqr_df.csv')
        l = iqr_df.iloc[ids-1]['lower']
        u = iqr_df.iloc[ids-1]['upper']

        data = data_eda(df,l,u)  

        # 각 feature에 대해 이상치 처리
        for feature,iqr_features in zip(features,iqr_features):
            feature_iqr = pd.read_csv(f'./IQR_DATA/{iqr_features}.csv')
            mi = feature_iqr.iloc[ids-1]['Min']
            ma = feature_iqr.iloc[ids-1]['Max']
            avg = feature_iqr.iloc[ids-1]['Avg']

            feature_values = data[feature]
            
            # 결측값 확인 및 처리
            missing_mask = feature_values.isnull()
            if missing_mask.sum() > 0:
                # 결측값 발생 정보 출력
                missing_indices = data.index[missing_mask]
                missing_times = data.loc[missing_mask, 'ymdhms']
                
                st.markdown(f"### ⚠️ {feature} 결측값 감지")
                st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
                st.markdown(f"- 발견된 결측값 개수: **{len(missing_indices):,}개**")
                
                # 결측값 상세 정보를 데이터프레임으로 표시
                missing_df = pd.DataFrame({
                    '시간': missing_times,
                    '대체될 평균값': avg
                })
                st.dataframe(missing_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 결측값을 평균값으로 대체
                data.loc[missing_mask, feature] = avg
            
            # 이상치 처리
            outlier_mask = (feature_values < mi) | (feature_values > ma)
            if outlier_mask.sum() > 0:
                # 이상치 발생 정보 출력
                outlier_indices = data.index[outlier_mask]
                outlier_values = data.loc[outlier_mask, feature]
                outlier_times = data.loc[outlier_mask, 'ymdhms']
                
                st.markdown(f"### ⚠️ 이상치 감지 : {feature}")
                st.markdown(f"- 발견된 이상치 개수: **{len(outlier_indices):,}개**")
                
                # 이상치 상세 정보를 데이터프레임으로 표시
                outlier_df = pd.DataFrame({
                    '시간': outlier_times,
                    '이상치 값': outlier_values,
                    '평균값으로 대체': avg
                })
                st.dataframe(outlier_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                data.loc[outlier_mask, feature] = avg
    

        zero_index = data.loc[data['generate_gap'] <= 0].index
        value_index = data.loc[data['generate_gap'] > 0].index  

        X_test = data.iloc[value_index][features]
        y_test = data.iloc[value_index][target_col]
        
        fea_sc_path = './feature_scaler/feature_scaler_' + str(ids)+ '.pkl'
        with open(fea_sc_path, 'rb') as file:
            feature_scaler = pickle.load(file)

        tar_sc_path = './target_scaler/target_scaler_' + str(ids)+ '.pkl'
        with open(tar_sc_path, 'rb') as file:
            target_scaler = pickle.load(file)
        
        #정규화
        Scaled_X_test= feature_scaler.transform(X_test)

        Test_df = pd.DataFrame(Scaled_X_test, columns=features)
        Test_df['ds'] = data.iloc[value_index]['ymdhms'].values

        # 모델 로드 
        model_path_ = './prophet/solar_model' + str(ids)+ '.pkl'
        with open(model_path_, "rb") as f:
            loaded_model = pickle.load(f)

        # 예측
        forecast = loaded_model.predict(Test_df)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))

        # 역변환
        y_pred_sc_bi = target_scaler.inverse_transform(pd.DataFrame(forecast['yhat'].values, index=value_index))

        predict_result = pd.DataFrame(index = range(0,len(data)),columns=['ymdhms','GT','Pred'])
        predict_result['ymdhms'] = data['ymdhms']  
        predict_result['ymdhms'] = data['ymdhms'].dt.strftime("%Y-%m-%d %H:00")       

        predict_result.iloc[value_index, predict_result.columns.get_loc('GT')] = list(y_test)
        predict_result.iloc[value_index, predict_result.columns.get_loc('Pred')] = list(y_pred_sc_bi.squeeze())

        predict_result.iloc[zero_index, predict_result.columns.get_loc('GT')] = 0 
        predict_result.iloc[zero_index, predict_result.columns.get_loc('Pred')] = 0 

        # 결과 저장
        # output_filename = f'{base_filename}_pred.csv'
        # output_path = os.path.join(output_dir, output_filename)
        # predict_result.to_csv(output_path, index=False, encoding='cp949')
        # st.success(f"예측 결과가 저장되었습니다: {output_filename}")

        # 성능 평가
        MAPE = np.mean(np.abs((predict_result.iloc[value_index]['GT'] - predict_result.iloc[value_index]['Pred']) /  predict_result.iloc[value_index]['GT'])) * 100
        
        # 종료 시간 기록 및 실행 시간 계산
        end_time = time.time()
        end_datetime = datetime.now()
        execution_time = end_time - start_time
        
        # 결과 출력 
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 모델 성능")
            st.metric("MAPE", f"{MAPE:.2f}%")
        with col2:
            st.markdown("### ⏱️ 프로그램 실행 정보")
            st.metric("실행 시간", f"{execution_time:.2f}초")
        

        # 예측 결과 시각화
        st.markdown("### 📉 실제값 vs 예측값 그래프")
        st.markdown("#### 시계열 예측 결과")

        # Convert datetime to string format
        x_axis_data = [str(dt) for dt in predict_result['ymdhms'].values]

        option = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Actual", "Predicted"]},
            "xAxis": {
                "type": "category",
                "data": x_axis_data,
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": "Actual",
                    "type": "line",
                    "data": list(predict_result['GT'].values),
                    "smooth": True,
                },
                {
                    "name": "Predicted",
                    "type": "line",
                    "data": list(predict_result['Pred'].values),
                    "smooth": True,
                },
            ],
        }

        # ECharts 차트 출력
        st_echarts(options=option, height="500px")

    # 결과 저장
        output_filename = f'{base_filename}_pred.csv'
        output_path = os.path.join(output_dir, output_filename)
        predict_result.to_csv(output_path, index=False, encoding='cp949')
        st.success(f"예측 결과가 저장되었습니다: {output_filename}")

else:
    st.info("CSV 파일을 업로드하세요.")

