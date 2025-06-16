import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(layout="wide")

st.title("Population Trend Analysis")

uploaded_file = st.file_uploader("Upload population_trends.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 기본 전처리
    df.replace('-', 0, inplace=True)
    df[['인구', '출생아수(명)', '사망자수(명)']] = df[['인구', '출생아수(명)', '사망자수(명)']].apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["기초 통계", "연도별 추이", "지역별 분석", "변화량 분석", "시각화"])

    with tab1:
        st.header("데이터프레임 구조 및 통계")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.dataframe(df.describe())

    with tab2:
        st.header("연도별 전체 인구 추이")
        df_total = df[df['지역'] == '전국']
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_total, x='연도', y='인구')
        # 예측
        recent = df_total.sort_values('연도', ascending=False).head(3)
        birth_mean = recent['출생아수(명)'].mean()
        death_mean = recent['사망자수(명)'].mean()
        last_year = df_total['연도'].max()
        last_pop = df_total[df_total['연도'] == last_year]['인구'].values[0]
        predicted_2035 = last_pop + (birth_mean - death_mean) * (2035 - last_year)
        plt.axhline(predicted_2035, color='red', linestyle='--')
        plt.text(2035, predicted_2035, f'2035 Predicted: {int(predicted_2035):,}', color='red')
        plt.title("Population Trend")
        plt.xlabel("Year")
        plt.ylabel("Population")
        st.pyplot(plt)

    with tab3:
        st.header("지역별 인구 변화량 순위")
        recent_years = sorted(df['연도'].unique())[-5:]
        df_recent = df[df['연도'].isin(recent_years) & (df['지역'] != '전국')]
        pivot = df_recent.pivot(index='지역', columns='연도', values='인구')
        delta = pivot[recent_years[-1]] - pivot[recent_years[0]]
        delta_rate = (delta / pivot[recent_years[0]]) * 100
        df_change = pd.DataFrame({
            '지역': delta.index,
            '변화량': delta.values / 1000,
            '변화율': delta_rate.values
        }).sort_values('변화량', ascending=False)
        # 영어로 지역명 치환 (간단 예시)
        translate = {'서울': 'Seoul', '부산': 'Busan', '세종': 'Sejong', '대구': 'Daegu', '인천': 'Incheon', '광주': 'Gwangju',
                     '대전': 'Daejeon', '울산': 'Ulsan', '경기': 'Gyeonggi', '강원': 'Gangwon', '충북': 'Chungbuk',
                     '충남': 'Chungnam', '전북': 'Jeonbuk', '전남': 'Jeonnam', '경북': 'Gyeongbuk', '경남': 'Gyeongnam',
                     '제주': 'Jeju'}
        df_change['지역'] = df_change['지역'].map(translate).fillna(df_change['지역'])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_change, x='변화량', y='지역', ax=ax)
        for i, v in enumerate(df_change['변화량']):
            ax.text(v, i, f'{v:,.1f}', va='center')
        ax.set_title("Population Change (K)")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_change, x='변화율', y='지역', ax=ax2)
        for i, v in enumerate(df_change['변화율']):
            ax2.text(v, i, f'{v:.1f}%', va='center')
        ax2.set_title("Population Change Rate (%)")
        st.pyplot(fig2)

        st.markdown("**Interpretation**: The above charts show total and relative population changes by region in the last 5 years.")

    with tab4:
        st.header("증감률 상위 지역/연도")
        df_diff = df[df['지역'] != '전국'].copy()
        df_diff['증감'] = df_diff.groupby('지역')['인구'].diff()
        top_diff = df_diff.sort_values('증감', ascending=False).head(100)
        top_diff['증감_천명'] = (top_diff['증감'] / 1000).astype(int)
        top_diff['증감_색'] = np.where(top_diff['증감'] > 0, '증가', '감소')

        st.dataframe(
            top_diff[['연도', '지역', '인구', '증감_천명']]
            .style.background_gradient(subset=['증감_천명'], cmap='coolwarm')
            .format({'인구': '{:,}', '증감_천명': '{:,}'})
        )

    with tab5:
        st.header("누적 인구 영역 그래프")
        df_pivot = df.pivot_table(index='연도', columns='지역', values='인구', aggfunc='sum')
        df_pivot = df_pivot.drop(columns='전국', errors='ignore')
        df_pivot.fillna(0, inplace=True)
        df_pivot.columns = [translate.get(col, col) for col in df_pivot.columns]

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        df_pivot.plot.area(ax=ax3, cmap='tab20')
        ax3.set_title("Population by Region Over Time")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Population")
        st.pyplot(fig3)