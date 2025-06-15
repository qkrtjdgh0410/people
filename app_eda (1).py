import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Population Trends EDA App")

uploaded_file = st.file_uploader("Upload population_trends.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 전처리
    df.replace('-', 0, inplace=True)
    for col in ['인구', '출생아수(명)', '사망자수(명)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["기초 통계", "연도별 추이", "지역별 분석", "변화량 분석", "시각화"])

    with tab1:
        st.header("Basic Statistics")
        st.subheader("DataFrame Info")
        buffer = []
        df.info(buf=buffer)
        s = "\n".join(map(str, buffer))
        st.text(s)

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Duplicates")
        st.write(df.duplicated().sum())

    with tab2:
        st.header("National Population Trend")
        df_national = df[df['지역'] == '전국']

        fig, ax = plt.subplots()
        sns.lineplot(data=df_national, x='연도', y='인구', ax=ax)
        ax.set_title("Population Trend (National)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")

        # 간단한 선형 예측 추가
        recent = df_national.sort_values('연도').tail(3)
        coef = np.polyfit(recent['연도'], recent['출생아수(명)'] - recent['사망자수(명)'], 1)
        annual_change = np.mean(recent['출생아수(명)'] - recent['사망자수(명)'])
        predicted_2035 = df_national['인구'].iloc[-1] + annual_change * (2035 - df_national['연도'].iloc[-1])
        ax.plot(2035, predicted_2035, 'ro')
        ax.text(2035, predicted_2035, f"2035: {int(predicted_2035):,}", color='red')
        st.pyplot(fig)

    with tab3:
        st.header("Population Change by Region")
        recent_years = sorted(df['연도'].unique())[-5:]
        df_recent = df[df['연도'].isin(recent_years) & (df['지역'] != '전국')]
        change = df_recent.pivot(index='지역', columns='연도', values='인구')
        change['변화량'] = change[recent_years[-1]] - change[recent_years[0]]
        change_sorted = change.sort_values('변화량', ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x='변화량', y=change_sorted.index, data=change_sorted.reset_index(), ax=ax)
        ax.set_title("Population Change (Last 5 Years)")
        st.pyplot(fig)

        change['증감률'] = ((change[recent_years[-1]] - change[recent_years[0]]) / change[recent_years[0]]) * 100
        fig2, ax2 = plt.subplots()
        sns.barplot(x='증감률', y=change.index, data=change.reset_index(), ax=ax2)
        ax2.set_title("Growth Rate (Last 5 Years)")
        st.pyplot(fig2)

    with tab4:
        st.header("Top Changes")
        df_local = df[df['지역'] != '전국']
        df_local['증감'] = df_local.groupby('지역')['인구'].diff()
        top_change = df_local.nlargest(100, '증감')

        styled_table = top_change.style.format({
            '증감': '{:,.0f}'
        }).background_gradient(subset='증감', cmap='coolwarm')

        st.dataframe(styled_table, use_container_width=True)

    with tab5:
        st.header("Heatmap Visualization")
        pivot = df[df['지역'] != '전국'].pivot(index='지역', columns='연도', values='인구')
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, cmap='YlGnBu', ax=ax, annot=False)
        st.pyplot(fig)

else:
    st.info("Please upload the population_trends.csv file.")