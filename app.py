import streamlit as st
import pandas as pd

from methods import get_history, get_imoex, market_metrics, calc_beta, calc_moving_averages, calc_rsi

st.title("Анализ акций Мосбиржи")

tickers = ["AFLT", "SVCB", "GAZP", "MGNT", "MVID", "SBER", "YDEX"]

ticker = st.selectbox("Выберите акцию", tickers)

date_from = st.date_input("Дата начала", value=pd.to_datetime("2026-01-01"))
date_to = st.date_input("Дата окончания", value=pd.to_datetime("2026-05-01"))

history = get_history(
    ticker,
    str(date_from),
    str(date_to)
)

st.subheader("История цены закрытия")
st.line_chart(history.set_index("TRADEDATE")["CLOSE"])

metrics = market_metrics(
    history["CLOSE"].tolist(),
    history["VOLUME"].tolist()
)

st.subheader("Рыночные показатели")

col1, col2, col3 = st.columns(3)

col1.metric("Доходность", f"{metrics['total_return_pct']}%")
col2.metric("Волатильность", f"{metrics['annual_volatility_pct']}%")
col3.metric("Просадка", f"{metrics['max_drawdown_pct']}%")

col4, col5 = st.columns(2)

col4.metric("Средний объем", f"{metrics['avg_volume']:.0f}")
col5.metric("Sharpe", metrics["sharpe_ratio"])

ma = calc_moving_averages(history["CLOSE"].tolist())
rsi = calc_rsi(history["CLOSE"].tolist(), period=14)

st.subheader("Технические сигналы")
st.write("Скользящие средние:", ma)
st.write("RSI:", rsi)

st.subheader("Данные")
st.dataframe(history)