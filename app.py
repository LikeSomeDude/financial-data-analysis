import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from methods import get_history, get_imoex, market_metrics, calc_beta, calc_moving_averages, calc_rsi

st.title("Анализ акций Мосбиржи")

tickers = ["AFLT", "SVCB", "GAZP", "MGNT", "MVID", "SBER", "YDEX"]

ticker = st.selectbox("Выберите акцию", tickers)

date_to = st.date_input("Дата окончания", value=pd.Timestamp.today())
period = st.selectbox(
    "Период анализа",
    ["Произвольный период", "1 месяц", "6 месяцев", "1 год", "3 года", "5 лет"]
)

if period == "Произвольный период":
    date_from = st.date_input("Дата начала", value=pd.to_datetime("2026-01-01"))
else:
    period_offsets = {
        "1 месяц": pd.DateOffset(months=1),
        "6 месяцев": pd.DateOffset(months=6),
        "1 год": pd.DateOffset(years=1),
        "3 года": pd.DateOffset(years=3),
        "5 лет": pd.DateOffset(years=5)
    }

    date_from = (pd.to_datetime(date_to) - period_offsets[period]).date()
    st.caption(f"Дата начала рассчитана автоматически: {date_from}")

date_from_str = pd.to_datetime(date_from).strftime("%Y-%m-%d")
date_to_str = pd.to_datetime(date_to).strftime("%Y-%m-%d")

history = get_history(
    ticker,
    date_from_str,
    date_to_str
)

currency_id = history["CURRENCYID"].iloc[0] if not history.empty else ""
currency = "RUB" if currency_id == "SUR" else currency_id
actual_date_from = history["TRADEDATE"].min() if not history.empty else "нет данных"
actual_date_to = history["TRADEDATE"].max() if not history.empty else "нет данных"

st.subheader("История цены закрытия")
st.caption(f"Фактический период данных: {actual_date_from} — {actual_date_to}")
st.caption(f"Цены отображаются в валюте торгов: {currency}")
st.line_chart(history.set_index("TRADEDATE")["CLOSE"], use_container_width=True)

st.subheader("Свечной график")
last_row = history.iloc[-1]
previous_close = history["CLOSE"].iloc[-2] if len(history) > 1 else last_row["CLOSE"]
price_change = last_row["CLOSE"] - previous_close
price_change_pct = price_change / previous_close * 100 if previous_close != 0 else 0
change_color = "#089981" if price_change >= 0 else "#f23645"
volume_colors = [
    "#8bd3c7" if close_price >= open_price else "#ff9b9b"
    for open_price, close_price in zip(history["OPEN"], history["CLOSE"])
]

st.markdown(
    f"""
    **{last_row['SHORTNAME']} · 1Д · {currency}** &nbsp;&nbsp;
    ОТКР <span style="color:#089981">{last_row['OPEN']:.2f}</span>
    МАКС <span style="color:#089981">{last_row['HIGH']:.2f}</span>
    МИН <span style="color:#089981">{last_row['LOW']:.2f}</span>
    ЗАКР <span style="color:#089981">{last_row['CLOSE']:.2f}</span>
    <span style="color:{change_color}">{price_change:+.2f} ({price_change_pct:+.2f}%)</span>
    """,
    unsafe_allow_html=True
)

candlestick = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22]
)

candlestick.add_trace(
    go.Candlestick(
        x=history["TRADEDATE"],
        open=history["OPEN"],
        high=history["HIGH"],
        low=history["LOW"],
        close=history["CLOSE"],
        increasing_line_color="#089981",
        increasing_fillcolor="#089981",
        decreasing_line_color="#f23645",
        decreasing_fillcolor="#f23645",
        name=ticker
    ),
    row=1,
    col=1
)

candlestick.add_trace(
    go.Bar(
        x=history["TRADEDATE"],
        y=history["VOLUME"],
        marker_color=volume_colors,
        name="Объем"
    ),
    row=2,
    col=1
)

candlestick.update_layout(
    template="plotly_white",
    xaxis_rangeslider_visible=False,
    height=720,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified",
    showlegend=False
)
candlestick.update_yaxes(title_text=f"Цена, {currency}", row=1, col=1)
candlestick.update_yaxes(title_text="Объем", row=2, col=1)
candlestick.update_xaxes(showgrid=True, gridcolor="#eeeeee")
candlestick.update_yaxes(showgrid=True, gridcolor="#eeeeee")
st.plotly_chart(candlestick, use_container_width=True)

metrics = market_metrics(
    history["CLOSE"].tolist(),
    history["VOLUME"].tolist()
)

beta = calc_beta(
    history["CLOSE"].tolist(), 
    get_imoex(date_from_str, date_to_str)["CLOSE"].tolist()
)

st.subheader("Рыночные показатели")

col1, col2, col3 = st.columns(3)

col1.metric("Доходность", f"{metrics['total_return_pct']}%")
col2.metric("Волатильность", f"{metrics['annual_volatility_pct']}%")
col3.metric("Просадка", f"{metrics['max_drawdown_pct']}%")

col4, col5, col6 = st.columns(3)

col4.metric("Средний объем", f"{metrics['avg_volume']:.0f}")
col5.metric("Sharpe", metrics["sharpe_ratio"])
col6.metric("Beta", beta["beta"])

last_close = history["CLOSE"].iloc[-1]
st.metric("Последняя цена закрытия", f"{last_close:.2f} {currency}")

ma = calc_moving_averages(history["CLOSE"].tolist())
rsi = calc_rsi(history["CLOSE"].tolist(), period=14)

st.subheader("Технические сигналы")
st.write("Скользящие средние:", ma)
st.write("RSI:", rsi)

st.subheader("Данные")
st.dataframe(history)
