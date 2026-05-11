import requests
import pandas as pd
from methods import get_history, get_imoex, market_metrics, calc_beta, calc_moving_averages, calc_rsi
import streamlit as st


tickers = ["AFLT", "SVCB", "GAZP", "MGNT", "MVID", "SBER", "YDEX"]

history = get_history(tickers[0])

print(history.head())
print(history.columns)


all_history = []

for ticker in tickers:
    ticker_history = get_history(ticker)
    all_history.append(ticker_history)

all_history = pd.concat(all_history, ignore_index=True)

print(all_history.head())
print(all_history["SECID"].unique())

result = []

for ticker in tickers:
    ticker_data = all_history[all_history["SECID"] == ticker]
    ticker_data = ticker_data.sort_values("TRADEDATE")

    metrics = market_metrics(
        prices = ticker_data["CLOSE"].tolist(),
        volumes = ticker_data["VOLUME"].tolist()
    )

    metrics["SECID"] = ticker
    result.append(metrics)

summary = pd.DataFrame(result)

print("\nРыночные показатели акций за выбранный период:")
print(summary)

print("\nЧто означают показатели:")
print("total_return_pct — общая доходность акции за период, в процентах.")
print("annual_volatility_pct — годовая волатильность: чем выше, тем сильнее колебания и выше риск.")
print("max_drawdown_pct — максимальная просадка: самое сильное падение от локального максимума.")
print("avg_volume — средний дневной объем торгов, показывает ликвидность акции.")
print("sharpe_ratio — соотношение доходности и риска: чем выше, тем лучше.")