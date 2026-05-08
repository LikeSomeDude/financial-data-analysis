import requests
import pandas as pd

def get_history(ticker, date_from="2025-01-01", date_to="2025-12-31"):
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"

    params = {
        "from": date_from,
        "till": date_to
    }

    response = requests.get(url, params=params)
    data = response.json()

    columns = data["history"]["columns"]
    rows = data["history"]["data"]

    history = pd.DataFrame(rows, columns=columns)

    history = history[history["BOARDID"] == "TQBR"]

    history = history[[
        "TRADEDATE",   # дата
        "SECID",       # код акции
        "SHORTNAME",   # название
        "OPEN",        # цена открытия
        "LOW",         # минимум за день
        "HIGH",        # максимум за день
        "CLOSE",       # цена закрытия
        "VOLUME",      # объем в штуках
        "VALUE",       # оборот в деньгах
        "NUMTRADES",   # количество сделок
        "TRENDCLSPR"   # изменение цены закрытия в процентах
    ]]

    return history


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