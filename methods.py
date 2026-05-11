import requests
import numpy as np
import pandas as pd


def load_moex_history(url, date_from, date_to):
    all_rows = []
    columns = []
    start = 0

    while True:
        params = {
            "from": date_from,
            "till": date_to,
            "start": start
        }

        response = requests.get(url, params=params)
        data = response.json()

        columns = data["history"]["columns"]
        rows = data["history"]["data"]

        if not rows:
            break

        all_rows.extend(rows)
        start += len(rows)

    return pd.DataFrame(all_rows, columns=columns)


def get_history(ticker, date_from="2026-01-01", date_to="2026-05-01"):
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"

    history = load_moex_history(url, date_from, date_to)

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
        "CURRENCYID",   # валюта торгов
        "TRENDCLSPR"   # изменение цены закрытия в процентах
    ]]

    return history

def get_imoex(date_from="2026-01-01", date_to="2026-05-01"):
    url = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"

    imoex = load_moex_history(url, date_from, date_to)

    imoex = imoex[[
        "TRADEDATE",
        "SECID",
        "CLOSE"
    ]]

    return imoex


def market_metrics (prices, volumes):
    returns = calc_returns(prices)
    volatility = calc_volatility(prices)
    sharpe = calc_sharpe(prices)
    drawdown = calc_max_drawdown(prices)

    return {
        "total_return_pct": returns["total_return_pct"],
        "annual_volatility_pct": volatility["annual_volatility_pct"],
        "max_drawdown_pct": drawdown["max_drawdown_pct"],
        "avg_volume": np.mean(volumes),
        "sharpe_ratio": sharpe["sharpe_ratio"]
    }
    


def calc_returns(prices):
    """
    Вход:  prices — список/массив цен закрытия
    Выход: дневная, недельная, месячная, общая доходность
    """
    prices = np.array(prices)
    
    daily = np.diff(prices) / prices[:-1]
    total = (prices[-1] / prices[0] - 1) * 100
    
    # Недельная (5 дней)
    if len(prices) >= 5:
        weekly = (prices[4:] / prices[:-4] - 1) * 100
    else:
        weekly = None
    
    return {
        'daily_mean_pct': np.mean(daily) * 100,
        'total_return_pct': total,
        'daily_returns': daily
    }

def calc_volatility(prices):
    """
    Вход:  prices — список цен закрытия
    Выход: дневная и годовая волатильность
    """
    prices = np.array(prices)
    daily_ret = np.diff(prices) / prices[:-1]
    
    daily_vol = np.std(daily_ret)
    annual_vol = daily_vol * np.sqrt(250)
    
    return {
        'daily_volatility_pct': daily_vol * 100,
        'annual_volatility_pct': annual_vol * 100
    }

def calc_sharpe(prices, risk_free=0.16):
    """
    Вход:  prices — цены, risk_free — безрисковая ставка
    Выход: коэффициент Шарпа
    """
    prices = np.array(prices)
    daily_ret = np.diff(prices) / prices[:-1]
    
    mean_daily = np.mean(daily_ret)
    std_daily = np.std(daily_ret)
    
    if std_daily == 0:
        return {'sharpe': 0}
    
    sharpe = (mean_daily * 250 - risk_free) / (std_daily * np.sqrt(250))
    
    return {'sharpe_ratio': sharpe}

def calc_max_drawdown(prices):
    """
    Вход:  prices — цены
    Выход: max_drawdown_pct, даты начала и конца
    """
    prices = np.array(prices)
    daily_ret = np.diff(prices) / prices[:-1]
    
    cumulative = np.cumprod(1 + daily_ret)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) * 100
    
    # Индекс максимальной просадки
    dd_end = np.argmin(drawdowns) + 1
    dd_start = np.argmax(cumulative[:dd_end])
    
    return {
        'max_drawdown_pct': max_dd,
        'dd_start_idx': dd_start,
        'dd_end_idx': dd_end
    }

def calc_beta(stock_prices, market_prices):
    """
    Вход:  stock_prices — цены акции
           market_prices — цены индекса (например, IMOEX)
    Выход: Бета — чувствительность акции к движению рынка.
    """
    stock_ret = np.diff(stock_prices) / stock_prices[:-1]
    market_ret = np.diff(market_prices) / market_prices[:-1]
    
    # Минимальная длина
    n = min(len(stock_ret), len(market_ret))
    stock_ret = stock_ret[:n]
    market_ret = market_ret[:n]
    
    cov = np.cov(stock_ret, market_ret)[0, 1]
    var = np.var(market_ret)
    
    beta = cov / var if var != 0 else None
    
    return {'beta': beta}

def calc_moving_averages(prices, short_window=20, long_window=50):
    """
    Вход:  prices — цены
           short_window — короткое окно (дни)
           long_window — длинное окно (дни)
    Выход: MA_short, MA_long, сигнал
    """
    prices = np.array(prices)
    
    if len(prices) < long_window:
        return {'signal': 'Недостаточно данных'}
    
    ma_short = np.mean(prices[-short_window:])
    ma_long = np.mean(prices[-long_window:])
    
    if ma_short > ma_long:
        signal = 'Бычий (восходящий тренд)'
    elif ma_short < ma_long:
        signal = 'Медвежий (нисходящий тренд)'
    else:
        signal = 'Нейтральный'
    
    return {
        'ma_short': ma_short,
        'ma_long': ma_long,
        'signal': signal
    }

def calc_rsi(prices, period):
    """
    Вход:  prices — цены, period — период
    Выход: RSI — перекупленность / перепроданность., сигнал
    """
    prices = np.array(prices)
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    if len(gains) < period:
        return {'rsi': None, 'signal': 'Недостаточно данных'}
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    if rsi > 70:
        signal = 'Перекуплен (сигнал к продаже)'
    elif rsi < 30:
        signal = 'Перепродан (сигнал к покупке)'
    else:
        signal = 'Нейтрально'
    
    return {'rsi': round(rsi, 2), 'signal': signal}

