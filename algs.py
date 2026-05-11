import numpy as np

def ddm_fair_price(market_price, eps, div_yield, roe, risk_free=0.16, premium=0.03):
    """
    Справедливая цена одной акции по модели дисконтирования дивидендов.
    
    Параметры:
        market_price : float — текущая рыночная цена
        eps          : float — прибыль на акцию (TTM)
        div_yield    : float — дивидендная доходность (%)
        roe          : float — рентабельность капитала (%)
        risk_free    : float — безрисковая ставка (ключевая ЦБ)
        premium      : float — премия за риск
    
    Возвращает:
        dict: fair_price, upside_pct, signal
    """
    if eps <= 0 or div_yield <= 0:
        return {
            'fair_price': None,
            'upside_pct': None,
            'signal': 'Н/Д (убыток или нет дивидендов)'
        }
    
    dps0 = market_price * (div_yield / 100)           # DPS текущий
    payout = dps0 / eps                                # доля прибыли на дивиденды
    g = np.clip((roe / 100) * (1 - payout), 0.01, 0.15)  # темп роста
    R = risk_free + premium                            # требуемая доходность
    dps1 = dps0 * (1 + g)                              # DPS следующий
    fair = dps1 / (R - g)                              # справедливая цена
    upside = (fair / market_price - 1) * 100
    
    signal = 'Недооценена' if upside > 0 else 'Переоценена'
    
    return {
        'fair_price': round(fair, 2),
        'upside_pct': round(upside, 2),
        'growth_rate_pct': round(g * 100, 2),
        'required_return_pct': round(R * 100, 2),
        'signal': signal
    }


def graham_number(eps, bvps, market_price, pe=None, pb=None):
    """
    Число Грэма для одной акции.
    
    Параметры:
        eps          : float — прибыль на акцию
        bvps         : float — балансовая стоимость на акцию
        market_price : float — текущая рыночная цена
        pe           : float или None — P/E (необязательно, для фильтра Грэма)
        pb           : float или None — P/B (необязательно, для фильтра Грэма)
    
    Возвращает:
        dict: graham_number, margin_pct, signal
    """
    if eps <= 0:
        return {
            'graham_number': None,
            'margin_pct': None,
            'signal': 'Неприменимо (EPS ≤ 0)'
        }
    
    gn = np.sqrt(22.5 * eps * bvps)
    margin = (gn - market_price) / gn * 100
    
    if pe is not None and pb is not None and (pe * pb) > 22.5:
        signal = f'Не проходит фильтр (P/E×P/B = {pe*pb:.1f} > 22.5)'
    elif margin > 30:
        signal = 'Отличный запас прочности'
    elif margin > 0:
        signal = 'Недооценена'
    else:
        signal = 'Переоценена'
    
    return {
        'graham_number': round(gn, 2),
        'margin_pct': round(margin, 2),
        'signal': signal
    }

def multiplier_score(pe, pb, roe, de, div_yield, pe_min, pe_max, pb_min, pb_max,
                     roe_min, roe_max, de_min, de_max, div_min, div_max,
                     weights=None):
    """
    Скоринговый балл для одной акции на основе мультипликаторов.
    Нужны min/max по каждому показателю среди всей группы для нормировки.
    
    Параметры:
        pe, pb, roe, de, div_yield : float — показатели акции
        pe_min, pe_max, ...         : float — min/max каждого показателя по всем акциям
        weights                     : dict — веса {'pe': 0.20, ...}
    
    Возвращает:
        dict: total_score, rank (нужно посчитать отдельно после всех)
    """
    if weights is None:
        weights = {'pe': 0.20, 'pb': 0.15, 'roe': 0.25, 'de': 0.15, 'div_yield': 0.25}
    
    if pe is None:
        return {'total_score': None, 'signal': 'Н/Д (убыток)'}
    
    def score(value, vmin, vmax, reverse=False):
        """Нормировка в 0-100. reverse=True: чем ниже value, тем выше балл"""
        if vmax == vmin:
            return 50.0
        if reverse:
            return (vmax - value) / (vmax - vmin) * 100
        return (value - vmin) / (vmax - vmin) * 100
    
    s_pe = score(pe, pe_min, pe_max, reverse=True)
    s_pb = score(pb, pb_min, pb_max, reverse=True)
    s_roe = score(roe, roe_min, roe_max, reverse=False)
    s_de = score(de, de_min, de_max, reverse=True)
    s_div = score(div_yield, div_min, div_max, reverse=False)
    
    total = (
        s_pe * weights['pe'] +
        s_pb * weights['pb'] +
        s_roe * weights['roe'] +
        s_de * weights['de'] +
        s_div * weights['div_yield']
    )
    
    return {
        'total_score': round(total, 2),
        'score_pe': round(s_pe, 2),
        'score_pb': round(s_pb, 2),
        'score_roe': round(s_roe, 2),
        'score_de': round(s_de, 2),
        'score_div': round(s_div, 2)
    }

def market_metrics(prices, volumes):
    """
    Метрики доходности и риска для одной акции.
    
    Параметры:
        prices  : list/array — цены закрытия по дням
        volumes : list/array — объемы торгов по дням
    
    Возвращает:
        dict: total_return, annual_vol, max_drawdown, avg_volume, sharpe
    """
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    daily_ret = np.diff(prices) / prices[:-1]
    total_ret = (prices[-1] / prices[0] - 1) * 100
    annual_vol = np.std(daily_ret) * np.sqrt(250)
    avg_volume = np.mean(volumes)
    
    cumulative = np.cumprod(1 + daily_ret)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) * 100
    
    sharpe = (np.mean(daily_ret) * 250 - 0.16) / annual_vol if annual_vol > 0 else 0
    
    return {
        'total_return_pct': round(total_ret, 2),
        'annual_volatility_pct': round(annual_vol * 100, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_volume': round(avg_volume, 0),
        'sharpe_ratio': round(sharpe, 2)
    }