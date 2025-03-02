import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

interval_into_freq = {"1m": "1T", "2m": "2T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H", "1d": "1D", "1wk": "1W", "1mo": "1M"}
future_data = []
def extend_dates_for_ML(data, interval, entry_periods):
    last_date = data.index[-1]
    if interval in ["1m", "2m", "5m", "15m", "30m"]:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=int(interval[:-1])), periods=entry_periods,
                                     freq=interval_into_freq[interval])
    elif interval == "1d":
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=entry_periods,
                                     freq=interval_into_freq[interval])
    else:
        future_dates = pd.date_range(start=last_date, periods=entry_periods, freq=interval_into_freq[interval])

    future_data = pd.DataFrame(index=future_dates)

    return future_data


def indicators_switch_case(data, indicators, interval, entry_periods, future_data):
    addplot_indicators = []
    if indicators["bullish_bearish"]:
        addplot_indicators += bullish_bearish(data)

    if indicators["fibonacci_retracement"]:
        addplot_indicators += fibonacci_retracement(data)
    if indicators["bollinger_bands"]["show"] and indicators["bollinger_bands"]["window"]:  # error can occure if window less then len(data) (ValueError: zero-size array to reduction operation maximum which has no identity)
        addplot_indicators += bollinger_bands(data, indicators["bollinger_bands"]["window"])

    if indicators["moving_averages"]["show"] and indicators["moving_averages"]["window"]:  # error can occure if window less then len(data) (ValueError: zero-size array to reduction operation maximum which has no identity)
        addplot_indicators += moving_averages(data, indicators["moving_averages"]["window"])

    if indicators["relative_strength"]["show"] and indicators["relative_strength"]["window"]:  # error can occure if window less then len(data) (ValueError: zero-size array to reduction operation maximum which has no identity)
        addplot_indicators += relative_strength(data, indicators["relative_strength"]["window"], future_data)[1]

    if indicators["forest_regression"]["show"]:
        addplot_indicators += predict_rsi(data, interval, entry_periods)
    return addplot_indicators

def generate_plot(entry_ticker, entry_start, entry_end, interval, indicators, entry_periods):
    entry_periods = int(entry_periods)

    data = yf.download(entry_ticker, start=entry_start, end=entry_end, interval=interval)
    data.dropna(inplace=True)

    future_data = extend_dates_for_ML(data, interval, entry_periods)
    addplot_indicators = indicators_switch_case(data, indicators, interval, entry_periods, future_data)

    data_extended = pd.concat([data, future_data])

    column_names = ["Close", "High", "Low", "Open", "Volume"]
    data.columns = column_names
    data_extended.columns = column_names

    # Just for easier debugging
    data_extended.to_csv("stock_data_extended.csv", index=True)

    mpf.plot(data_extended, type='candle', style='yahoo', volume=True, addplot=addplot_indicators)

def bullish_bearish(data):
    data['bull_egf'] = (data['Open'] < data['Close']) & (data['Open'].shift(1) > data['Close'].shift(1)) & (data['Open'] < data['Close'].shift(1)) & (data['Close'] > data['Open'].shift(1))
    data['bear_egf'] = (data['Open'] > data['Close']) & (data['Open'].shift(1) < data['Close'].shift(1)) & (data['Open'] > data['Close'].shift(1)) & (data['Close'] < data['Open'].shift(1))

    # Додавання стовпців для маркерів
    data['marker_bull_egf'] = None
    data['marker_bear_egf'] = None

    # Маркування позицій
    for i in range(1, len(data)):
        if data['bull_egf'].iloc[i]:
            data['marker_bull_egf'].iloc[i] = data['Low'].iloc[i]
        if data['bear_egf'].iloc[i]:
            data['marker_bear_egf'].iloc[i] = data['High'].iloc[i]
    data.to_csv("stock_data_bullish_bearish.csv", index=True)

    # Create marker columns with NaNs everywhere except for bullish or bearish signals
    bullish_marker = pd.Series([data['marker_bull_egf'].iloc[i] if data['bull_egf'].iloc[i] else None for i in range(len(data))], index=data.index)
    bearish_marker = pd.Series([data['marker_bear_egf'].iloc[i] if data['bear_egf'].iloc[i] else None for i in range(len(data))], index=data.index)

    # Create the list of addplot objects for markers (only on true patterns)
    addplots = []

    # Add markers for bullish engulfing candles (up triangles)
    if bullish_marker.dropna().any():
        addplots.append(mpf.make_addplot(bullish_marker, type='scatter', markersize=100, marker='^', color='g', panel=0))

    # Add markers for bearish engulfing candles (down triangles)
    if bearish_marker.dropna().any():
        addplots.append(mpf.make_addplot(bearish_marker, type='scatter', markersize=100, marker='v', color='r', panel=0))
    return addplots

def fibonacci_retracement(data):
    max_price = data['High'].max()
    min_price = data['Low'].min()

    # Обчислення рівнів
    diff = max_price - min_price
    levels = {
        "0.236": max_price - diff * 0.236,
        "0.382": max_price - diff * 0.382,
        "0.5": max_price - diff * 0.5,
        "0.618": max_price - diff * 0.618,
        "0.786": max_price - diff * 0.786
    }

    plot_fib = []

    for level, price in levels.items():
        plot_fib.append(
            mpf.make_addplot([price] * len(data), color='blue', linestyle='dashed', width=0.8, label=f"Fib {level}"))

    return plot_fib

def bollinger_bands(data, window):
    window = int(window)
    data['Middle'] = data['Close'].rolling(window=window).mean()  # Ковзне середнє
    data['Std'] = data['Close'].rolling(window=window).std()  # Стандартне відхилення
    data['Upper'] = data['Middle'] + (2 * data['Std'])  # Верхня межа
    data['Lower'] = data['Middle'] - (2 * data['Std'])  # Нижня межа

    # Додаємо Bollinger Bands на графік
    ap_bb = [
        mpf.make_addplot(data['Upper'], color="red", label='Upper Bollinger Band'),
        mpf.make_addplot(data['Middle'], color="gray", label='Middle Bollinger Band'),
        mpf.make_addplot(data['Lower'], color="green", label='Lower Bollinger Band')
    ]
    return ap_bb

def moving_averages(data, window):
    window = int(window)
    data['SMA'] = data['Close'].rolling(window=window).mean()

    # EMA (Експоненціально згладжене середнє)
    data['EMA'] = data['Close'].ewm(span=window, adjust=False).mean()

    # WMA (Зважене ковзне середнє)
    weights = range(1, window + 1)
    data['WMA'] = data['Close'].rolling(window=window).apply(lambda prices: sum(prices * weights) / sum(weights), raw=True)

    # Додаємо ці ковзні середні до графіку
    addplots = [
        mpf.make_addplot(data['SMA'], color='blue', width=1, panel=0, label="Simple Moving Average"),
        mpf.make_addplot(data['EMA'], color='orange', width=1, panel=0, label="Exponential Moving Average"),
        mpf.make_addplot(data['WMA'], color='purple', width=1, panel=0, label="Weighted Moving Average")
    ]

    return addplots

def relative_strength(data, window, future_data = None):
    window = int(window)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Додавання рівнів для покупок та продажів
    rsi_30 = pd.Series(30, index=rsi.index)  # рівень перепродажу
    rsi_70 = pd.Series(70, index=rsi.index)  # рівень перекупленості

    # Створення графіка RSI
    plot_rsi = []
    plot_rsi.append(mpf.make_addplot(pd.concat([rsi, future_data]), panel=1, color='purple', label="RSI"))
    plot_rsi.append(mpf.make_addplot(pd.concat([rsi_30, future_data]), panel=1, color='green', linestyle='dashed', width=0.8, label="Buy Level (30)"))
    plot_rsi.append(mpf.make_addplot(pd.concat([rsi_70, future_data]), panel=1, color='red', linestyle='dashed', width=0.8, label="Sell Level (70)"))
    return rsi, plot_rsi

def predict_rsi(data, interval, entry_periods, window=14, past_days=10):
    rsi_values = relative_strength(data, window)[0]
    print("yoy6")
    # Створюємо ознаки: беремо попередні `past_days` значень RSI для прогнозу
    X = np.array([rsi_values.shift(i) for i in range(1, past_days + 1)]).T
    y = rsi_values.shift(-1)  # Прогнозуємо наступне значення RSI
    print(X)
    print(y)
    # Видаляємо NaN-значення
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    print("yoy4")
    # Розділяємо дані на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print("yoy3")
    # Навчаємо модель
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("yoy2")
    # Передбачаємо RSI на `future_steps` днів уперед
    future_rsi = []
    last_values = X[-1].tolist()
    print("yoy1")
    for _ in range(entry_periods):
        next_rsi = model.predict(np.array(last_values).reshape(1, -1))[0]
        future_rsi.append(next_rsi)

        # Оновлюємо останні значення для наступного прогнозу
        last_values.pop(0)
        last_values.append(next_rsi)
    print("yoy")
    # Додаємо прогнозовані значення в DataFrame
    future_dates = pd.date_range(start=data.index[-1], periods=entry_periods + 1, freq=interval_into_freq[interval])[1:]
    future_df = pd.DataFrame({'Close': future_rsi}, index=future_dates)

    all_dates_df = pd.concat([rsi_values, future_df])
    print(all_dates_df)
    plot_rsi = []
    plot_rsi.append(
        mpf.make_addplot(all_dates_df, panel=1, color='blue', linestyle='dotted', width=1.2, label="Predicted RSI"))
    print(plot_rsi)
    return plot_rsi