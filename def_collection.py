# def get_sma(ticker, window):
#     df = load_ohlcv(ticker)
#     return df['close'].rolling(window=window).mean().iloc[-1] if df is not None and not df.empty else 0

# def get_ema(ticker, window):
#     df = load_ohlcv(ticker)

#     if df is not None and not df.empty:
#         return df['close'].ewm(span=window, adjust=False).mean().iloc[-1]  # EMA 계산 후 마지막 값 반환
    
#     else:
#         return 0  # 데이터가 없으면 0 반환


# def get_rsi(ticker, period):
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
#     df_rsi = load_ohlcv(ticker)
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
#     delta = df_rsi['close'].diff(1)
#     gain = delta.where(delta > 0, 0).rolling(window=period).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs)).iloc[-1]
# time.sleep(1)  # API 호출 제한을 위한 대기

# def get_stoch_rsi(ticker, rsi_period, stoch_period):
#     # RSI 계산
#     rsi = get_rsi(ticker, rsi_period)
    
#     # RSI의 최근 n일 데이터 가져오기
#     df_rsi = load_ohlcv(ticker)
#     rsi_values = df_rsi['close'].rolling(window=rsi_period).apply(lambda x: get_rsi(x, rsi_period), raw=False)
    
#     # 스토캐스틱 RSI 계산
#     min_rsi = rsi_values.rolling(window=stoch_period).min()
#     max_rsi = rsi_values.rolling(window=stoch_period).max()
    
#     # stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi)
#     stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)  # 0으로 나누는 경우를 방지하기 위해 np.nan으로 대체
    
#     return stoch_rsi if not stoch_rsi.empty else 0