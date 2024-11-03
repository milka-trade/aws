import time
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
import pandas as pd

load_dotenv()

# # Upbit API 설정
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
# df=pyupbit.get_ohlcv("KRW-BTC")
# print(df)

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
df_tickers = {}    # 전역변수:일봉 데이터프레임

# def send_discord_message(msg):
#     """discord 메시지 전송"""
#     try:
#         message ={"content":msg}
#         requests.post(DISCORD_WEBHOOK_URL, data=message)
#     except Exception as e:
#         print(f"디스코드 메시지 전송 실패 : {e}")
#         time.sleep(5) 

def load_ohlcv(ticker):
    global df_tickers
    if ticker not in df_tickers:   # 티커가 캐시에 없으면 데이터 가져오기     
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60", count=200) 
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"load_ohlcv / No data returned for ticker: {ticker}")
                # send_discord_message(f"load_ohlcv / No data returned for ticker: {ticker}")
                time.sleep(0.5)  # API 호출 제한을 위한 대기

        except Exception as e:
            print(f"load_ohlcv / 디스코드 메시지 전송 실패 : {e}")
            # send_discord_message(f"load_ohlcv / Error loading data for ticker {ticker}: {e}")
            time.sleep(1)
    return df_tickers.get(ticker)

def get_ema(ticker, window):
    df = load_ohlcv(ticker)  # OHLCV 데이터 로드

    if df is not None and not df.empty:
        return df['close'].ewm(span=window, adjust=False).mean().iloc[-1]  # EMA 계산 후 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환
    
def get_rsi(ticker, period):
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]
time.sleep(1)  # API 호출 제한을 위한 대기

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

def get_rsi_and_stoch_rsi(ticker, rsi_period, stoch_period):
    # 데이터 로드
    # df_rsi = load_ohlcv(ticker)
    df_rsi = pyupbit.get_ohlcv(ticker, interval="minute60", count=200) 
    
    # RSI 계산
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 스토캐스틱 RSI 계산
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)  # 0으로 나누는 경우를 방지하기 위해 np.nan으로 대체
    
    # return rsi.iloc[-1], stoch_rsi.iloc[-1] if not stoch_rsi.empty else 0
    return stoch_rsi if not stoch_rsi.empty else 0

# API 호출 제한을 위한 대기
time.sleep(1)

# def calculate_heikin_ashi(ticker):
#     df = load_ohlcv(ticker)  # 데이터 로드
#     if df.empty:
#         raise ValueError(f"No data found for ticker: {ticker}")
    
#     # 필수 열이 존재하는지 확인
#     required_columns = ['open', 'high', 'low', 'close']
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Missing required column: {col}")
    
#     ha_df = df.copy()   # 데이터프레임의 복사본 생성 (원본에 영향 없음)

#     # 하이킨 아시 종가 계산
#     ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

#     # 첫 번째 HA_Open은 원본 시가로 설정
#     ha_df['HA_Open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
#     # ha_df.loc[0, 'HA_Open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2  # 첫 번째 행의 HA_Open은 초기값으로 설정


#     # HA_Open의 나머지 값 계산
#     for i in range(1, len(ha_df)):
#         # 이전 HA_Open과 HA_Close를 기반으로 현재 HA_Open 계산
#         ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2

#     # HA_High 및 HA_Low 계산
#     ha_df['HA_High'] = ha_df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
#     ha_df['HA_Low'] = ha_df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

#     # 디버깅: 중간 결과 확인
#     print(ha_df.head())  # 중간 결과 출력

#     return ha_df

def calculate_ha_candles(ticker):
    df = load_ohlcv(ticker)  # 데이터 로드
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # 하이킨아시 캔들 초기화
    ha_df = pd.DataFrame(index=df.index)

    # 첫 번째 하이킨아시 캔들 시가, 종가는 일반 캔들과 동일
    ha_df.loc[0, 'HA Close'] = (df['open'].iloc[0] + df['high'].iloc[0] + df['low'].iloc[0] + df['close'].iloc[0]) / 4
    ha_df.loc[0, 'HA Open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    ha_df.loc[0, 'HA High'] = df['high'].iloc[0]
    ha_df.loc[0, 'HA Low'] = df['low'].iloc[0]

    # 나머지 하이킨아시 캔들 계산
    for i in range(1, len(df)):
        ha_df.loc[i, 'HA Open'] = (ha_df.loc[i - 1, 'HA Open'] + ha_df.loc[i - 1, 'HA Close']) / 2
        ha_df.loc[i, 'HA Close'] = (df['open'].iloc[i] + df['high'].iloc[i] + df['low'].iloc[i] + df['close'].iloc[i]) / 4
        ha_df.loc[i, 'HA High'] = max(df['high'].iloc[i], ha_df.loc[i, 'HA Open'], ha_df.loc[i, 'HA Close'])
        ha_df.loc[i, 'HA Low'] = min(df['low'].iloc[i], ha_df.loc[i, 'HA Open'], ha_df.loc[i, 'HA Close'])

    return ha_df

def get_current_price(ticker):
    """현재가를 조회합니다."""
    if not ticker.startswith("KRW-"):
        print(f"current_price/잘못된 티커 형식: {ticker}")
        # send_discord_message(f"current_price/잘못된 티커 형식: {ticker}")
        return None
    
    try:
        orderbook = pyupbit.get_orderbook(ticker=ticker)
        if orderbook is None or "orderbook_units" not in orderbook or not orderbook["orderbook_units"]:
            raise ValueError(f"'{ticker}'에 대한 유효한 orderbook이 없습니다.")
        current_price = orderbook["orderbook_units"][0]["ask_price"]
        time.sleep(0.5)
        return current_price
    
    except Exception as e:
        print(f"current_price/현재가 조회 오류 ({ticker}): {e}")
        # send_discord_message(f"current_price/현재가 조회 오류 ({ticker}): {e}")
        time.sleep(1)
        return None

# def filtered_tickers(tickers, held_coins):
#     """특정 조건에 맞는 티커 필터링"""
#     filtered_tickers = []
#     threshold_value = get_dynamic_threshold(tickers)

tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회

for t in tickers:
    currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        # if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
        #     continue

df = load_ohlcv(t) 
            # if df is None or df.empty:
            #     print(f"filtered_tickers/No data for ticker: {t}")
            #     send_discord_message(f"filtered_tickers/No data for ticker: {t}")
            #     continue

# df_day = pyupbit.get_ohlcv(t, interval="day", count=7)  
            
cur_price = get_current_price(t)
            

            # New Indicators and Patterns
# rsi = get_rsi(t, 14)
# print(f"rsi: {t} / {rsi}")
            # New Indicators and Patterns #2
            # ha_df = calculate_heikin_ashi(t)   #하이킨 아시 캔들 계산
            # if ha_df.empty or 'HA_Close' not in ha_df.columns:
            #     raise ValueError("ha_df:Heikin Ashi DataFrame is empty or HA_Close column is missing.")
            
# ema200 = get_ema(t, 200)    #200봉 가중이동평균 계산
# print(f"ema200: {t} / {ema200}")
stoch_rsi = get_rsi_and_stoch_rsi('KRW-BTC', 14, 14)  #스토캐스틱 RSI 계산
stoch_rsi_1 = stoch_rsi.iloc[-1]   #스토캐스틱 RSI 계산
stoch_rsi_2 = stoch_rsi.iloc[-2]   #스토캐스틱 RSI 계산
# print(f"stoch_rsi : {t} / {stoch_rsi}")
# print(f"stoch_rsi : {t} / {stoch_rsi_1}")
# print(f"stoch_rsi : {t} / {stoch_rsi_2}")

ha_candles = calculate_ha_candles('krw-btc')
print(ha_candles)

if stoch_rsi.empty or len(stoch_rsi) < 2:
            raise ValueError("stoch_rsi : Stochastic RSI DataFrame is empty or has insufficient data.")

if not stoch_rsi.empty and len(stoch_rsi) >= 2:
                last_stoch_rsi = stoch_rsi.iloc[-1]
                previous_stoch_rsi = stoch_rsi.iloc[-2]
else:
                raise ValueError("stoch_rsi : Stochastic RSI data is insufficient.")

last_stoch_rsi = stoch_rsi.iloc[-1]
previous_stoch_rsi = stoch_rsi.iloc[-2]

            # 하이킨 아시 캔들의 종가
            # last_ha_close = ha_df['HA_Close'].iloc[-1]
            # print(f"Last HA Close: {last_ha_close}")  # 마지막 HA_Close 값 출력

            # if day_value_1 > 25_000_000_000 or day_value_2 > 20_000_000_000 :    
                # print(f"cond1: {t} / 당일 거래량 > 10십억 or 전일 거래량 > 25십억")

            # if threshold_value < atr :  # Volatility check
            #     print(f"cond2: {t} / 임계값:{threshold_value:,.2f} < 평균진폭:{atr:,.2f}")
            
            # print(f"cond2-1: {t} / price:{cur_price:,.2f} > ema200:{ema200:,.2f}")
            # if cur_price > ema200 :
            #         print(f"cond2-1: {t} / price:{cur_price:,.2f} > ema200:{ema200:,.2f}")

                    # print(f"cond2-2: {t} / 스토캐스틱RSI1:{previous_stoch_rsi:,.2f} > RSI2:{last_stoch_rsi:,.2f}")
                    # if previous_stoch_rsi <= 0.20 and last_stoch_rsi > 0.20:
                    #         print(f"cond2-2: {t} / 스토캐스틱RSI1:{previous_stoch_rsi:,.2f} > RSI2:{last_stoch_rsi:,.2f}")

                    # if ma20 < ma5 < cur_price:  # Short-term momentum
                        # print(f"cond3: {t} / ma20 < ma5")

                        # print(f"cond4: {t} / rsi{rsi:,.2f} < 60 / ma50:{ma50:,.2f} < price:{cur_price:,.2f}")
                    # if rsi < 30 :  # RSI in a favorable range
                    #         print(f"cond4: {t} / rsi:{rsi:,.2f} < 30")

                    #         if cur_price < day_open_price_1 * 1.03:
                    #             # print(f"cond5: {t} / 5% 이내 상승")
                        
                    #             if cur_price < df_open_1*1.02 :    
                    #                 # print(f"cond6: {t} / 분봉 3프로 이내 상승")
                             
                    #                 ai_decision = get_ai_decision(t)  
                    #                 send_discord_message(f"{t} / AI: {ai_decision}")
                    #                 if ai_decision == "BUY" :
                    #                     filtered_tickers.append(t)
            
    #     except Exception as e:
    #         send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
    #         time.sleep(5)  # API 호출 제한을 위한 대기

    # return filtered_tickers