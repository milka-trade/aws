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

df_tickers = {} 

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

def get_wma(ticker, window):
    df = load_ohlcv(ticker)

    if df is not None and not df.empty:
        # WMA 계산
        weights = range(1, window + 1)
        wma = df['close'].rolling(window=window).apply(lambda x: sum(weights * x) / sum(weights), raw=True)
        return wma  # WMA의 마지막 값 반환
    else:
        return 0  # 데이터가 없으면 0 반환
    
last_wma200 = get_wma('KRW-BTC', 200).iloc[-1]    #200봉 가중이동평균 계산
print(last_wma200)
pre_wma200 = get_wma('krw-btc', 200).iloc[-2]