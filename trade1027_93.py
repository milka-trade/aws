import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import requests
import pandas as pd

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
df_tickers = {}    # 전역변수:일봉 데이터프레임

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

def load_ohlcv(ticker):
    global df_tickers
    if ticker not in df_tickers:   # 티커가 캐시에 없으면 데이터 가져오기     
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60", count=100) 
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"load_ohlcv / No data returned for ticker: {ticker}")
                send_discord_message(f"load_ohlcv / No data returned for ticker: {ticker}")
                time.sleep(0.5)  # API 호출 제한을 위한 대기
        except Exception as e:
            print(f"load_ohlcv / 디스코드 메시지 전송 실패 : {e}")
            send_discord_message(f"load_ohlcv / Error loading data for ticker {ticker}: {e}")
            time.sleep(1)
    return df_tickers.get(ticker)

def get_balance(ticker):
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == ticker:
                time.sleep(0.5)
                return float(b['balance']) if b['balance'] is not None else 0
    except Exception as e:
        print(f"get_balance/잔고 조회 오류: {e}")
        send_discord_message(f"get_balance/잔고 조회 오류: {e}")
        time.sleep(1)
        return 0
    return 0

def get_current_price(ticker):
    """현재가를 조회합니다."""
    if not ticker.startswith("KRW-"):
        print(f"current_price/잘못된 티커 형식: {ticker}")
        send_discord_message(f"current_price/잘못된 티커 형식: {ticker}")
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
        send_discord_message(f"current_price/현재가 조회 오류 ({ticker}): {e}")
        time.sleep(1)
        return None

def get_sma(ticker, window):
    df = load_ohlcv(ticker)
    return df['close'].rolling(window=window).mean().iloc[-1] if df is not None and not df.empty else 0

def get_best_k(ticker="KRW-BTC"):
    bestK = 0.5  # 초기 K 값
    interest = 0  # 초기 수익률
    df = load_ohlcv(ticker)  # 데이터 로드
    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환
    for k in np.arange(0.1, 0.5, 0.05):  # K 값을 0.1부터 0.5까지 반복
        df['range'] = (df['high'] - df['low']) * k      #변동성 계산
        df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정
        fee = 0.0005  # 거래 수수료 (0.05%로 설정)
        df['ror'] = (df['close'] / df['target'] - fee).where(df['high'] > df['target'], 1)
        ror_series = df['ror'].cumprod()  # 누적 수익률
        if len(ror_series) < 2:  # 데이터가 부족한 경우
            continue
        ror = ror_series.iloc[-2]   # 마지막 이전 값
        if ror > interest:  # 이전 수익률보다 높으면 업데이트
            interest = ror
            bestK = k
            time.sleep(1)  # API 호출 제한을 위한 대기

    return bestK

def get_rsi(ticker, period):
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]
time.sleep(1)  # API 호출 제한을 위한 대기

def get_atr(ticker, period):
    try:
        # df_atr_day = pyupbit.get_ohlcv(ticker, interval="day", count=15)
        df_atr_day = load_ohlcv(ticker)
        time.sleep(0.5)  # API 호출 제한을 위한 대기
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    if df_atr_day is None or df_atr_day.empty:
        print(f"get_atr/ Error: No data for {ticker}")
        return None  # 또는 기본값을 반환할 수 있음
    
    high_low = df_atr_day['high'] - df_atr_day['low']
    high_close = abs(df_atr_day['high'] - df_atr_day['close'].shift().fillna(0))
    low_close = abs(df_atr_day['low'] - df_atr_day['close'].shift().fillna(0))
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=period).mean()

    if atr.empty or atr.iloc[-1] is None:
        print(f"Error: No ATR data for {ticker}")
        return None  # 또는 기본값을 반환할 수 있음
    
    return atr.iloc[-1]

def get_dynamic_threshold(tickers):
    atr_values = []
    for t in tickers:
        try:
            atr = get_atr(t, 21)
            # print(f"ATR for {t}: {atr}")
            if atr is not None and not np.isnan(atr):  # ATR이 None이나 NaN이 아닐 경우에만 추가
                atr_values.append(atr)
        except Exception as e:
            print(f"Error getting ATR for {t}: {e}")
            continue

    # NaN 값 필터링 후 중앙값 계산
    atr_values = [value for value in atr_values if not np.isnan(value)]

    return np.median(atr_values) if atr_values else 0.05  # Fallback to 0.05 if no ATR values

def filtered_tickers(tickers, held_coins):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    threshold_value = get_dynamic_threshold(tickers)

    for t in tickers:
        currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
            continue

        try:
            df = load_ohlcv(t) 
            if df is None or df.empty:
                print(f"filtered_tickers/No data for ticker: {t}")
                send_discord_message(f"filtered_tickers/No data for ticker: {t}")
                continue

            df_day = pyupbit.get_ohlcv(t, interval="day", count=7)  
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue  
            
            cur_price = get_current_price(t)
            
            if len(df_day) >= 3:
                day_value_1 = df_day['value'].iloc[-1]      #일봉 9시 기준 당일 거래량
                day_value_2 = df_day['value'].iloc[-2]      #일봉 9시 기준 전일 거래량 
            else:
                continue
          
            day_open_price_1 = df_day['open'].iloc[-1]  #9시 기준 당일 시가 

            df_open_1 = df['open'].iloc[-1]

            # New Indicators and Patterns
            rsi = get_rsi(t, 21)
            ma20 = get_sma(t, 20)
            ma5 = get_sma(t, 5)
            ma50 = get_sma(t, 50)
            avg_week_value = df_day['value'].rolling(window=7).mean().iloc[-1]
            min60_value = df['value'].iloc[-2]
            atr = get_atr(t, 21)

            if day_value_1 > 5_000_000_000 or day_value_2 > 15_000_000_000 :    
                # print(f"cond1: {t} / 당일 거래량 > 10십억 or 전일 거래량 > 25십억")

                if threshold_value < atr :  # Volatility check
                    # print(f"cond2: {t} / 임계값:{threshold_value:,.2f} < 평균진폭:{atr:,.2f}")

                    if ma20 < ma5 :  # Short-term momentum
                        print(f"cond3: {t} / ma20 < ma5")

                        # print(f"cond4: {t} / rsi{rsi:,.2f} < 60 / ma50:{ma50:,.2f} < price:{cur_price:,.2f}")
                        if rsi < 70 and ma50 < cur_price : # RSI in a favorable range
                            print(f"cond4: {t} / rsi:{rsi:,.2f} < 70 / ma50:{ma50:,.2f} < price:{cur_price:,.2f}")

                            if cur_price < day_open_price_1 * 1.1:
                                print(f"cond5: {t} / 10% 이내 상승")
                        
                                if cur_price < df_open_1*1.03 :    
                                    print(f"cond6: {t} / 분봉 3프로 이내 상승")
                                        
                                    filtered_tickers.append(t)
            
        except Exception as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

    return filtered_tickers

def get_best_ticker():  
    
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_list = filtered_tickers(tickers, held_coins)
    send_discord_message(f"{filtered_list}") 
    print(f"{filtered_list}")       
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.5  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        df = load_ohlcv(ticker)
        # df = pyupbit.get_ohlcv(ticker, interval="day") 
        if df is None or df.empty:
            continue
    
        df['range'] = (df['high'] - df['low']) * k  # *고가 - 저가)*k로 range열 생성
        df['target'] = df['open'] + df['range'].shift(1)  # 시가 + range로 target열 생성
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으면 업데이트
            bestC = ticker
            interest = df['hpr'].iloc[-1]
            best_k = k  # 최적 K 값도 업데이트

        time.sleep(1)  # API 호출 제한을 위한 대기

    return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환
    
def get_target_price(ticker, k):  #변동성 돌파 전략 구현
    df = load_ohlcv(ticker)
    # df = pyupbit.get_ohlcv(ticker, interval="day", count=2) 
    if df is not None and not df.empty:
        return df['close'].iloc[-1] + (df['high'].iloc[-1] - df['low'].iloc[-1]) * k
    return 0

def get_ai_decision(ticker):
    df = load_ohlcv(ticker)

    if df is None or df.empty:
        send_discord_message("get_ai_decision/데이터가 없거나 비어 있습니다.")
        print("get_ai_decision/데이터가 없거나 비어 있습니다.")
        return None  # 데이터가 없을 경우 None 반환
    
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                "type": "text",
                "text": "You are the foremost expert in cryptocurrency investing, specifically applying the most profitable techniques for short-term trading based on the data for a given cryptocurrency and making comprehensive judgments based on all available indicators to give accurate buy and sell opinions."
                    },
                    {
                "type": "text",
                "text": "For example, you need to utilize all possible methods for price forecasting, such as moving average (MA), volatility, relative strength index (RSI) indicators, etc. to make a professional price prediction based on the given chart data and tell you whether to buy or sell at the moment."
                    },
                    {
                "type": "text",
                "text": "In particular, you should use specialized price forecasting techniques to predict prices. You should use linear regression models, random forests, or whatever specialized analytical techniques you have available to you to tell you the best time to buy and sell."
                    },
                     {
                "type": "text",
                "text": "And you can find the most profitable way to trade the RSI indicator and apply it to your trading. So, for example, if the RSI is below 30, you buy when it's oversold, and so on, you can find the best technique and apply it."
                    },
                    {
                "type": "text",
                "text": "So, based on what we've discussed so far, you use your expertise to analyze the current price based on the chart data provided and tell us whether the coin will rise by more than 1.05 times the current price within 3 hours."
                    },
                    {
                "type": "text",
                "text": "So, if your analysis indicates that the price will be more than 1.05 times higher than the current price in 3 hours, please provide a 'BUY' response; if the analysis indicates that the price will be about 1.01 times higher than the current price in 3 hours, please provide a 'HOLD' response; and if the analysis indicates that the price will be lower than the current price in 3 hours, please provide a 'SELL' response in JSON format.\n\nResponse Example:\n{\"decision\": \"BUY\"}\n{\"decision\": \"SELL\"}\n{\"decision\": \"HOLD\"}"
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": df.to_json()
                    }
                ]
                }
            ],
            response_format={
                "type": "json_object"
            }
            )
    except Exception as e:
        print(f"get_ai_decision / AI 요청 중 오류 발생: {e}")
        send_discord_message(f"get_ai_decision / AI 요청 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None  # 오류 발생 시 None 반환
    
    decision_data = response.choices[0].message.content      # 응답에서 필요한 정보만 추출

    if decision_data:
        try:
            decision_json = json.loads(decision_data)
            decision = decision_json.get('decision')
            if decision in {'BUY', 'SELL', 'HOLD'}:
                return decision
        except json.JSONDecodeError:
            print("get_ai_decision / 응답을 JSON으로 파싱하는 데 실패")
            send_discord_message("응답을 JSON으로 파싱하는 데 실패")
            time.sleep(5)  # API 호출 제한을 위한 대기
    send_discord_message("get_ai_decision/유효하지 않은 응답")
    print("get_ai_decision/유효하지 않은 응답")
    return None  # 유효하지 않은 경우 None 반환

def trade_buy(ticker, k):
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1]) 
    max_retries = 10  
    buy_size = min(krw * 0.9995, 200_000)  
    ai_decision = get_ai_decision(ticker)  

    attempt = 0  # 시도 횟수 초기화
    
    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 50000 :  # 매수 조건 확인
        if ai_decision == "BUY" :
            target_price = get_target_price(ticker, k)
            
        
        while attempt < max_retries:
            current_price = get_current_price(ticker)
            print(f"가격 확인 중: {ticker}, 목표가의 98% {target_price * 0.98:,.2f} / 현재가 {current_price:,.2f} / 목표가 {target_price:,.2f}(시도 {attempt + 1}/{max_retries})")
            # send_discord_message(f"가격 확인 중: {ticker}, 목표가 {target_price:,.2f} / 현재가 {current_price:,.2f} (시도 {attempt + 1}/{max_retries})")
            # print(f"[DEBUG] 시도 {attempt + 1} / {max_retries} - 목표가 {target_price:,.2f} / 현재가: {current_price:,.2f}")

            if target_price * 0.98 <= current_price < target_price * 1.000 :
                    send_discord_message(f"{ticker}, AI: {ai_decision}")
                    print(f"매수 시도: {ticker}, 현재가 {current_price:,.2f}")
                    buy_attempts = 3
                    for i in range(buy_attempts):
                        try:
                            buy_order = upbit.buy_market_order(ticker, buy_size)
                            print(f"매수 성공: {ticker}, 현재가 {current_price:,.2f}")
                            send_discord_message(f"매수 성공: {ticker}, 목표가 98% {target_price*0.98:,.2f} < 현재가 {current_price:,.2f} < 목표가 {target_price:,.2f} / AI:{ai_decision} /(시도 {attempt + 1}/{max_retries})")
                            return buy_order
                        except Exception as e:
                            print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            # send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            time.sleep(5 * (i + 1))  # Exponential backoff
                    return "Buy order failed", None
            else:
                # print(f"현재가가 목표 범위에 도달하지 않음. 다음 시도로 넘어갑니다.")
                attempt += 1  # 시도 횟수 증가
                time.sleep(60)

        # 10회 시도 후 가격 범위에 도달하지 못한 경우
        print(f"10회 시도완료: {ticker}, 목표가 범위에 도달하지 못함")
        # send_discord_message(f"10회 시도완료: {ticker}, 목표가 범위에 도달하지 못함")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):
    """주어진 티커에 대해 매도 실행 및 수익률 출력, 매도 시간 체크"""
    selltime = datetime.now()

    """전량매도 : EC2 변환"""
    # sell_start = selltime.replace(hour=23, minute=59, second=0, microsecond=0)    #EC2
    # sell_end = selltime.replace(hour=0, minute=1, second=0, microsecond=0)      #EC2
    sell_start = selltime.replace(hour=8, minute=58 , second=00, microsecond=0)       #VC
    sell_end = selltime.replace(hour=8, minute=59, second=50, microsecond=0)         #VC

    current_price = get_current_price(ticker)
    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    avg_buy_price = upbit.get_avg_buy_price(currency)
    
    max_attempts = 1_000  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수

    if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
        sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 전량 매도
        send_discord_message(f"전량 매도: {ticker}, 현재가 {current_price} 수익률 {profit_rate:.2f}%")
        return sell_order           
    
    else:
        current_price = get_current_price(ticker)  # 현재 가격 재조회
        profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
        if profit_rate >= 0.4:  # 수익률이 0.4% 이상일 때
            while attempts < max_attempts:
                current_price = get_current_price(ticker)  # 현재 가격 재조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                
                print(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 현재가 {current_price} 수익률 {profit_rate:.2f}%")
                
                if profit_rate >= 0.75:
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    send_discord_message(f"매도: {ticker}/ 현재가 {current_price}/ 수익률 {profit_rate:.2f}%")
                    return sell_order
                else:
                    # print("수익률 0.4% 이하")
                    time.sleep(0.5)  # 짧은 대기                
                attempts += 1  # 조회 횟수 증가
                
    return None

def send_profit_report():
    while True:
        try:
            now = datetime.now()  # 현재 시간을 루프 시작 시마다 업데이트 (try 루프안에 있어야 실시간 업데이트 주의)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)   # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
            time_until_next_hour = (next_hour - now).total_seconds()
            time.sleep(time_until_next_hour)    # 다음 정시까지 기다림

            balances = upbit.get_balances()     
            report_message = "현재 수익률 보고서:\n"
            
            for b in balances:
                if b['currency'] in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 제외할 코인 리스트
                    continue
                
                ticker = f"KRW-{b['currency']}"
                buyed_amount = float(b['balance'])
                avg_buy_price = float(b['avg_buy_price'])
                current_price = get_current_price(ticker)

                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"{b['currency']} 평균가 {avg_buy_price:,.2f} 현재가 {current_price:,.3f} 수익률 {profit_rate:.1f}% \n"

            send_discord_message(report_message)  # 슬랙으로 보고서 전송
        except Exception as e:            
            print(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            send_discord_message(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

trade_start = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
print(f'{trade_start} trading start')

profit_report_thread = threading.Thread(target=send_profit_report)  # 수익률 보고 쓰레드 시작
profit_report_thread.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
profit_report_thread.start()

def selling_logic():
    while True:
        try:
            balances = upbit.get_balances()
            # print(f"selling_logic/잔고체크 {balances}")
            for b in balances:
                if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:
                        ticker = f"KRW-{b['currency']}"
                        # print(f"selling_logic/매도함수 {ticker}")
                        trade_sell(ticker)
                time.sleep(1)

        except Exception as e:
            print(f"selling_logic / 에러 발생: {e}")
            send_discord_message(f"selling_logic / 에러 발생: {e}")
            time.sleep(5)

def buying_logic():
    # 매수 제한 시간 설정
    restricted_start_hour = 8
    restricted_start_minute = 00
    restricted_end_hour = 9
    restricted_end_minute = 30

    while True:
        try:
            stopbuy_time = datetime.now()
            restricted_start = stopbuy_time.replace(hour=restricted_start_hour, minute=restricted_start_minute, second=0, microsecond=0)
            restricted_end = stopbuy_time.replace(hour=restricted_end_hour, minute=restricted_end_minute, second=0, microsecond=0)

            if restricted_start <= stopbuy_time <= restricted_end:  # 매수 제한 시간 체크
                time.sleep(600) 
                continue

            else:  # 매수 금지 시간이 아닐 때
                krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
                if krw_balance > 100_000: 
                    best_ticker, interest, best_k = get_best_ticker()
                    if best_ticker:
                        print(f"선정코인 : {best_ticker} / k값 : {best_k:,.2f} / 수익률 : {interest:,.2f}")
                        # send_discord_message(f"선정코인 : {best_ticker} / k값 : {best_k:,.2f} / 수익률 : {interest:,.2f}")
                        result = trade_buy(best_ticker, best_k)
                        if result:  # 매수 성공 여부 확인
                            time.sleep(30)
                        else:
                            time.sleep(30)
                    else:
                        time.sleep(30)
                else:
                    # print("잔고 부족 / 20분 후 다시 확인")

                    time.sleep(600)

        except Exception as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

# 매도 쓰레드 생성
selling_thread = threading.Thread(target=selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target=buying_logic)
buying_thread.start()