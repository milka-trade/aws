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
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60", count=200) 
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

def calculate_moving_average(df, window):
    return df['close'].rolling(window=window).mean().iloc[-1] if df is not None and not df.empty else 0

def get_ma5(ticker):
    df = load_ohlcv(ticker)   # 데이터프레임을 한 번 로드하고 재사용
    return calculate_moving_average(df, 5)

def get_ma10(ticker):
    df = load_ohlcv(ticker)   # 데이터프레임을 한 번 로드하고 재사용
    return calculate_moving_average(df, 10)

def get_ma15(ticker):
    df = load_ohlcv(ticker)   # 데이터프레임을 한 번 로드하고 재사용
    return calculate_moving_average(df, 15)

def get_ma20(ticker):
    df = load_ohlcv(ticker)   # 데이터프레임을 한 번 로드하고 재사용
    return calculate_moving_average(df, 20)

def get_best_k(ticker="KRW-BTC"):
    bestK = 0.5  # 초기 K 값
    interest = 0  # 초기 수익률
    df = load_ohlcv(ticker)  # 데이터 로드
    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환
    for k in np.arange(0.1, 0.6, 0.05):  # K 값을 0.1부터 0.55까지 반복
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

def filtered_tickers(tickers, held_coins):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []

    for t in tickers:
        currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
            continue

        try:
            df = load_ohlcv(t)  # 캐싱된 데이터를 사용
            # print(df)
            if df is None or df.empty:
                print(f"filtered_tickers/No data for ticker: {t}")
                send_discord_message(f"filtered_tickers/No data for ticker: {t}")
                continue

            df_day = pyupbit.get_ohlcv(t, interval="day", count=3)   #일봉 20개 조회
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue  

            cur_price = get_current_price(t)
            
            if len(df_day) >= 3:
                day_value_1 = df_day['value'].iloc[-1]      #일봉 9시 기준 당일 거래량
                day_value_2 = df_day['value'].iloc[-2]      #일봉 9시 기준 전일 거래량 
                day_value_3 = df_day['value'].iloc[-3]      #일봉 9시 기준 2일전 거래량
            else:
                print(f"filtered_tickers/Insufficient data for ticker: {t}")
                # send_discord_message(f"filtered_tickers/Insufficient data for ticker: {t}")
                continue

            if len(df) >= 3:
                min60_value_1 = df['value'].iloc[-1]      #min60_봉 전 봉 거래량
                min60__value_2 = df['value'].iloc[-2]      #min60_봉 전 봉 거래량
                min60__value_3 = df['value'].iloc[-3]      #min60_봉 전 봉 거래량
            else:
                print(f"filtered_tickers/Insufficient data for ticker: {t}")
                # send_discord_message(f"filtered_tickers/Insufficient data for ticker: {t}")
                continue

            day_open_price_1 = df_day['open'].iloc[-1]  #9시 기준 당일 시가 
            day_close_price_1 = df_day['close'].iloc[-1]  #9시 기준 당일 종가 
            day_high_price_1 = df_day['high'].iloc[-1]  #9시 기준 당일 고가 

            df_open_1 = df['open'].iloc[-1]           #현재 정시 기준 시가 
            df_close_1 = df['close'].iloc[-1]         #현재 정시 기준 종가 
            df_high_1 = df['high'].iloc[-1]           #현재 정시 기준 고가 
            
            ma20=get_ma20(t)

            # EMA 계산
            # short_period = 5  # 짧은 기간 EMA
            # long_period = 20  # 긴 기간 EMA
            # df['short_ema'] = df['close'].ewm(span=short_period, adjust=False).mean()
            # df['long_ema'] = df['close'].ewm(span=long_period, adjust=False).mean()

            # # 교차 여부 확인
            # if (df['short_ema'].iloc[-1] > df['long_ema'].iloc[-1] and 
            #     df['short_ema'].iloc[-2] <= df['long_ema'].iloc[-2]):
            #     print(f"EMA 교차 발생: {t}")
            #     send_discord_message(f"EMA 교차 발생: {t}")

                        # print(f"{t} /day:{day_value_1:,.0f} /60min:{value_1:,.0f}")
            if day_value_1 >= 25_000_000_000 : 
                print(f"1.거래량 {t} 일 2,000백만:{day_value_1:,.0f}")
                            # send_discord_message(f"1.거래량 {t} 일 25,000백만:{day_value_1:,.0f} 또는 분봉 1,500백만:{value_1:,.0f}")
                        # if value_1>=500_000_000 : 
                        #     print(f"1.분봉 거래량 500백만 : {t} / 전봉 : {value_2:,.0f} / 현재봉 : {value_1:,.0f}")
                            # send_discord_message(f"1.분봉 거래량 1,500백만:{value_1:,.0f}")
                            
            # if min60_value_1 >= 1_500_000_000 : 
            #     print(f"1.거래량 {t} 60봉 1,500백만:{min60_value_1:,.0f}")
                            
                if day_open_price_1*1.05 >= cur_price :                            
                    # print(f"2.일봉 5%이내 : {t} / 일봉시가:{day_open_price_1} / 일봉종가:{day_close_price_1} / 현재가:{cur_price}")
                                    # send_discord_message(f"2.일봉 5%이내 : {t} / 일봉시가:{day_open_price_1} / 일봉종가:{day_close_price_1} / 현재가:{cur_price}")
                                    # send_discord_message(f"2.일봉 양봉~6%이내 : {t} / 일봉시가:{df_open_1} / 현재가:{cur_price}") 
                                    # if df_close_3 > df_close_2 :                           #3.2봉전 종가보다 1봉전 종가보다 작음 (음봉)
                                    # if day_open_price_1*1.3 <= day_high_price_1 :                            
                                    #     print(f"3-1.당일봉 고가 20프로 이내 : {t} / 일봉시가:{day_open_price_1} / 일봉 고가:{day_high_price_1}")
                                    # if day_open_price_2*1.3 <= day_high_price_2 :                            
                                    #         print(f"3-2.전일봉 고가 20프로 이내 : {t} / 전일봉시가:{day_open_price_2} / 전일봉 고가:{day_high_price_2}")
                                    #         if day_open_price_3*1.3 <= day_high_price_3 :                            
                                    #             print(f"3-3.2전일봉 고가 20프로 이내 : {t} / 2전일봉시가:{day_open_price_3} / 2전일봉 고가:{day_high_price_3}")
                                    
                    if ma20 < cur_price and df_open_1*1.03 > cur_price :    #4.현재가가 60분봉 20일 이동평균 이상이고 시가보다 큼 (상승지표)
                        # print(f"3.분봉 20이평 이상|시가 3프로 이내 : {t} /  분봉시가:{df_open_1:,.0f} / 현재가:{cur_price:,.0f}/ / ma20:{ma20:,.2f}")

                            # RSI 계산
                        # rsi_period = 14
                        # delta = df['close'].diff()
                        # gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                        # loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                        # rs = gain / loss
                        # rsi = 100 - (100 / (1 + rs))

                        #                 # RSI가 30 이하에서 반등하는지 확인
                        # if rsi.iloc[-2] < 30 and rsi.iloc[-1] >= 30:
                        #     print(f"4.RSI 반등 발생: {t} / RSI:{rsi.iloc[-1]:.2f}")


                            # send_discord_message(f"3.분봉 20이평 이상|시가 3프로 이내 : {t} /  분봉시가:{df_open_1:,.0f} / 현재가:{cur_price:,.0f}/ / ma20:{ma20:,.2f}")
                        
                                # if df_high_1 > cur_price * 1.05 :    #4.현재가가 60분봉 20일 이동평균 이상이고 시가보다 큼 (상승지표)
                                #     print(f"4.분봉 현재가 고가의 5% 이내: {t} / 분봉고가:{df_high_1} / 현재가:{cur_price}")


                                # send_discord_message(f"{t} / price:{cur_price}/ 시가:{df_open_1} / ma20:{ma20:,.2f}")  
                                # if value_2 < value_1 :                                                                                                                                                                            #6.현재봉 거래량이 1봉전 거래량보다 큼 (상승지표) 
                                    # print(f"4.분봉 거래량 상승 : {t} / price:{cur_price}/ 시가:{df_open_1} / ma20:{ma20:,.2f}/value2:{value_2:,.0f} / value1:{value_1:,.0f}") 
                                    # send_discord_message(f"{t} / value2:{value_2:,.0f} / value1:{value_1:,.0f}")  

                        ai_decision = get_ai_decision(t)  #7.AI의 판단을 구함
                        print(f"5.AI판단 : {t} / AI:{ai_decision}")
                        # send_discord_message(f"{t} / AI:{ai_decision}")  
                        if ai_decision == "BUY" :
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
                "text": "You are the all-time expert on cryptocurrency investing, with billions of years of know-how and a tremendous ability to make money, especially in short-term trading."
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
                "text": "Based on the chart data given to you, you will use specialized techniques to analyze the current price and tell me whether the coin will increase by more than 1.05 times the current price within 3 hours."
                    },
                    {
                "type": "text",
                "text": "So, if your analysis indicates that the price will be more than 1.05 times higher than the current price in 3 hours, please provide a 'buy' response; if the analysis indicates that the price will be about 1.01 times higher than the current price in 3 hours, please provide a 'hold' response; and if the analysis indicates that the price will be lower than the current price in 3 hours, please provide a 'sell' response in JSON format.\n\nResponse Example:\n{\"decision\": \"BUY\"}\n{\"decision\": \"SELL\"}\n{\"decision\": \"HOLD\"}"
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
    max_retries = 30  # Maximum attempts to check if price is within range
    buy_size = min(krw * 0.9995, 150000)  # Example: 5% of available KRW balance or 100,000, whichever is smaller            

    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 50000 :  # 매수 조건 확인
        target_price = get_target_price(ticker, k)
        attempt = 0  # 시도 횟수 초기화
        
        while attempt < max_retries:
            current_price = get_current_price(ticker)
            print(f"가격 확인 중: {ticker}, 목표가 {target_price:,.2f} / 현재가 {current_price:,.2f} (시도 {attempt + 1}/{max_retries})")
            # send_discord_message(f"가격 확인 중: {ticker}, 목표가 {target_price:,.2f} / 현재가 {current_price:,.2f} (시도 {attempt + 1}/{max_retries})")
            # print(f"[DEBUG] 시도 {attempt + 1} / {max_retries} - 목표가 {target_price:,.2f} / 현재가: {current_price:,.2f}")

            if target_price * 0.98 <= current_price < target_price * 1.005 :
                
                    print(f"매수 시도: {ticker}, 현재가 {current_price:,.2f}")
                    buy_attempts = 3
                    for i in range(buy_attempts):
                        try:
                            buy_order = upbit.buy_market_order(ticker, buy_size)
                            print(f"매수 성공: {ticker}, 현재가 {current_price:,.2f}")
                            send_discord_message(f"매수 성공: {ticker}, 목표가 {target_price:,.2f} / 현재가 {current_price:,.2f} / (시도 {attempt + 1}/{max_retries})")
                            return buy_order
                        except Exception as e:
                            print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            # send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            time.sleep(5 * (i + 1))  # Exponential backoff
                    return "Buy order failed", None
            else:
                print(f"현재가가 목표 범위에 도달하지 않음. 다음 시도로 넘어갑니다.")
            
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
    
    
    max_attempts = 50  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수

    if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
        current_price = get_current_price(ticker)  # 현재 가격 재조회
        profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
        if profit_rate > -1 :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 전량 매도
            send_discord_message(f"전량 매도: {ticker}, 현재가 {current_price} 수익률 {profit_rate:.2f}%")
            return sell_order
    
    else:
        current_price = get_current_price(ticker)  # 현재 가격 재조회
        profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
        if profit_rate >= 0.80:  # 수익률이 0.8% 이상일 때
            while attempts < max_attempts:
                current_price = get_current_price(ticker)  # 현재 가격 재조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                
                print(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 수익률 {profit_rate:.2f}%")
                # send_discord_message(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 수익률 {profit_rate:.2f}%")
                
                # 수익률이 0.6% 미만이거나 1.5% 초과인 경우 매도
                if profit_rate <= 0.6 or profit_rate >= 1.5:
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    send_discord_message(f"매도: {ticker}/ 현재가 {current_price}/ 수익률 {profit_rate:.2f}%")
                    return sell_order
                else:
                    print("수익률이 0.6% 이상 1.5% 이하입니다. 계속 감시합니다.")
                    time.sleep(0.1)  # 짧은 대기
                
                attempts += 1  # 조회 횟수 증가
                
    return None

    # else:
    #     if buyed_amount > 0 :  # 보유잔고가 0 이상이면
    #         if profit_rate >= 0.65:  # 0.65% 이상 수익률일 때 AI의 판단을 구함
    #             while attempts < max_attempts:
    #                 ai_decision = get_ai_decision(ticker)
    #                 print(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 수익률 {profit_rate:.2f}% / AI {ai_decision}")
    #                 send_discord_message(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 수익률 {profit_rate:.2f}% / AI {ai_decision}")
    
    #                 if ai_decision == "SELL" or profit_rate >= 1.1 :
    #                     sell_order = upbit.sell_market_order(ticker, buyed_amount)
    #                     send_discord_message(f"매도: {ticker}/ 현재가 {current_price}/ 수익률 {profit_rate:.2f}%/ AI {ai_decision}")
    #                     return sell_order
    #                 else:
    #                      print("조건 미충족, AI 판단을 다시 확인합니다.")

    #                 attempts += 1  # 조회 횟수 증가
    #                 time.sleep(0.1)
    #         # print(f"{datetime.now().strftime('%m/%d %H:%M:%S')} 매도 확인용")
    #         return
    #     return
                

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
    restricted_start_hour = 7
    restricted_start_minute = 0
    restricted_end_hour = 9
    restricted_end_minute = 10

    while True:
        try:
            stopbuy_time = datetime.now()
            restricted_start = stopbuy_time.replace(hour=restricted_start_hour, minute=restricted_start_minute, second=0, microsecond=0)
            restricted_end = stopbuy_time.replace(hour=restricted_end_hour, minute=restricted_end_minute, second=0, microsecond=0)

            if restricted_start <= stopbuy_time <= restricted_end:  # 매수 제한 시간 체크
                time.sleep(1200) 
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
                            time.sleep(1200)
                        else:
                            time.sleep(1200)
                    else:
                        time.sleep(1200)
                else:
                    # print("잔고 부족 / 20분 후 다시 확인")

                    time.sleep(1200)

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