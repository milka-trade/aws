import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import pytz   #pytx라이브러리 호출 : 시간대 관련 작업 처리
import requests

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5)  # API 호출 제한을 위한 대기


df_tickers = {}    # 전역변수:일봉 데이터프레임

def load_ohlcv(ticker):
    global df_tickers
    # print(df_tickers)   #검증용

    if ticker not in df_tickers:   # 티커가 캐시에 없으면 데이터 가져오기     
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, count=20) # interval="minute60",
            # print(df_tickers)  #검증용
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"No data returned for ticker: {ticker}")
                time.sleep(0.5)  # API 호출 제한을 위한 대기
        except Exception as e:
            print(f"Error loading data for ticker {ticker}: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기
    return df_tickers.get(ticker)

def get_balance(ticker):    #잔고조회
    try:
        balances = upbit.get_balances()  # 모든 잔고 조회
        for b in balances:
            if b['currency'] == ticker:  # 요청한 통화의 잔고를 찾음
                time.sleep(0.5)  # API 호출 제한을 위한 대기
                return float(b['balance']) if b['balance'] is not None else 0
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
        time.sleep(5)  # API 호출 제한을 위한 대기
        return 0
    return 0  # 잔고가 없거나 조회에 실패한 경우 0 반환

def get_current_price(ticker):
    """현재가를 조회합니다."""
    if not ticker.startswith("KRW-"):
        print(f"잘못된 티커 형식: {ticker}")
        return None

    try:
        orderbook = pyupbit.get_orderbook(ticker=ticker)
        if orderbook is None or "orderbook_units" not in orderbook or not orderbook["orderbook_units"]:
            raise ValueError(f"'{ticker}'에 대한 유효한 orderbook이 없습니다.")
        current_price = orderbook["orderbook_units"][0]["ask_price"]
        time.sleep(0.5)  # API 호출 제한을 위한 대기
        return current_price
    except Exception as e:
        print(f"현재가 조회 오류 ({ticker}): {e}")
        time.sleep(5)  # API 호출 제한을 위한 대기
        return None

def calculate_moving_average(df, window):
    """이동평균선 계산"""
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
    """최적의 K 값 계산"""
    bestK = 0.5  # 초기 K 값
    interest = 0  # 초기 수익률

    df = load_ohlcv(ticker)  # 데이터 로드
    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환

    for k in np.arange(0.1, 1.0, 0.1):  # K 값을 0.1부터 0.1까지 반복
        df['range'] = (df['high'] - df['low']) * k      #변동성 계산
        df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정
        fee = 0.0005  # 거래 수수료 (0.05%로 설정)
        # 수익률 계산
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
                print(f"No data for ticker: {t}")
                send_discord_message(f"No data for ticker: {t}")
                continue

            df_day = pyupbit.get_ohlcv(t, interval="day", count=3)   #3일봉 조회
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue  

            today_open_price = df_day['open'].iloc[0]   # 당일 시가 조회
            current_price = get_current_price(t)
        
            if current_price is not None and today_open_price is not None:  # 현재가와 시가가 모두 유효한 경우
                if current_price > today_open_price * 1.05: # 현재가가 시가 대비 5% 이상 상승한 경우 제외
                    continue

            # yesterday_volume = df['volume'].iloc[-2]  # 전봉 거래량
            today_volume = df_day['volume'].iloc[-1]      # 현재봉 거래량
            
            # 거래량 증가 비율 계산
            if today_volume >= 50_000_000:  # 전일 거래량이 50백만 이상
                df_open=df['open'].iloc[0]
                df_close=df['close'].iloc[-1]
                if current_price >= df_open and current_price <= df_open*1.03 :  # 현재가 양봉, 분봉의 3% 이내 상승
                    if get_ma20(t) >= get_ma5(t):  # 5이평이 20이평 아래에 있는 경우
                        if get_ma10(t) <= get_ma5(t):  # 5이평이 10이평 위에 있는 경우
                            if get_ma5(t) <= current_price and get_ma5(t)*1.01 > current_price :  # 현재가가 5이평 이상 5이평*1%이내에 있는 경우
                                filtered_tickers.append(t)
                                # print(filtered_tickers)    #검증용
                        time.sleep(0.5)  # API 호출 제한을 위한 대기
        except Exception as e:
            print(f"Error processing ticker {t: {e}}")
            time.sleep(5)  # API 호출 제한을 위한 대기

    return filtered_tickers

def get_best_ticker():
    """최고의 코인과 수익률 반환"""
    
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
        # print(tickers)
        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_discord_message(f"티커 조회 중 오류 발생: {e}")
        print(f"티커 조회 중 오류 발생: {e}")
        time.sleep(5)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_list = filtered_tickers(tickers, held_coins)
    # print(filtered_list)
    send_discord_message(f"{filtered_list}")        
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.5  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        df = load_ohlcv(ticker)
        if df is None or df.empty:
            continue
    
        df['range'] = (df['high'] - df['low']) * k  # 변동성 계산
        df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으면 업데이트
            bestC = ticker
            interest = df['hpr'].iloc[-1]
            best_k = k  # 최적 K 값도 업데이트

        time.sleep(1)  # API 호출 제한을 위한 대기

    return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환
    
def get_target_price(ticker, k):  #변동성 돌파 전략 구현
    """목표 가격 계산"""
    df = load_ohlcv(ticker)
    if df is not None and not df.empty:
        return df['close'].iloc[0] + (df['high'].iloc[0] - df['low'].iloc[0]) * k
    return 0

def get_ai_decision(ticker):
    """AI에게 매매 결정을 요청하는 함수"""
    df = load_ohlcv(ticker)

    if df is None or df.empty:
        print("데이터가 없거나 비어 있습니다.")
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
                "text": "You are the ultimate expert on cryptocurrency investing. You've never failed to make a trade, and you're a master at analyzing when you're going to make money or lose money, especially when it comes to timing your trades."
                    },
                    {
                "type": "text",
                "text": "Based on the chart data provided, tell us whether you're currently buying, selling."
                    },
                    {
                "type": "text",
                "text": "Response in json format.\n\nResponse Example:\n{\"decision\": \"BUY\"}\n{\"decision\": \"SELL\"}\n{\"decision\": \"HOLD\"}"
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
        print(f"AI 요청 중 오류 발생: {e}")
        send_discord_message(f"AI 요청 중 오류 발생: {e}")
        time.sleep(5)  # API 호출 제한을 위한 대기
                
        return None  # 오류 발생 시 None 반환
    
    decision_data = response.choices[0].message.content      # 응답에서 필요한 정보만 추출

    if decision_data:
        try:
            decision_json = json.loads(decision_data)
            decision = decision_json.get('decision')
            if decision in {'BUY', 'SELL', 'HOLD'}:
                return decision
        except json.JSONDecodeError:
            print("응답을 JSON으로 파싱하는 데 실패했습니다.")
            time.sleep(5)  # API 호출 제한을 위한 대기
            send_discord_message("응답을 JSON으로 파싱하는 데 실패")
    
    print("유효하지 않은 응답입니다.")
    send_discord_message("유효하지 않은 응답")
    return None  # 유효하지 않은 경우 None 반환

# KST = pytz.timezone('Asia/Seoul')

def trade_buy(ticker, k):
    """주어진 티커에 대해 매수 실행"""
    current_price = get_current_price(ticker) 
    target_price = get_target_price(ticker, k)
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1])
    ma5 = get_ma5(ticker)
    ma20 = get_ma20(ticker)

    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 5000 :  # 매수 조건 확인
            # print(f"매수시도: {ticker}, 목표가: {target_price}, 현재가: {current_price}")    #이평5: {ma5:.2f}, 이평15: {ma15:.2f}, , AI: {ai_decision}
            # send_discord_message(f"매수시도: {ticker}, 목표가: {target_price}, 현재가: {current_price}")
            if target_price <= current_price and current_price < target_price*1.05 :  #현재가가 목표가 이상이면서 목표가의 5% 이내
                ai_decision = get_ai_decision(ticker)  
                # print(ai_decision)
                # send_discord_message(f"AI: {ai_decision}")
                if ai_decision != 'SELL' :  # AI의 판단이 NOT SELL이면
                    try:
                        buy_order = upbit.buy_market_order(ticker, 160_000)
                        # print(f"매수: {ticker}, 현재가: {current_price}")
                        send_discord_message(f"매수 {ticker}, 목표가 {target_price:.3f} 현재가{current_price} 5이평{ma5} 20이평{ma20} AI{ai_decision}")
                        # time.sleep(5)  # API 호출 제한을 위한 대기
                        return buy_order['price'], target_price
                            
                    except Exception as e:
                        # print(f"매수 주문 실행 중 오류 발생: {e}")
                        send_discord_message(f"매수 주문 실행 중 오류 발생: {e}")
                        time.sleep(5)  # API 호출 제한을 위한 대기
                        return "Buy order failed", None

def trade_sell(ticker):
    """주어진 티커에 대해 매도 실행 및 수익률 출력, 매도 시간 체크"""
    current_time = datetime.now()

    """매도제한시간 체크 : EC2 변환"""
    # sell_start = current_time.replace(hour=23, minute=59, second=00, microsecond=0)    #EC2
    # sell_end = current_time.replace(hour=23, minute=59, second=50, microsecond=0)      #EC2
    sell_start = current_time.replace(hour=8, minute=59, second=00, microsecond=0)       #VC
    sell_end = current_time.replace(hour=8, minute=59, second=50, microsecond=0)         #VC

    current_price = get_current_price(ticker)
    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    # print(buyed_amount)
    avg_buy_price = upbit.get_avg_buy_price(currency)
    # print(avg_buy_price)

    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    # print(profit_rate)

    # 매도 조건: 23시 59분 0초 ~ 23시 59분 50초 사이
    if sell_start <= current_time <= sell_end:
        # balances = upbit.get_balances()  # 모든 보유 코인 조회
        # for b in balances:
        #     if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 해당 잔고 제외
        #         ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 전량 매도
                send_discord_message(f"전량 매도: {ticker}, 현재가 {current_price} 수익률 {profit_rate:.2f}%")
                return sell_order
    
    if buyed_amount > 0 :  # 보유잔고가 0 이상이면
        # if currency not in ["BTC", "ETH"]:
            if profit_rate > 0.6:  # 0.6% 이상 수익률일 때 AI의 판단을 구함
                ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
                send_discord_message(f"{ticker} AI:{ai_decision}")
                # print(f"ai:{ai_decision}")            
                if ai_decision == 'SELL' or profit_rate > 1.2:  # AI의 판단이 not buy or 수익률이 1.1%를 넘는 경우 매도
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                    # sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                    # print(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (수익률 5% 초과 또는 AI 판단 SELL)")
                    send_discord_message(f"매도: {ticker}, 현재가 {current_price} 수익률 {profit_rate:.2f}% (수익률 2% 초과 또는 AI 판단 sell)")
                    return sell_order

def send_profit_report():       #한 시간마다 수익률을 디스코드 메시지로 전송
    while True:
        try:
            now = datetime.now()     # 현재 시간을 확인
            
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)   # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
            time_until_next_hour = (next_hour - now).total_seconds()
            time.sleep(time_until_next_hour)    # 다음 정시까지 기다림
            
            # hours, remainder = divmod(time_until_next_hour, 3600)       # 남은 시간을 시간, 분, 초로 변환
            # minutes, seconds = divmod(remainder, 60)
            # print(f"미국시간: {now}, 다음 정시: {next_hour}, 남은 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
            # send_discord_message(f"현재 시간: {now}, 남은 시간: {int(minutes)}분 {int(seconds)}초")

            balances = upbit.get_balances()     # 정시에 수익률 계산 및 슬랙 메시지 전송
            report_message = "현재 수익률 보고서:\n"
            
            for b in balances:
                if b['currency'] in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 제외할 코인 리스트
                    continue
                
                ticker = f"KRW-{b['currency']}"
                buyed_amount = float(b['balance'])
                avg_buy_price = float(b['avg_buy_price'])
                current_price = get_current_price(ticker)

                if buyed_amount > 0:
                    ai_decision = get_ai_decision(ticker)
                    # try_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 보고시간
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"{b['currency']} 현재가{current_price} 평균가{avg_buy_price:.3f} 수익률{profit_rate:.1f}% AI{ai_decision} \n"

            send_discord_message(report_message)  # 슬랙으로 보고서 전송
        except Exception as e:            
            print(f"수익률 보고 중 오류 발생: {e}")
            send_discord_message(f"수익률 보고 중 오류 발생: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

print("trading start")      # 자동매매 시작

profit_report_thread = threading.Thread(target=send_profit_report)  # 수익률 보고 쓰레드 시작
profit_report_thread.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
profit_report_thread.start()

while True:
    try:
        current_time = datetime.now()
        """매수제한시간 체크 : EC2 변환"""
        restricted_start = current_time.replace(hour=8, minute=0, second=0, microsecond=0)  # 08:00  #EC2
        restricted_end = current_time.replace(hour=9, minute=30, second=0, microsecond=0)  # 09:30   #EC2
        # restricted_start = current_time.replace(hour=23, minute=0, second=0, microsecond=0)  # 23:00  #VC
        # restricted_end = current_time.replace(hour=0, minute=30, second=0, microsecond=0)  # 00:30    #VC
        krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회

        if krw_balance < 160_000:       # 잔고가 매수설정금액 미만일 경우
            balances = upbit.get_balances()  
            for b in balances:
                if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 보유 잔고가 있는 경우
                    ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                    trade_sell(ticker)  # 매도 실행
            time.sleep(1)  # 1초 대기 후 재 실행

        else:  # 잔고가 매수설정금액 이상일 경우
            # print(best_ticker, interest, best_k)   #검증용
            if restricted_start <= current_time <= restricted_end:  # 제한 시간이면
                    balances = upbit.get_balances()  # 모든 보유 코인 조회
                    for b in balances:
                        if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 해당 잔고는 재외
                            ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                            trade_sell(ticker)  # 매도 실행
                        
            else:  # 제한 시간대가 아니면
                balances = upbit.get_balances()  # 모든 보유 코인 조회
                for b in balances:
                    if b['currency'] not in ["KRW",  "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 해당 잔고는 제외
                        ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                        # print(ticker)   #검증용
                        trade_sell(ticker)  # 매도 실행
                        
                best_ticker, interest, best_k = get_best_ticker()  # 최고의 코인 조회
                if best_ticker:  # 최고의 코인이 존재하면
                    # print("매수시도")   #검증용
                    result = trade_buy(best_ticker, best_k)  # 매수 실행
                    time.sleep(1)  # 매수 후 1초 대기
                
        time.sleep(1)

    except Exception as e:
        print(f"에러 발생: {e}")  # 발생하는 에러 출력
        send_discord_message(f"에러 발생: {e}")
        time.sleep(5)  # 에러 발생 시 10초 대기
