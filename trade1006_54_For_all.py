import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from slack_sdk import WebClient
import pytz   #pytx라이브러리 호출 : 시간대 관련 작업 처리

load_dotenv()

slack_token = os.getenv("slack_token")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
client = WebClient(token=slack_token)

def send_slack_message(channel, message):
    """슬랙 메시지 전송"""
    try:
        client.chat_postMessage(channel=channel, text=message)
    except Exception as e:
        print(f"슬랙 메시지 전송 실패 : {e}")

# 1시간마다 수익률 전송을 위한 함수
def send_profit_report():
    """보유한 모든 자산에 대한 수익률을 슬랙으로 보고"""
    balances = upbit.get_balances()  # 모든 잔고 조회
    report_message = "현재 수익률 보고서:\n"

    for b in balances:
        ticker = f"KRW-{b['currency']}"  # 티커 형식으로 변환
        if b['currency'] in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:
            continue  # 제외할 코인

        buyed_amount = float(b['balance'])  # 보유량
        avg_buy_price = float(b['avg_buy_price'])  # 평균 매수 가격
        current_price = get_current_price(ticker)  # 현재가 조회

        if buyed_amount > 0:
            profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100  # 수익률 계산
            report_message += f"{b['currency']}: 현재가 {current_price}, 수익률 {profit_rate:.2f}%\n"

    send_slack_message('#api_test', report_message)  # 슬랙으로 보고서 전송

df_tickers = {}    # 전역변수:일봉 데이터프레임

def load_ohlcv(ticker):   # 일봉 데이터프레임 로드 (캐시 사용)
    global df_tickers
    if ticker not in df_tickers:
        df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60")  
    return df_tickers.get(ticker)

def get_balance(ticker):    #잔고조회
    try:
        balances = upbit.get_balances()  # 모든 잔고 조회
        for b in balances:
            if b['currency'] == ticker:  # 요청한 통화의 잔고를 찾음
                return float(b['balance']) if b['balance'] is not None else 0
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
        send_slack_message('#api_test', f"잔고 조회 오류: {e}")
        return 0
    return 0  # 잔고가 없거나 조회에 실패한 경우 0 반환

def get_current_price(ticker):
    """현재가를 조회합니다."""
    if not ticker.startswith("KRW-"):
        print(f"잘못된 티커 형식: {ticker}")
        send_slack_message('#api_test', f"잘못된 티커 형식: {ticker}")
        return None

    try:
        orderbook = pyupbit.get_orderbook(ticker=ticker)
        if orderbook is None or "orderbook_units" not in orderbook or not orderbook["orderbook_units"]:
            raise ValueError(f"'{ticker}'에 대한 유효한 orderbook이 없습니다.")
        
        current_price = orderbook["orderbook_units"][0]["ask_price"]

        return current_price
    except Exception as e:
        print(f"현재가 조회 오류 ({ticker}): {e}")
        send_slack_message('#api_test', f"현재가 조회 오류 ({ticker}): {e}")
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

    for k in np.arange(0.1, 1.0, 0.05):  # K 값을 0.1부터 0.95까지 반복
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

    return bestK

def filtered_tickers(tickers, held_coins):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []

    for ticker in tickers:
        currency = ticker.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
            continue

        try:
            df = load_ohlcv(ticker)  # 캐싱된 데이터를 사용
            if df is None or df.empty:
                continue

            today_open_price = df['open'].iloc[0]
            current_price = df['close'].iloc[-1]

            if current_price >= today_open_price * 1.05:          # 현재가가 시가 대비 5% 이상 상승한 경우 제외
                continue

            yesterday_volume = df['volume'].iloc[-2]  # 전봉 거래량
            today_volume = df['volume'].iloc[-1]      # 현재봉 거래량
            
            # 거래량 증가 비율 계산
            if yesterday_volume > 0:  # 전일 거래량이 0보다 큰 경우
                volume_increase = (today_volume - yesterday_volume) / yesterday_volume
                if volume_increase > 0:  # 전봉대비 거래량이 증가한 코인 중
                    if current_price >= df['open'].iloc[-1]:  # 직전봉 거래량보다 현재봉 거래량이 크고
                        if current_price > get_ma20(ticker) and current_price > get_ma5(ticker) : # 현재가가 20봉 이평 이상, 5봉 이평 이상인 경우
                            filtered_tickers.append(ticker)

        except Exception as e:
            print(f"Error processing ticker {ticker: {e}}")

    return filtered_tickers

def get_best_ticker():
    """최고의 코인과 수익률 반환"""
    
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_slack_message('#api_test', f"티커 조회 중 오류 발생: {e}")
        print(f"티커 조회 중 오류 발생: {e}")
        return None, None, None

    filtered_list = filtered_tickers(tickers, held_coins)
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
    # print(f"best_ticker: {bestC}, interest: {interest}, best_k: {best_k}")
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
        # print("데이터가 없거나 비어 있습니다.")
        send_slack_message('#api_test', "데이터가 없거나 비어 있음")
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
        send_slack_message('#api_test', f"AI 요청 중 오류 발생: {e}")
                
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
            send_slack_message('#api_test', "응답을 JSON으로 파싱하는 데 실패")
    
    print("유효하지 않은 응답입니다.")
    send_slack_message('#api_test', "유효하지 않은 응답")
    return None  # 유효하지 않은 경우 None 반환

KST = pytz.timezone('Asia/Seoul')
UTC = pytz.utc

def get_btc_market_open_time():     #for EC2
    """Returns the BTC market open time in UTC"""
    # Assuming BTC market starts at 00:00 UTC, adjust as needed based on market schedule.
    market_open_time_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return market_open_time_utc

def is_within_restricted_time():    #for EC2
    """Checks if current time is within 1 hour before and 30 minutes after BTC market open"""
    market_open_time_utc = get_btc_market_open_time()
    current_time_utc = datetime.now(UTC)  #한국시간보다 9시간 느림 09:00 → 00:00
    
        # Define restricted time frame
    restricted_start = market_open_time_utc - timedelta(hours=1)  # 1 hour before market open
    restricted_end = market_open_time_utc + timedelta(minutes=30)  # 30 minutes after market open

    return restricted_start <= current_time_utc <= restricted_end

# """(start)for NOT EC2"""
# def is_within_restricted_time_kst():  #for NOT EC2
#     """Checks if the current time is between 08:00 and 09:30 KST"""
#     current_time_kst = datetime.now(KST)

#     # Define restricted time window (08:00 to 09:30 KST)
#     restricted_start = current_time_kst.replace(hour=8, minute=0, second=0, microsecond=0)
#     restricted_end = current_time_kst.replace(hour=9, minute=30, second=0, microsecond=0)

#     return restricted_start <= current_time_kst <= restricted_end
# """(end)for NOT EC2"""


def trade_buy(ticker, k):
    """주어진 티커에 대해 매수 실행"""
    current_price = get_current_price(ticker) 
    target_price = get_target_price(ticker, k)
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1])
    ma5 = get_ma5(ticker)
    # ma10 = get_ma10(ticker)
    ma20 = get_ma20(ticker)
    current_time = datetime.now()  # 현재 시간 조회
    # current_time_ust = datetime.now(UTC)  # 현재 시간 조회
    # current_time_kst = datetime.now(KST)
    
    """(start)for EC2"""
        # Get current time in UTC   #for EC2
    
     # Check if we are within the restricted time frame
    if is_within_restricted_time():     #for EC2
        # print(f"Trading paused during restricted period. Current time: {current_time}")
        return None  # No trade executed
    """(end)for EC2"""


    # """(start)for NOT EC2"""
    #     # Get current time in KST
    # # current_time_kst = datetime.now(KST)  #for NOT EC2
    #     #Check if we are within the restricted time frame
    # if is_within_restricted_time_kst():     #for NOT EC2
    #     # print(f"Trading paused from 08:00 to 09:30 KST. Current time: {current_time_kst}")
    #     return None  # No trade executed
    # """(end)for NOT EC2"""


    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 5000 :  # 매수 조건 확인
        try_time = current_time.strftime('%Y-%m-%d %H:%M:%S')                
        
        if target_price*0.95<= current_price :
            if current_price >= ma5 :
                ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
                # print(f"최고의 코인: {ticker}, AI의 판단: {ai_decision}")
                print(f"{try_time} 코인: {ticker}, 목표가: {target_price}, 이평5: {ma5:.2f}, 이평20: {ma20:.2f}, 현재가: {current_price}, AI의 판단: {ai_decision}")
                # send_slack_message('#api_test', f"{try_time} 코인: {ticker}, 목표가: {target_price}, 이평5: {ma5:.2f}, 이평20: {ma20:.2f}, 현재가: {current_price}, AI의 판단: {ai_decision}")
                try:
                        buy_order = upbit.buy_market_order(ticker, krw*0.9995)
                        buy_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        # print(f"매수 시간: {buy_time}, AI의 판단: {ai_decision}, Ticker: {ticker}, 현재가: {current_price}")
                        send_slack_message('#api_test', f"매수 시간: {buy_time}, Ticker: {ticker}, 현재가: {current_price}, AI의 판단: {ai_decision}")
                        return buy_order['price'], target_price
                        
                except Exception as e:
                        # print(f"매수 주문 실행 중 오류 발생: {e}")
                        send_slack_message('#api_test', f"매수 주문 실행 중 오류 발생: {e}")
                        return "Buy order failed", None

def calculate_profit_rate(buyed_amount, avg_buy_price, ticker):
    """수익률을 계산"""
    if buyed_amount <= 0:
        return None  # 보유량이 없을 경우 수익률 계산 불가
    
    current_price = get_current_price(ticker)
    if current_price is None or avg_buy_price == 0:
        return None
    
    return (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0

# """(start)for NOT EC2"""
# def is_sell_time_kst():
#     """Checks if the current time is between 08:59:00 and 08:59:50 KST"""
#     current_time_kst = datetime.now(KST)

#     # Define the sell window (08:59:00 to 08:59:50 KST)
#     sell_start = current_time_kst.replace(hour=8, minute=59, second=0, microsecond=0)
#     sell_end = current_time_kst.replace(hour=8, minute=59, second=50, microsecond=0)

#     return sell_start <= current_time_kst <= sell_end
# """(end)for NOT EC2"""

def is_sell_time_utc():
    """Checks if the current time is between 23:59:00 and 23:59:50 UTC"""
    current_time_utc = datetime.now(UTC)

    # Define the sell window (23:59:00 to 23:59:50 UTC)
    sell_start = current_time_utc.replace(hour=23, minute=59, second=0, microsecond=0)
    sell_end = current_time_utc.replace(hour=23, minute=59, second=50, microsecond=0)

    return sell_start <= current_time_utc <= sell_end

def trade_sell(ticker, buyed_amount, avg_buy_price):
    """주어진 티커에 대해 매도 실행 및 수익률 출력"""
    current_price = get_current_price(ticker)
    evaluation_amount = buyed_amount * current_price  # 평가금액 계산

    # current_time_kst = datetime.now(KST)  # Get current time in KST
    current_time_utc = datetime.now(UTC)  # Get current time in UTC

    # 수익률 계산
    # profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
    profit_rate = calculate_profit_rate(buyed_amount, avg_buy_price, ticker)

    
    # """(start)for NOT EC2"""    
    #  # Check if we're within the special sell time frame (08:59:00 - 08:59:50 KST)
    # if is_sell_time_kst():
    #     sell_order = upbit.sell_market_order(ticker, buyed_amount)  # Market sell order
    #     sell_time = current_time_kst.strftime('%Y-%m-%d %H:%M:%S')  # Log sell time in KST
    #     send_slack_message('#api_test', f"Sold full balance at: {sell_time}, Ticker: {ticker}, Profit: {profit_rate:.2f}%")
    #     return sell_order
    # """(end)for NOT EC2"""

# Check if we're within the special sell time frame (23:59:00 - 23:59:50 UTC)
    if is_sell_time_utc():
        sell_order = upbit.sell_market_order(ticker, buyed_amount)  # Market sell order
        sell_time = current_time_utc.strftime('%Y-%m-%d %H:%M:%S')  # Log sell time in UTC
        send_slack_message('#api_test', f"Sold full balance at: {sell_time}, Ticker: {ticker}, Profit: {profit_rate:.2f}%")
        return sell_order

    if evaluation_amount > 5000:  # 평가 금액이 5000 이상인 경우에만 매도 조건 체크
        if ticker.split("-")[1] not in ["BTC", "ETH"]:
            if profit_rate > 1.25:  # 1.25% 이상 수익률일 때 AI의 판단을 구함
                ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함

                if ai_decision == 'SELL' or profit_rate > 5.0:  # AI의 판단이 SELL이거나 수익률이 5.0%를 넘는 경우 매도
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                    sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                    send_slack_message('#api_test', f"매도: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (수익률 5.0% 초과 또는 AI 판단 SELL)")
                    return sell_order

            # if profit_rate < -4:  # -4% 이하 손실 시 AI 판단 후 매도
            #     ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
                
            #     if ai_decision == 'SELL':
            #         sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
            #         sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
            #         profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
            #         # print(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (-5% 이하)")
            #         client.chat_postMessage(channel='#api_test', text=f"매도: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (-5% 이하)")
            #         return sell_order

            #     sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시도 시간 기록
            #     # print(f"매도 시도 시간: {sell_time}, 코인명 : {currency}, AI의 판단: {ai_decision}, 수익률: {profit_rate:.2f}%: 매도 조건을 충족하지 않았습니다.")
                # client.chat_postMessage(channel='#api_test', text=f"매도 실패: {sell_time}, 코인명 : {currency}, AI의 판단: {ai_decision}, 수익률: {profit_rate:.2f}%: 매도 조건 미충족")

def sell_all_assets():
    """보유하고 있는 모든 자산 매도"""
    balances = upbit.get_balances()  # 모든 보유 코인 조회
    for b in balances:
        if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 제외할 코인 리스트
            ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
            buyed_amount = float(b['balance'])  # 해당 자산의 보유량
            avg_buy_price = upbit.get_avg_buy_price(b['currency'])  # 평균 매수 가격
            trade_sell(ticker, buyed_amount, avg_buy_price)  # 매도 실행

def send_profit_report():
    """한 시간마다 수익률을 슬랙 메시지로 전송"""
    while True:
        try:

            # 현재 시간을 확인
            now = datetime.now()

            # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            time_until_next_hour = (next_hour - now).total_seconds()

            # 남은 시간을 시간, 분, 초로 변환
            hours, remainder = divmod(time_until_next_hour, 3600)
            minutes, seconds = divmod(remainder, 60)

            # print(f"미국시간: {now}, 다음 정시: {next_hour}, 남은 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
            send_slack_message('#api_test', f"현재 시간: {now}, 남은 시간: {int(minutes)}분 {int(seconds)}초")

            # 다음 정시까지 기다림
            time.sleep(time_until_next_hour)

             # 정시에 수익률 계산 및 슬랙 메시지 전송
            balances = upbit.get_balances()
            for b in balances:
                if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 제외할 코인 리스트
                    ticker = f"KRW-{b['currency']}"
                    buyed_amount = float(b['balance'])
                    avg_buy_price = float(b['avg_buy_price'])
                    current_price = get_current_price(ticker)

                    # 수익률 계산
                    profit_rate = calculate_profit_rate(buyed_amount, avg_buy_price, ticker)
                    if profit_rate is not None:
                        message = f"{ticker} 현재가:{current_price}, 평균가 : {avg_buy_price}, 수익률: {profit_rate:.2f}%."
                        send_slack_message('#api_test', message)
        except Exception as e:            
            print(f"수익률 보고 중 오류 발생: {e}")
            send_slack_message('#api_test', f"수익률 보고 중 오류 발생: {e}")


# 자동매매 시작
print("자동 매매 시작")

# 수익률 보고 쓰레드 시작
profit_report_thread = threading.Thread(target=send_profit_report)
profit_report_thread.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
profit_report_thread.start()

while True:
    try:

        krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
                
        if krw_balance < 5000:
            # 잔고가 5천원 미만일 경우 보유 코인 매도
            sell_all_assets()  # 보유 자산 매도
            time.sleep(10)  # 10초 대기 후 재 실행

        else:  # 잔고가 5천원 이상일 경우
            best_ticker, interest, best_k = get_best_ticker()  # 최고의 코인 조회
            if best_ticker:  # 최고의 코인이 존재할 경우 매수시도
                # print(f"매수 시도 : best_ticker {best_ticker}, best_k : {best_k}")
                send_slack_message('#api_test', f"매수 시도 : best_ticker {best_ticker}, interest: {interest:.2f}, best_k : {best_k:.2f}")
                result = trade_buy(best_ticker, best_k)  # 매수 실행
                time.sleep(180)  # 매수 후 잠시 대기
                
                sell_all_assets()  # 보유 자산 매도
        
        time.sleep(10)

    except Exception as e:
        print(f"에러 발생: {e}")
        send_slack_message('#api_test',f"에러 발생: {e}")
        time.sleep(10)