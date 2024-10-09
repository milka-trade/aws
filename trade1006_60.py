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
        
df_tickers = {}    # 전역변수:일봉 데이터프레임

def load_ohlcv(ticker):
    global df_tickers
    if ticker not in df_tickers:
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60", count=30)
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"No data returned for ticker: {ticker}")
        except Exception as e:
            print(f"Error loading data for ticker {ticker}: {e}")
            time.sleep(10)  # API 호출 제한을 위한 대기
    return df_tickers.get(ticker)

def get_balance(ticker):    #잔고조회
    try:
        balances = upbit.get_balances()  # 모든 잔고 조회
        for b in balances:
            if b['currency'] == ticker:  # 요청한 통화의 잔고를 찾음
                return float(b['balance']) if b['balance'] is not None else 0
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
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

        return current_price
    except Exception as e:
        print(f"현재가 조회 오류 ({ticker}): {e}")
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

    for t in tickers:
        currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
            continue

        try:
            df = load_ohlcv(t)  # 캐싱된 데이터를 사용
            # print(df)
            if df is None or df.empty:
                print(f"No data for ticker: {t}")
                # send_slack_message('#api_test', f"No data for ticker: {t}")
                continue

            df_day = pyupbit.get_ohlcv(t, interval="day", count=3)   #3일봉 조회
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue  

            today_open_price = df_day['open'].iloc[0]   # 당일 시가 조회
            current_price = get_current_price(t)
        
            if current_price is not None and today_open_price is not None:  # 현재가와 시가가 모두 유효한 경우에만 비교
                if current_price < today_open_price and current_price > today_open_price * 1.05: # 현재가가 시가 대비 5% 이상 상승한 경우 제외
                    continue

            # yesterday_volume = df['volume'].iloc[-2]  # 전봉 거래량
            today_volume = df_day['volume'].iloc[-1]      # 현재봉 거래량
            
            # 거래량 증가 비율 계산
            if today_volume >= 20_000_000:  # 현재봉 거래량이 15백만 이상
                df_open=df['open'].iloc[-1]
                if current_price >= df_open and current_price <= df_open*1.02 :  # 현재가 양봉, 분봉의 2% 이내 상승
                    if get_ma15(t) >= get_ma5(t):  # 5이평이 15이평 아래에 있는 경우
                        ai_decision = get_ai_decision(t)  
                        if ai_decision != 'SELL' :  # AI의 판단이 NOT SELL이면
                            filtered_tickers.append(t)
                            time.sleep(5)  # API 호출 제한을 위한 대기
        except Exception as e:
            print(f"Error processing ticker {t: {e}}")

    return filtered_tickers

def get_best_ticker():
    """최고의 코인과 수익률 반환"""
    
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
        # print(tickers)
        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_slack_message('#api_test', f"티커 조회 중 오류 발생: {e}")
        print(f"티커 조회 중 오류 발생: {e}")
        return None, None, None

    filtered_list = filtered_tickers(tickers, held_coins)
    """오류검증용 print"""
    # print(filtered_list)        
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.5  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        print(f"{filtered_list}")
        # send_slack_message('#api_test', f"{filtered_list}")
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

        time.sleep(5)  # API 호출 제한을 위한 대기

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

# KST = pytz.timezone('Asia/Seoul')
UTC = pytz.utc

def get_btc_market_open_time():     #for EC2 / Returns the BTC market open time in UTC      # Assuming BTC market starts at 00:00 UTC, adjust as needed based on market schedule.
    market_open_time_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return market_open_time_utc

def is_within_restricted_time():    #for EC2  / Checks if current time is within 1 hour before and 30 minutes after BTC market open
    market_open_time_utc = get_btc_market_open_time()
    current_time_utc = datetime.now(UTC)  #한국시간보다 9시간 느림 09:00 → 00:00
    
    restricted_start = market_open_time_utc - timedelta(hours=1)  # 1 hour before market open
    restricted_end = market_open_time_utc + timedelta(minutes=20)  # 30 minutes after market open

    return restricted_start <= current_time_utc <= restricted_end

def trade_buy(ticker, k):
    """주어진 티커에 대해 매수 실행"""
    current_price = get_current_price(ticker) 
    target_price = get_target_price(ticker, k)
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1])
    ma5 = get_ma5(ticker)
    ma15 = get_ma15(ticker)
    current_time = datetime.now()  # 현재 시간 조회

    if is_within_restricted_time():     #for EC2    /# Check if we are within the restricted time frame
        restrict_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        # print(f"매수 제한시간대 : {restrict_time}")
        return None  # No trade executed
    
    else:  # 제한시간이 아닐 경우 매수 프로세스 진행

        if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 5000 :  # 매수 조건 확인
            try_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            if target_price <= current_price and current_price < target_price*1.03 :  #현재가가 목표가 이상이면서 목표가의 3% 이내
                ai_decision = get_ai_decision(ticker)  
                if ai_decision != 'SELL' :  # AI의 판단이 NOT SELL이면
                    print(f"{try_time} 코인: {ticker}, 목표가: {target_price}, 현재가: {current_price}")    #이평5: {ma5:.2f}, 이평15: {ma15:.2f}, , AI: {ai_decision}
                    try:
                        buy_order = upbit.buy_market_order(ticker, krw*0.9995)
                        buy_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"매수 시간: {buy_time}, Ticker: {ticker}, 현재가: {current_price}")
                        send_slack_message('#api_test', f"매수 시간: {buy_time}, {ticker}, 목표가: {target_price} 현재가: {current_price} 이평5: {ma5:.2f}, 이평15: {ma15:.2f}" )
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

def is_sell_time_utc():     #Checks if the current time is between 23:59:00 and 23:59:50 UTC
    current_time_utc = datetime.now(UTC)
    sell_start = current_time_utc.replace(hour=23, minute=59, second=50, microsecond=0)      # Define the sell window (23:59:50 to 24:00:00 UTC)
    sell_end = current_time_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return sell_start <= current_time_utc <= sell_end

def trade_sell(ticker, buyed_amount, avg_buy_price):
    """주어진 티커에 대해 매도 실행 및 수익률 출력"""
    current_price = get_current_price(ticker)
    # evaluation_amount = buyed_amount * current_price  # 평가금액 계산
    current_time_utc = datetime.now(UTC)  # Get current time in UTC
    profit_rate = calculate_profit_rate(buyed_amount, avg_buy_price, ticker)       # 수익률 계산


    # if is_sell_time_utc():          # Check if we're within the special sell time frame (23:59:00 - 23:59:50 UTC)
    #     ai_decision = get_ai_decision(ticker)  
    #     sell_order = upbit.sell_market_order(ticker, buyed_amount)  # Market sell order
    #     if ai_decision != 'BUY' :  # AI의 판단이 NOT BUY이거나 수익률이 1.0%를 넘는 경우 매도
    #         sell_time = current_time_utc.strftime('%Y-%m-%d %H:%M:%S')  # Log sell time in UTC
    #         print(f"매도제한시간 : {sell_time}")
    #         send_slack_message('#api_test', f"Sold full balance at: {sell_time}, Ticker: {ticker}, price: {current_price}, Profit: {profit_rate:.2f}%")
    #         return sell_order
    
    if buyed_amount > 5000:  # 평가 금액이 5000 이상인 경우
        if ticker.split("-")[1] not in ["BTC", "ETH"]:
            if profit_rate > 0.65:  # 수익률 조건
                ai_decision = get_ai_decision(ticker)  
                if ai_decision != 'BUY' or profit_rate > 1.2:  # AI의 판단이 NOT BUY이거나 수익률이 1.0%를 넘는 경우 매도
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                    sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                    # print(f"매도: {sell_time}, Ticker: {ticker}, 현재가:{current_price}, 수익률: {profit_rate:.2f}%, AI판단: {ai_decision}")
                    send_slack_message('#api_test', f"매도: {sell_time}, Ticker: {ticker}, 현재가:{current_price}, 수익률: {profit_rate:.2f}%, AI판단: {ai_decision}")
                    return sell_order

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
            now = datetime.now()     # 현재 시간을 확인
            
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)   # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
            time_until_next_hour = (next_hour - now).total_seconds()
            time.sleep(time_until_next_hour)    # 다음 정시까지 기다림
            
            # hours, remainder = divmod(time_until_next_hour, 3600)       # 남은 시간을 시간, 분, 초로 변환
            # minutes, seconds = divmod(remainder, 60)
            # print(f"미국시간: {now}, 다음 정시: {next_hour}, 남은 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
            # send_slack_message('#api_test', f"현재 시간: {now}, 남은 시간: {int(minutes)}분 {int(seconds)}초")

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
                    try_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 보고시간
                    profit_rate = calculate_profit_rate(buyed_amount, avg_buy_price, ticker)    # 수익률 계산
                    report_message += f"{try_time} {b['currency']}: 현재가,{current_price}, 평균가 : {avg_buy_price}, 수익률 {profit_rate:.2f}%, ai판단:{ai_decision}"

            send_slack_message('#api_test', report_message)  # 슬랙으로 보고서 전송
        except Exception as e:            
            print(f"수익률 보고 중 오류 발생: {e}")
            send_slack_message('#api_test', f"수익률 보고 중 오류 발생: {e}")


# 자동매매 시작
print("trading start")

# 수익률 보고 쓰레드 시작
profit_report_thread = threading.Thread(target=send_profit_report)
profit_report_thread.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
profit_report_thread.start()

while True:
    current_time = datetime.now()  # 현재 시간 조회
    try:
        krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
        
        # 매수 제한 시간대 체크
        if is_within_restricted_time():
            print("현재 매수 제한 시간대입니다. 매수를 진행하지 않습니다.")
            result = trade_buy(best_ticker, best_k)  # 매수 실행
                
        if krw_balance < 5000:      # 잔고가 5천원 미만일 경우 보유 코인 매도
            sell_all_assets()  # 보유 자산 매도
            time.sleep(5)  # 5초 대기 후 재 실행
        

        else:  # 잔고가 5천원 이상일 경우
            best_ticker, interest, best_k = get_best_ticker()  # 최고의 코인 조회
            if best_ticker:  # 최고의 코인이 존재할 경우 매수시도
                # print(f"매수 시도 : best_ticker {best_ticker}, interest {interest:.2f}, best_k : {best_k:.2f}")
                send_slack_message('#api_test', f"매수 시도 : best_ticker {best_ticker}, interest: {interest:.2f}, best_k : {best_k:.2f}")
                result = trade_buy(best_ticker, best_k)  # 매수 실행
                time.sleep(360)  # 매수 후 잠시 대기
                
                sell_all_assets()  # 보유 자산 매도
        
        time.sleep(5)

    except Exception as e:
        print(f"에러 발생: {e}")
        send_slack_message('#api_test',f"에러 발생: {e}")
        time.sleep(1)