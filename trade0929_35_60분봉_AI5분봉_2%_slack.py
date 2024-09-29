import time
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import slack_sdk

"""
load_ohlcv :  30개의 60분봉 데이터프레임 
AI_decision : 60개의  5분봉 데이터프레임 
trade_buy : 변동성돌파전략 적용, 수익률 2%~5%, 8~9:15 매매 금지 설정 삭제
trade_sell : 손절 5%적용
while try : 8~9:30 매매 금지 설정 추가
"""

load_dotenv()

slack_token = os.getenv("slack_token")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
client = slack_sdk.WebClient(token=slack_token)

df_tickers = {}    # 전역변수: 최근 50개의 일봉 데이터프레임

def load_ohlcv(ticker):
    """특정 티커의 3분봉 데이터 로드"""
    global df_tickers
    if ticker not in df_tickers:
        df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute60", count=30)  # 30개의 60분봉 데이터 가져오기
    return df_tickers[ticker]

def get_balance(ticker):
    """잔고 조회"""
    try:
        balances = upbit.get_balances()  # 모든 잔고 조회
        for b in balances:
            if b['currency'] == ticker:  # 요청한 통화의 잔고를 찾음
                return float(b['balance']) if b['balance'] is not None else 0
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
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

def get_ma15(ticker):
    """15일 이동 평균선 조회"""
    df = load_ohlcv(ticker)  # 데이터 로드
    if df is not None and not df.empty:
        ma15 = df['close'].rolling(15).mean().iloc[-1]  
        return ma15
    return 0  # 데이터가 없을 경우 0 반환

def get_best_k(coin="KRW-BTC"):
    """최적의 K 값 계산"""
    bestK = 0.5  # 초기 K 값
    interest = 0  # 초기 수익률

    df = load_ohlcv(coin)  # 데이터 로드
    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환

    sufficient_data = False  # 데이터가 충분한지 여부

    for k in np.arange(0.1, 1.0, 0.1):  # K 값을 0.1부터 0.9까지 반복
        # 변동성 계산
        df['range'] = (df['high'] - df['low']) * k
        df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정

        fee = 0.0005  # 거래 수수료 (0.05%로 설정)

        # 수익률 계산
        df['ror'] = (df['close'] / df['target'] - fee).where(df['high'] > df['target'], 1)
        ror_series = df['ror'].cumprod()  # 누적 수익률
        
        if len(ror_series) < 2:  # 데이터가 부족한 경우
            continue
        
        # 마지막 이전 값
        ror = ror_series.iloc[-2]  
        
        if ror > interest:  # 이전 수익률보다 높으면 업데이트
            interest = ror
            bestK = k
            sufficient_data = True  # 데이터가 충분하다고 표시

    if not sufficient_data:  # 데이터가 부족할 경우 초기 K 반환
        return bestK
        
    return bestK  # 최적 K 값을 반환

from collections import defaultdict

def get_best_ticker():
    """최고의 코인과 그에 대한 수익률을 찾습니다."""
    tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인을 조회합니다.
    held_coins = {b['currency'] for b in upbit.get_balances() if float(b['balance']) > 0}  # 보유하고 있는 코인 목록

    # 최근 3시간 동안 최고의 코인 기록
    recent_best_coins = defaultdict(lambda: (0, None))  # 코인과 선정 횟수를 저장할 딕셔너리

    # 처음 필터링된 코인 리스트
    filtered_tickers = []

    # 조회할 코인 필터링
    for t in tickers:
        currency = t.split("-")[1]  # 티커에서 통화 정보를 추출합니다.

        if currency in ["BTC", "ETH"] or currency in held_coins:
            continue  # BTC, ETH 제외 및 보유한 코인인 경우 건너뛰기

        df = load_ohlcv(t)  # 각 코인에 대해 데이터 로드          
        if df is None or df.empty:
            continue  # 데이터가 없으면 다음 코인으로 넘어갑니다.

        # 수익률 계산
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률

        if len(df['hpr']) >= 2 and df['hpr'].iloc[-1] > 1.1:  # 수익률이 10% 초과하는지 확인
            continue  # 10% 초과 시 건너뛰기

        filtered_tickers.append(t)  # 필터링된 코인 리스트에 추가합니다.

    # 최고의 코인 찾기 반복
    while filtered_tickers:
        bestC = None  # 매 반복마다 초기화
        interest = 0  # 매 반복마다 초기화
        best_k = 0.5  # 매 반복마다 초기화

        for t in filtered_tickers:
            k = get_best_k(t)  # 최적의 K 값을 계산합니다.
            df = load_ohlcv(t)  # 데이터를 가져옵니다.

            if df is None or df.empty:
                continue
            
            df['range'] = (df['high'] - df['low']) * k  # 변동성 계산
            df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정
            
            # 수익률 계산
            df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산
            df['hpr'] = df['ror'].cumprod()  # 누적 수익률

            if len(df['hpr']) < 2:  # 데이터가 부족한 경우
                continue
            
            if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으면 업데이트
                bestC = t
                interest = df['hpr'].iloc[-1]
                best_k = k  # 최적 K 값도 업데이트

            time.sleep(1)  # API 호출 제한을 위한 대기

        # AI 판단을 구함
        if bestC:
            ai_decision = get_ai_decision(bestC)  # AI의 판단을 구함
            print(f"판단전 최고의 코인: {bestC}, AI의 판단: {ai_decision}")
            # client.chat_postMessage(channel='#api_test', text=str(f"판단전 최고의 코인: {bestC}, AI의 판단: {ai_decision}")) 

            if ai_decision != 'BUY':
                print(f"{bestC}에 대한 AI의 판단: {ai_decision}, 해당 코인은 제외합니다.")
                # client.chat_postMessage(channel='#api_test', text=str(f"{bestC}에 대한 AI의 판단: {ai_decision}, 해당 코인은 제외합니다.")) 
                filtered_tickers.remove(bestC)  # AI 판단이 'BUY'가 아닐 경우 해당 코인을 리스트에서 제거
                continue  # 다음 반복으로 이동

        if bestC is not None:  # 유효한 최고의 코인이 있을 경우
            # 최근 3시간 동안 5번 이상 선정된 코인 목록 확인 및 업데이트
            count, last_time = recent_best_coins[bestC]
            recent_best_coins[bestC] = (count + 1, datetime.now())  # 카운트 및 시간 업데이트

            if count >= 5 and last_time is not None and (datetime.now() - last_time) <= timedelta(hours=3):
                print(f"{bestC} 코인 선정횟수 : {count}, 해당 코인은 제외합니다.")
                continue  # 최근 3시간 동안 5회 이상 선정된 코인은 제외
            
            return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환

    return None, 0, 0  # 모든 코인을 확인했지만 유효한 코인이 없을 경우


def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = load_ohlcv(ticker)  # 데이터 로드
    if df is not None and not df.empty and len(df) > 0:
        # 첫 번째 행의 데이터가 존재할 경우에만 계산
        close_price = df.iloc[0]['close']
        high_price = df.iloc[0]['high']
        low_price = df.iloc[0]['low']
        
        target_price = close_price + (high_price - low_price) * k  # 목표가 계산
        return target_price
    return 0  # 데이터가 없거나 유효하지 않은 경우 0 반환


def get_ai_decision(ticker):
    """AI에게 매매 결정을 요청하는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=60)    # 60개의 5분봉 사용

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
                "text": "Based on the chart data provided, tell us whether you're currently buying, selling, or holding."
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
        
        return None  # 오류 발생 시 None 반환
    
    decision_data = response.choices[0].message.content      # 응답에서 필요한 정보만 추출

    if decision_data:
        try:
            decision_json = json.loads(decision_data)
            decision = decision_json.get('decision')
            if decision in {'BUY', 'HOLD', 'SELL'}:
                return decision
        except json.JSONDecodeError:
            print("응답을 JSON으로 파싱하는 데 실패했습니다.")
    
    print("유효하지 않은 응답입니다.")
    return None  # 유효하지 않은 경우 None 반환

def trade_buy(ticker, k):
    """주어진 티커에 대해 매수 실행"""
    current_price = get_current_price(ticker) 
    target_price = get_target_price(ticker, k)
    ma15 = get_ma15(ticker)
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1])

    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 100000 :  # 매수 조건 확인
        if target_price <= current_price and ma15 < current_price:
    
            ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
            print(f"최고의 코인: {ticker}, AI의 판단: {ai_decision}")
            if ai_decision == 'BUY':
                try:
                    buy_order = upbit.buy_market_order(ticker, 100000)
                    buy_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"매수 시간: {buy_time}, AI의 판단: {ai_decision}, Ticker: {ticker}, 현재가: {current_price}")
                    client.chat_postMessage(channel='#api_test', text=str(f"매수 시간: {buy_time}, AI의 판단: {ai_decision}, Ticker: {ticker}, 현재가: {current_price}")) 
                    return buy_order['price'], target_price
                    
                
                except Exception as e:
                    print(f"매수 주문 실행 중 오류 발생: {e}")
                    client.chat_postMessage(channel='#api_test', text=str(f"매수 주문 실행 중 오류 발생: {e}")) 
                    return "Buy order failed", None

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

def trade_sell(ticker):
    """주어진 티커에 대해 매도 실행 및 수익률 출력"""
    current_price = get_current_price(ticker)
    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    avg_buy_price = upbit.get_avg_buy_price(currency)

    evaluation_amount = buyed_amount * current_price  # 평가금액 계산

    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    
    if evaluation_amount > 5000 :  # 평가 금액이 5000 이상인 경우에만 매도 조건 체크
        if currency not in ["BTC", "ETH"]:
        
            if profit_rate > 2:  # 2% 이상 수익률일 때 AI의 판단을 구함
                ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
            
                if ai_decision != 'BUY' or profit_rate > 5:  # AI의 판단이 BUY가 아니거나 수익률이 5%를 넘는 경우 매도
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                    sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                    print(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (수익률 5% 초과 또는 AI 판단 SELL)")
                    client.chat_postMessage(channel='#api_test', text=str(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (수익률 5% 초과 또는 AI 판단 SELL)")) 
                    return sell_order

                sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시도 시간 기록
                print(f"매도 시도 시간: {sell_time}, 코인명 : {currency}, AI의 판단: {ai_decision}, 수익률: {profit_rate:.2f}%: 매도 조건을 충족하지 않았습니다.")


            if profit_rate < -5:  # -5% 이하 손실 시 AI 판단 후 매도
                ai_decision = get_ai_decision(ticker)  # AI의 판단을 구함
                
                if ai_decision == 'SELL':
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                    sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    print(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (-5% 이하)")
                    client.chat_postMessage(channel='#api_test', text=str(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}% (-5% 이하)"))
                    return sell_order

                sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시도 시간 기록
                print(f"매도 시도 시간: {sell_time}, 코인명 : {currency}, AI의 판단: {ai_decision}, 수익률: {profit_rate:.2f}%: 매도 조건을 충족하지 않았습니다.")
                client.chat_postMessage(channel='#api_test', text=str(f"매도 시도 시간: {sell_time}, 코인명 : {currency}, AI의 판단: {ai_decision}, 수익률: {profit_rate:.2f}%: 매도 조건을 충족하지 않았습니다."))

            current_time = datetime.now()   # 매도3 조건 설정 : 특정 시간에 전량 매도
            start_time = get_start_time("KRW-BTC")
            end_time = start_time + timedelta(days=1)

            if end_time - timedelta(seconds=120) < current_time < end_time - timedelta(seconds=30):
                sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 매도
                sell_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 매도 시간 기록
                
                print(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}%")
                client.chat_postMessage(channel='#api_test', text=str(f"매도 시간: {sell_time}, Ticker: {ticker}, 수익률: {profit_rate:.2f}%"))
                return sell_order

# 자동매매 시작
print("자동 매매 시작")

while True:
    try:
        current_time = datetime.now()  # 현재 시간 조회
        krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회

        if krw_balance < 100000:
            # print("현금 잔고 10만원 미만/매도프로세스 진행 확인용")
            # client.chat_postMessage(channel='#api_test',
            #             text='현금 잔고가 10만원미만/매도프로세스 진행 확인용')
            """매도 로직 실행 (모든 보유 코인 대상 매도로직 실행)"""
            balances = upbit.get_balances()  # 모든 보유 코인 조회
            for b in balances:
                if b['currency'] not in ["KRW", "AUCTION", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 보유 잔고가 있는 경우
                    ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                    trade_sell(ticker)  # 매도 실행

            time.sleep(10)  # 10초 대기 후 재 실행

        else:
            """ 08:00 ~ 09:15 사이에 매수하지 않음 """
            if (current_time.hour == 8 and 0 <= current_time.minute < 60) or (current_time.hour == 9 and current_time.minute < 30):
                not_buying_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"지금시각: {not_buying_time}, 장 시작 전후 90분간 매수 보류합니다.")
                client.chat_postMessage(channel='#api_test', text=str(f"지금시각: {not_buying_time}, 장 시작 전후 90분간 매수 보류합니다."))
                time.sleep(10)  # 10초 대기 후 재실행
                continue  # 매수 로직을 건너뜁니다.

            """ 현금잔고 10만원 이상인 경우 다시 매수 로직 실행 (10초 대기 후 재실행) """
            best_ticker, interest, best_k = get_best_ticker()  # 최고의 코인 및 K 값 조회
            target_price = get_target_price(best_ticker, best_k)  # 목표가 조회
            current_price = get_current_price(best_ticker)  # 현재가 조회
            # print("현금 잔고가 10만원 이상입니다. 매수를 실행합니다.")
            # client.chat_postMessage(channel='#api_test',
            #             text='현금 잔고가 10만원 이상입니다. 매수를 실행합니다.')

            # trade_buy 함수 호출
            result = trade_buy(best_ticker, best_k)  # 매수 실행

            # 매도 로직 실행 
            balances = upbit.get_balances()  # 모든 보유 코인 조회
            for b in balances:
                if b['currency'] not in ["KRW", "AUCTION", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 보유 잔고가 있는 경우
                    ticker = f"KRW-{b['currency']}"  # 티커 형식 맞추기
                    # print("매도프로세스 진행 확인용.")
                    trade_sell(ticker)  # 매도 실행

            time.sleep(10)  

    except Exception as e:
        print(f"에러 발생: {e}")  # 발생하는 에러 출력
        client.chat_postMessage(channel='#api_test', text=str(f"에러 발생: {e}"))
        time.sleep(10)  # 에러 발생 시 10초 대기