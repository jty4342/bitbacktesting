access_key = ''#엑세스키
secret_key = ''#시크릿키



import pyupbit
import numpy as np
import pandas as pd

import requests
import json
upbit = pyupbit.Upbit(access_key, secret_key)
def send_discord_webhook(webhook_url, message):
    data = {
        'content': message,
        'allowed_mentions': {
            'parse': []
        }
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
    if response.status_code != 204:
        print('Failed to send Discord webhook')
        print('Response:', response.text)

# Example usage
webhook_url = ''#디스코드등 주소 

tradable_assets = ['KRW-BTC']
trading_records = []

wrl=webhook_url


message = '```diff\n'
message += '# Upbit Trading Records\n\n'
for record in trading_records:
    message += f'## 거래일자: {record["timestamp"]}\n'
    message += f'[Symbol]: {record["symbol"]}\n'
    message += f'[가격]: {record["price"]}\n'
    message += f'[비중에 대한 수량]: {record["quantity"]}\n'
    message += '---\n'
message += '```'
#디스코드 웹훅실행
#send_discord_webhook(wrl, message)
#send_discord_webhook(wrl,"#매도 주문\n매도 주문 가격 : " ) #이것도 됨
def sma(close, period):
    return close.rolling(window=period).mean()

def macd(close, fast_period, slow_period, signal_period):
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line - signal_line
def stochastic_oscillator(data, high_col='high', low_col='low', close_col='close', k=14, d=3):
    data['L14'] = data[low_col].rolling(window=k).min()
    data['H14'] = data[high_col].rolling(window=k).max()

    data['%K'] = 100 * ((data[close_col] - data['L14']) / (data['H14'] - data['L14']))
    data['%D'] = data['%K'].rolling(window=d).mean()
def strategy7(data, atr_period=10, atr_multiplier=3):
    data['high_low'] = (data['high'] + data['low']) / 2
    data['ATR'] = data['high'].combine(data['low'], max) - data['low'].combine(data['high'].shift(1), min)
    data['ATR'] = data['ATR'].rolling(window=atr_period).mean()

    data['up'] = data['high_low'] - atr_multiplier * data['ATR']
    data['up'] = np.where(data['close'] > data['up'].shift(1), np.maximum(data['up'], data['up'].shift(1)), data['up'])
    data['dn'] = data['high_low'] + atr_multiplier * data['ATR']
    data['dn'] = np.where(data['close'] < data['dn'].shift(1), np.minimum(data['dn'], data['dn'].shift(1)), data['dn'])
    data['trend'] = np.where(data['close'] > data['up'].shift(1), 1, np.where(data['close'] < data['dn'].shift(1), -1, np.nan))
    data['trend'].fillna(method='ffill', inplace=True)
    
    data['signal'] = np.where((data['trend'] == 1) & (data['trend'].shift(1) == -1), 'Buy',
                              np.where((data['trend'] == -1) & (data['trend'].shift(1) == 1), 'Sell', ''))

    data.drop(columns=['up', 'dn'], inplace=True)

    return data


def calculate_smacd(data, close_col='close', short_window=12, long_window=26, signal_window=9):
        sma1 = data[close_col].rolling(window=short_window).mean()
        sma2 = data[close_col].rolling(window=long_window).mean()
        data['SMACD'] = sma1 - sma2
        data['SMACD_Signal'] = data['SMACD'].rolling(window=signal_window).mean()
def calculate_stochastic(data, high_col='high', low_col='low', close_col='close', k_period=14, d_period=3):
        data['L14'] = data[low_col].rolling(window=k_period).min()
        data['H14'] = data[high_col].rolling(window=k_period).max()
        data['%K'] = 100 * ((data[close_col] - data['L14']) / (data['H14'] - data['L14']))
        data['%D'] = data['%K'].rolling(window=d_period).mean()
def calculate_atr(data, high_col='high', low_col='low', close_col='close', window=14):
    hl = data[high_col] - data[low_col]
    hc = (data[high_col] - data[close_col]).abs()
    lc = (data[low_col] - data[close_col]).abs()

    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=window).mean()
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    delta = delta[1:]  # Remove the first NaN value
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
class backTesting :

    def __init__(self, daily_data, start_cash) :
        self.data = daily_data
        self.daily_data = daily_data
        self.fee = 0.0011
        self.buy_signal = False
        self.start_cash = start_cash
        self.current_cash = start_cash
        self.highest_cash = start_cash
        self.lowest_cash = start_cash
        self.ror = 1
        self.accumulated_ror = 1
        self.mdd = 0
        self.trade_count = 0
        self.win_count = 0
        self.holding = False
        self.buy_price = 0
        self.target = 0.04
        self.volatility_breakout_ratio = 0.5
        self.target_profit_ratio = 1.03
        self.short_window = 5
        self.long_window = 20
        self.total_buy_count = 0
        self.strategy1_trade_count = 0
        self.strategy1_win_count = 0
        self.strategy2_trade_count = 0
        self.strategy2_win_count = 0
        self.strategy3_trade_count = 0
        self.strategy3_win_count = 0
        self.daily_data.reset_index(drop=True, inplace=True)
        self.total_games = {'strategy1': 0, 'strategy2': 0, 'strategy3': 0, 'strategy4': 0, 'strategy5': 0,
                            'strategy6': 0,'strategy7': 0,'strategy8' :0}
        self.wins = {"strategy1": 0, "strategy2": 0, "strategy3": 0, "strategy4": 0, "strategy5": 0, "strategy6": 0,"strategy7": 0,"strategy8" :0}
        self.current_strategy = None
    
    def execute(self) : #전략 실행하는 부분 
       #한국은 이동평균선 전략은 안좋음
        #볼랜저 밴드 +RSI 전략
        
        self.current_strategy = "strategy1"
        self.strategy1()
        self.current_strategy = None
        #음봉 2개 일때 구매
        self.current_strategy = "strategy2"
        self.strategy2()
        self.current_strategy = None
        #변동성 돌파 + 이동 평균 전략
        self.current_strategy = "strategy3"
        self.strategy3()
        self.current_strategy = None
        
        self.current_strategy = "strategy4"
        self.strategy4()
        self.current_strategy = None
        
        self.current_strategy = "strategy5"
        self.strategy5()
        self.current_strategy = None
        
        self.current_strategy = "strategy6"
        self.strategy6()
        self.current_strategy = None


        
        
        self.current_strategy = "strategy7"
        self.strategy7()
        self.current_strategy = None

        self.current_strategy = "strategy8"
        self.strategy8()
        self.current_strategy = None
        
        



        self.result()
#볼랜저 밴드 +RSI 전략 승률 좋음 100 근데 거래횟수가 적음
    def strategy1(self):
        self.daily_data['BB_Middle'] = self.daily_data['close'].rolling(window=20).mean()
        self.daily_data['BB_Up'] = self.daily_data['BB_Middle'] + 3 * self.daily_data['close'].rolling(window=20).std()
        self.daily_data['BB_Down'] = self.daily_data['BB_Middle'] - 1.5 * self.daily_data['close'].rolling(window=20).std()

        delta = self.daily_data['close'].diff()
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.daily_data['RSI'] = 100 - (100 / (1 + rs))

        self.daily_data = self.daily_data.dropna().reset_index(drop=True)

        for idx in self.daily_data.index[1:]:
            self.current_index = idx
            cur_close = self.daily_data.at[idx, 'close']

            if not self.holding and len(self.daily_data['close'][:idx]) >= 3:
                last_3_closes = self.daily_data['close'][idx-3:idx]
                if all(last_3_closes == last_3_closes.sort_values(ascending=False).values):
                    # Check RSI confirmation signal
                    rsi_confirmation = self.daily_data.at[idx, 'RSI']
                    if rsi_confirmation > 50:  # Modify the RSI threshold as needed
                        self.buy()
                        self.strategy1_trade_count += 1
                    # Update statistics data: add number of transactions
                    self.total_games["strategy1"] += 1

            elif self.holding and self.daily_data.at[idx, 'high'] >= self.buy_price * (1 + self.target):
                # Check RSI confirmation signal
                rsi_confirmation = self.daily_data.at[idx, 'RSI']
                if rsi_confirmation < 50:  # Modify the RSI threshold as needed
                    self.sell()
                    self.ror = cur_close * (1 - self.fee) / self.buy_price
                    self.accumulated_ror *= self.ror
                    # Updated stats data: added number of wins and conditions
                    if self.ror > 1:
                        self.strategy1_win_count += 1
                        self.wins["strategy1"] += 1

            

    def strategy2(self):
        #음봉 2개 일때 구매
        buy_price = 0
        is_prev_red = False
        for idx in range(2, len(self.daily_data)):
            cur_close = self.daily_data.at[idx, 'close']
            prev_close = self.daily_data.at[idx - 1, 'close']

            if pd.isna(cur_close) or pd.isna(prev_close):
                continue

            is_red = cur_close < prev_close
            if not self.holding and is_prev_red and is_red:
                self.holding = True
                buy_price = cur_close * (1 + self.fee)
                self.strategy2_trade_count += 1
                self.total_games["strategy2"] += 1

            elif self.holding and cur_close > self.daily_data.at[idx, 'BB_Up']:
                self.holding = False
                if buy_price > 0:
                    self.ror = cur_close * (1 - self.fee) / buy_price
                else:
                    self.ror = 1
                    

                self.win_count += 1 if self.ror > 1 else 0
                self.accumulated_ror *= self.ror
                self.current_cash *= self.ror
                self.highest_cash = max(self.highest_cash, self.current_cash)
                self.lowest_cash = min(self.lowest_cash, self.current_cash)
                dd = (self.highest_cash - self.current_cash) / self.highest_cash * 100
                self.mdd = max(self.mdd, dd)

                # 통계 데이터 업데이트: 승리 횟수 및 조건 추가
                if self.ror > 1:
                    self.strategy2_win_count += 1
                    self.wins["strategy2"] += 1

            is_prev_red = is_red
        # 음봉 3개 일때 구매
        # buy_price = 0
        # is_prev2_red = False
        # is_prev_red = False
        # for idx in range(3, len(self.daily_data)): # 시작 인덱스를 3으로 변경
        #     cur_close = self.daily_data.at[idx, 'close']
        #     prev_close = self.daily_data.at[idx - 1, 'close']

        #     if pd.isna(cur_close) or pd.isna(prev_close):
        #         continue

        #     is_red = cur_close < prev_close
        #     if not self.holding and is_prev2_red and is_prev_red and is_red:
        #         self.holding = True
        #         buy_price = cur_close * (1 + self.fee)
        #         self.strategy2_trade_count += 1
        #         self.total_games["strategy2"] += 1

        #     elif self.holding and cur_close > self.daily_data.at[idx, 'BB_Up']:
        #         self.holding = False
        #         if buy_price > 0:
        #             self.ror = cur_close * (1 - self.fee) / buy_price
        #         else:
        #             self.ror = 1

        #         self.win_count += 1 if self.ror > 1 else 0
        #         self.accumulated_ror *= self.ror
        #         self.current_cash *= self.ror
        #         self.highest_cash = max(self.highest_cash,  self.current_cash)
        #         self.lowest_cash = min(self.lowest_cash, self.current_cash)
        #         dd = (self.highest_cash - self.current_cash) / self.highest_cash * 100
        #         self.mdd = max(self.mdd, dd)

        #         # 통계 데이터 업데이트: 승리 횟수 및 조건 추가
        #         if self.ror > 1:
        #             self.strategy2_win_count += 1
        #             self.wins["strategy2"] += 1

        #     is_prev2_red = is_prev_red
        #     is_prev_red = is_red


    def strategy3(self):
        #양방향 변동성 돌파전략 =거래량이없음
        
        #변동성 돌파 + 이동 평균 전략
        self.daily_data['Short_MA'] = self.daily_data['close'].rolling(window=self.short_window).mean()
        self.daily_data['Long_MA'] = self.daily_data['close'].rolling(window=self.long_window).mean()
        self.daily_data['Range'] = self.daily_data['high'] - self.daily_data['low']
        self.daily_data['Buy_Target'] = self.daily_data['open'] + self.daily_data['Range'].shift(1) * self.volatility_breakout_ratio
        self.daily_data['Sell_Target'] = self.daily_data['open'] + self.daily_data['Range'].shift(1) * (1 - self.volatility_breakout_ratio)
        self.daily_data = self.daily_data.dropna().reset_index(drop=True)

        for idx in range(1, len(self.daily_data)):
            cur_short_ma = self.daily_data.at[idx, 'Short_MA']
            cur_long_ma = self.daily_data.at[idx, 'Long_MA']

            cur_buy_target = self.daily_data.at[idx, 'Buy_Target']
            cur_sell_target = self.daily_data.at[idx, 'Sell_Target']
            cur_open = self.daily_data.at[idx, 'open']
            cur_close = self.daily_data.at[idx, 'close']

            downtrend = cur_long_ma < cur_short_ma

            if not self.holding and cur_open < cur_buy_target < cur_close:
                self.holding = True
                self.buy_price = cur_buy_target
                self.strategy3_trade_count += 1
                # 통계 데이터 업데이트: 거래수 추가
                self.total_games["strategy3"] += 1

            elif self.holding and (cur_open < self.buy_price * self.target_profit_ratio < cur_close or cur_open < cur_sell_target < cur_close):
                self.holding = False
                self.ror = cur_sell_target / self.buy_price

                if self.ror > 1:
                    self.strategy3_win_count += 1
                    self.wins["strategy3"] += 1
    
    # 골든크로스 전략 +RSI
    def strategy4(self):
        window1 = 5
        window2 = 20
        rsi_window = 21
        rsi_upper = 75 #70
        rsi_lower = 25 #30

        self.daily_data['Volume_MA_Short'] = self.daily_data['volume'].rolling(window=window1).mean()
        self.daily_data['Volume_MA_Long'] = self.daily_data['volume'].rolling(window=window2).mean()

        # RSI 지표 계산
        calculate_rsi(self.daily_data, window=rsi_window)

        for idx in self.daily_data.index[window2 + 1:]:
            self.current_index = idx
            if not self.holding and self.daily_data.at[self.current_index - 1, 'Volume_MA_Short'] < self.daily_data.at[self.current_index - 1, 'Volume_MA_Long'] and\
            self.daily_data.at[self.current_index, 'Volume_MA_Short'] >= self.daily_data.at[self.current_index, 'Volume_MA_Long'] and\
            self.daily_data.at[self.current_index, 'RSI'] < rsi_lower:  # RSI 가격이 과매도 구간일 때 추가
                self.buy()

            elif self.holding and (self.daily_data.at[self.current_index, 'Volume_MA_Short'] < self.daily_data.at[self.current_index, 'Volume_MA_Long'] or\
            self.daily_data.at[self.current_index, 'RSI'] > rsi_upper):  # RSI 가격이 과매수 구간일 때 추가
                self.sell()
    #MACD + 스토캐스틱 지표
    def strategy5(self):
        '''short_window = 12
        long_window = 26
        signal_window = 9
    
        exp12 = self.daily_data['close'].ewm(span=short_window).mean()
        exp26 = self.daily_data['close'].ewm(span=long_window).mean()
        self.daily_data['MACD'] = exp12 - exp26
        self.daily_data['Signal'] = self.daily_data['MACD'].ewm(span=signal_window).mean()
        self.daily_data.dropna(inplace=True)
        
        for idx in self.daily_data.index[signal_window + 1:]:
            self.current_index = idx
            if not self.holding and self.daily_data.at[idx - 1, 'MACD'] < self.daily_data.at[idx - 1, 'Signal'] and\
            self.daily_data.at[idx, 'MACD'] >= self.daily_data.at[idx, 'Signal']:
                self.buy()
            
            elif self.holding and self.daily_data.at[idx, 'MACD'] < self.daily_data.at[idx, 'Signal']:
                self.sell()'''
        #macd
        short_window = 28
        long_window = 27
        signal_window = 9
        k_period = 28
        d_period = 3

        # 스토캐스틱 지표 계산
        calculate_stochastic(self.daily_data, k_period=k_period, d_period=d_period)

        exp12 = self.daily_data['close'].ewm(span=short_window).mean()
        exp26 = self.daily_data['close'].ewm(span=long_window).mean()
        self.daily_data['MACD'] = exp12 - exp26
        self.daily_data['Signal'] = self.daily_data['MACD'].ewm(span=signal_window).mean()
        self.daily_data.dropna(inplace=True)
        
        for idx in self.daily_data.index[k_period + d_period + 1:]:
            self.current_index = idx

            # 매수 조건: MACD가 Signal선을 아래서 위로 통과하고 있으며, %K가 %D (스토캐스틱 선들)를 아래서 위로 통과하는 경우
            if not self.holding and \
            self.daily_data.at[idx - 1, 'MACD'] < self.daily_data.at[idx - 1, 'Signal'] and \
            self.daily_data.at[idx, 'MACD'] >= self.daily_data.at[idx, 'Signal'] and \
            self.daily_data.at[idx - 1, '%K'] < self.daily_data.at[idx - 1, '%D'] and \
            self.daily_data.at[idx, '%K'] >= self.daily_data.at[idx, '%D']:
                self.buy()

            # 매도 조건: MACD가 Signal선 밑으로 내려오면 매도
            elif self.holding and self.daily_data.at[idx, 'MACD'] < self.daily_data.at[idx, 'Signal']:
                self.sell()

    #OBV 전략 On-Balance Volume)를 이용한 이동 평균 전략  거래량 바탕 추세감지 매매
    def strategy6(self):
        short_window = 20
        long_window = 50

        rsi_window = 21
        rsi_upper = 70
        rsi_lower = 30

        obv = [0]
        for idx in self.daily_data.index[1:]:
            prev_idx = self.daily_data.index.get_loc(idx) - 1
            prev_close = self.daily_data.iloc[prev_idx]['close']
            obv.append(obv[-1] + (self.daily_data.at[idx, 'close'] - prev_close))
        self.daily_data['OBV'] = obv
        self.daily_data['Short_MA'] = self.daily_data['OBV'].rolling(window=short_window).mean()
        self.daily_data['Long_MA'] = self.daily_data['OBV'].rolling(window=long_window).mean()

        # RSI 지표 계산
        calculate_rsi(self.daily_data, window=rsi_window)

        for idx in self.daily_data.index[long_window + 1:]:
            self.current_index = idx
            prev_idx = self.daily_data.index.get_loc(idx) - 1
            if not self.holding and \
                self.daily_data.at[prev_idx, 'Short_MA'] <= self.daily_data.at[prev_idx, 'Long_MA'] and \
                self.daily_data.at[idx, 'Short_MA'] >= self.daily_data.at[idx, 'Long_MA'] and \
                self.daily_data.at[idx, 'RSI'] < rsi_lower:  # RSI 가격이 과매도 구간일 때 추가
                self.buy()

            elif self.holding and (self.daily_data.at[idx, 'Short_MA'] < self.daily_data.at[idx, 'Long_MA'] or\
            self.daily_data.at[idx, 'RSI'] > rsi_upper):  # RSI 가격이 과매수 구간일 때 추가
                self.sell()
    #이동평균선 + RSI
    def strategy7(self):
        short_sma = 5   # 단기 이동 평균 기간
        long_sma = 20   # 장기 이동 평균 기간
        rsi_window = 14 # RSI 기간
        oversold_level = 30   # 과매도 구간 수준
        overbought_level = 70 # 과매수 구간 수준
        long_window = 14
        
        # 이동평균 계산
        self.data['SMA_short'] = self.data['close'].rolling(window=short_sma).mean()
        self.data['SMA_long'] = self.data['close'].rolling(window=long_sma).mean()

        # 이동평균 교차 구하기
        self.data['cross'] = np.where(self.data['SMA_short'] > self.data['SMA_long'], 1, np.where(self.data['SMA_short'] < self.data['SMA_long'], -1, 0))
        self.data['cross_position'] = self.data['cross'].replace(0, np.nan).ffill()
        
        # RSI 지표 추가
        calculate_rsi(self.data, window=rsi_window)
        if len(self.data.index) < long_window:  # Check if there is enough data
            print("Strategy7: Not enough data")
            return False, False
        last_row = self.data.iloc[-1]
        if not self.holding and last_row['cross_position'] > 0 and last_row['RSI'] < oversold_level:
              buy_signal = True
              self.buy()
              self.total_games["strategy7"] += 1
        else:
              buy_signal = False

        if self.holding:
            if last_row['RSI'] > overbought_level:
                sell_signal = True
                cur_sell_target = last_row['close']
            elif last_row['cross_position'] < 0:
                sell_signal = True
                cur_sell_target = last_row['close']
            else:
                sell_signal = False

            if sell_signal:
                self.sell()

                self.ror = cur_sell_target / self.buy_price

                # 통계 데이터 업데이트
                if self.ror > 1:
                    self.strategy7_win_count += 1
                    self.wins["strategy7"] += 1
              
        else:
              sell_signal = False

        return buy_signal, sell_signal
        # 주식 거래 로직
        # for idx in range(1, len(self.data)):
        #     # 매수 조건
        #     if not self.holding and self.data.at[idx, 'cross_position'] > 0 and self.data.at[idx, 'RSI'] < oversold_level:
        #         self.buy()

        #     # 매도 조건
        #     elif self.holding and (self.data.at[idx, 'cross_position'] < 0 or self.data.at[idx, 'RSI'] > overbought_level):
        #         self.sell()

        

    #슈퍼?트렌드  이 전략은 안좋은듯.
    '''def strategy7(self, trailing_stop=0.02, k_period=14, d_period=3, oversold_level=20, overbought_level=80, short_sma=5, long_sma=20):
        
        self.data = strategy7(self.data)  
        self.data['position'] = self.data['trend'].shift(1)

        # 기존 코드 유지
        self.data['SMA_short'] = self.data['close'].rolling(window=short_sma).mean()
        self.data['SMA_long'] = self.data['close'].rolling(window=long_sma).mean()
        self.data['volume_SMA_3'] = self.data['volume'].rolling(window=5).mean()

        # 이동 평균 교차 구하기
        self.data['cross'] = np.where(self.data['SMA_short'] > self.data['SMA_long'], 1, np.where(self.data['SMA_short'] < self.data['SMA_long'], -1, 0))
        self.data['cross_position'] = self.data['cross'].replace(0, np.nan).ffill()
        
        # Stochastic Oscillator 지표 추가
        stochastic_oscillator(self.data, k=k_period, d=d_period)

        # Trailing stop 추가
        trailing_sell_price = None
        for idx in range(1, len(self.data)):
            if self.data.at[idx, 'volume'] >= self.data.at[idx, 'volume_SMA_3']:
                # 매수 조건
                if not self.holding and self.data.at[idx, 'cross_position'] > 0 and self.data.at[idx, '%K'] <= oversold_level and self.data.at[idx, '%K'] < self.data.at[idx, '%D']:
                    self.buy()
                    trailing_sell_price = self.data.at[idx, 'close'] * (1 - trailing_stop)

                # 매도 조건
                elif self.holding:
                    if (self.data.at[idx, 'cross_position'] < 0) or (self.data.at[idx, '%K'] >= overbought_level and self.data.at[idx, '%K'] > self.data.at[idx, '%D']):
                        self.sell()
                        trailing_sell_price = None
                    elif self.data.at[idx, 'close'] * (1 - trailing_stop) > trailing_sell_price:
                        trailing_sell_price = self.data.at[idx, 'close'] * (1 - trailing_stop)
                    elif self.data.at[idx, 'close'] <= trailing_sell_price:
                        self.sell()
                        trailing_sell_price = None'''
    #상승다이버전스 RSI + 변동성돌파전략
    def strategy8(self):
        ticker = "KRW-BTC"
        df = pyupbit.get_ohlcv(ticker, interval="minute30", count=1000)
        
        rsi_period = 14
        ma_period = 20
        
        inputs = {
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        }
        
        # Calculate RSI
        df['RSI'] = calculate_rsi(df['close'], window=rsi_period)
        
        # Compute upward divergence (using MACD)
        
        df['MACD'] = macd(df['close'], 12, 26, 9)

        # Volatility breakout strategy (moving average calculation)
        df['MA'] = sma(df['close'], ma_period)

        # Compare last data price and moving average
        price = float(df.iloc[-1]['close'])
        ma_prev = float(df.iloc[-2]['MA'])
        ma_cur = float(df.iloc[-1]['MA'])
        
        rsi_prev = float(df.iloc[-2]['RSI'])
        rsi_cur = float(df.iloc[-1]['RSI'])
        
        macd_prev = float(df.iloc[-3]['MACD'])
        macd_cur = float(df.iloc[-2]['MACD'])
        macd_next = float(df.iloc[-1]['MACD'])

        # buy conditions
        if self.holding and (price > ma_cur > ma_prev) and (rsi_cur < 30) and (macd_prev < 0 < macd_cur) and (macd_next > macd_cur):
            buy_signal = True
            self.buy()
            self.total_games["strategy8"] += 1
        else:
            buy_signal = False

        # Sell condition
        if self.holding:
            if (rsi_cur > 70) and ((macd_next < macd_cur) or (price < ma_cur)):
                sell_signal = True
                cur_sell_target = price
            else:
                sell_signal = False

            if sell_signal:
                self.sell()

                ror = cur_sell_target / self.buy_price

                # update statistics data
                if ror > 1:
                    self.strategy8_win_count += 1
                    self.wins["strategy8"] += 1
        else:
            sell_signal = False

        return buy_signal, sell_signal
    #def strategy9(self):

    def buy(self):
        self.holding = True
        self.buy_price = self.daily_data.at[self.current_index, 'close']
        self.strategy_used = self.current_strategy
        self.total_games[self.current_strategy] += 1

    def sell(self):
        sell_price = self.daily_data.at[self.current_index, 'close']
        if sell_price > self.buy_price:
            self.wins[self.strategy_used] += 1
        self.holding = False

    

    def result(self) :
        total_games = sum(self.total_games.values())
        total_wins = sum(self.wins.values())
        win_rate = total_wins / total_games * 100 if total_games > 0 else 0
        print('='*40)
        print('테스트 결과')
        print('-'*40)
        print(f'총 거래 횟수: {total_games}')
        print(f'승리 횟수: {total_wins}')
        print(f'승률: {win_rate:.2f}%')
        print(f'누적 수익률: {(self.accumulated_ror - 1) * 100:.2f}%')
        print(f'현재 잔액: {self.current_cash}')
        print(f'최고 잔액: {self.highest_cash}')
        print(f'최저 잔액: {self.lowest_cash}')
        print(f'최대 낙폭 (MDD): {self.mdd}')
        print('-'*40)
        print('전략별 통계')
        
        for strategy in self.total_games.keys():
            strategy_win_rate = self.wins[strategy] / self.total_games[strategy] * 100 if self.total_games[strategy] > 0 else 0
            print(f"{strategy}: 승률 {strategy_win_rate:.2f}% (게임 수: {self.total_games[strategy]}), 승리: {self.wins[strategy]})")
        print('=' * 40)
        
        

df = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", count=5000) 
backtest = backTesting(df, 1000000)
backtest.execute()
