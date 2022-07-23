import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import seaborn
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import random
import warnings
random.seed(114514)
warnings.filterwarnings("ignore")

class Parameter:
    val = 0
    mnval = 0
    mxval = 0
    step = 0
    def __init__(self, mnval = 0, mxval = 0, step = 0):
        self.mnval = mnval
        self.mxval = mxval
        self.step = step

    def set(self, mnval, mxval, step = 1):
        self.mnval = mnval
        self.mxval = mxval
        self.step = step

    def genint(self):
        self.val = np.random.random_integers(self.mnval / self.step, self.mxval / self.step) * self.step

    def genfloat(self):
        self.val = np.random(self.mnval / self.step, self.mxval / self.step) * self.step

class Result:
    startCash = 0
    Cash = 0
    Asset = 0
    Share = 0
    Return = 0
    Alpha = 0
    Sharpe = 0
    buypoint = []
    sellpoint = []

    def __init__(self):
        pass

    def cal(self, bar, BMbar):
        self.Return = (self.Asset - self.startCash)/self.startCash      
        self.Alpha = self.alpha(bar, BMbar, 0.03)
        self.Sharpe = self.sharpeRatio(bar, 0.03)
        self.Beta = self.beta(bar, BMbar)

    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        pass
    
    def sharpeRatio(self, stock_data,rf=0):  # rf-无风险年利率
        data_len=len(stock_data)
        try:
            date_list = stock_data["Date"]
        except:
            date_list = list(stock_data.index)
        start_date=date_list[0]
        start_date=dt.datetime.strptime(str(start_date)[0:10],"%Y-%m-%d").date()
        end_date=date_list[data_len-1]
        end_date=dt.datetime.strptime(str(end_date)[0:10],"%Y-%m-%d").date()
        date_years=(end_date-start_date).days/365
        seq=np.array(stock_data["Close"])
        seq_return=seq[-1]/seq[0]-1
        seq_return_year=seq_return ** (1/date_years)
        seq=seq[1:]/seq[0:-1]-1
        e_seq=sum(seq)/len(seq)
        var_seq=(seq-e_seq) ** 2
        var_seq=sum(var_seq)/len(var_seq)
        vola_value=(var_seq ** (1/2))*(365 ** (1/2))
        sharpe_value=(seq_return_year-rf)/vola_value
        return sharpe_value

    def alpha(self, stock_data,benchmark_data,rf=0):  # rf-无风险年利率
        data_len=len(stock_data)
        try:date_list=stock_data["Date"]
        except:date_list=list(stock_data.index)
        start_date=date_list[0]
        start_date=dt.datetime.strptime(str(start_date)[0:10],"%Y-%m-%d").date()
        end_date=date_list[data_len-1]
        end_date=dt.datetime.strptime(str(end_date)[0:10],"%Y-%m-%d").date()
        date_years=(end_date-start_date).days/365
        rf=rf ** (1/date_years)
        seq=np.array(stock_data["Close"])
        seq_bench=np.array(benchmark_data["Close"])
        seq_return=seq[-1]/seq[0]-1
        seq_bench_return=seq_bench[-1]/seq_bench[0]-1
        seq_bench=seq_bench[1:]/seq_bench[0:-1]-1
        seq=seq[1:]/seq[0:-1]-1
        e_seq_bench=sum(seq_bench)/len(seq_bench)
        e_seq=sum(seq)/len(seq)
        cov=(seq_bench-e_seq_bench)*(seq-e_seq)
        cov=sum(cov)/len(cov)
        var=(seq_bench-e_seq_bench) ** 2
        var=sum(var)/len(var)
        beta_value=cov/var
        alpha_value=seq_return-rf-beta_value*seq_bench_return
        return alpha_value

    def beta(self, stock_data, benchmark_data):
        seq = np.array(stock_data["Close"])
        seq_bench = np.array(benchmark_data["Close"])
        seq_bench = seq_bench[1:] / seq_bench[0:-1] - 1
        seq = seq[1:] / seq[0:-1] - 1
        e_seq_bench = sum(seq_bench) / len(seq_bench)
        e_seq = sum(seq) / len(seq)
        cov = (seq_bench - e_seq_bench) * (seq - e_seq)
        cov = sum(cov) / len(cov)
        var = (seq_bench - e_seq_bench) ** 2
        var = sum(var) / len(var)
        return cov / var


    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        pass
    
    
class Stop_Price:
    def __init__(self):
        pass
    stop_profit = float('inf')
    stop_loss = -float('inf')
    def set_limit(self, stop_profit=float('inf'), stop_loss=-float('inf')):
        self.stop_profit = stop_profit
        self.stop_loss = stop_loss
        return stop_profit, stop_loss


class Strategy(Result, Stop_Price, Parameter):
    pdata = pd.DataFrame()

    def __init__(self):
        pass

    def clear(self):
        self.Asset = self.startCash
        self.Cash = self.startCash
        self.Share = 0
        self.Alpha = 0
        self.Sharpe = 0
        self.Return = 0
        self.buypoint = []
        self.sellpoint = []
        self.pdata = pd.DataFrame()

    def setcash(self, Cash):
        self.startCash = self.Cash = self.Asset = Cash

    def sell(self, time, price, shares):
        # limit
        self.Cash += price * shares
        self.Share -= shares
        sellpoint.append([time,price])
    def buy(self, time, price, shares):
        # limit
        print("buy: ",time,shares,price)
        self.buyprice = price
        self.Cash -= price * shares
        self.Share += shares
        self.buypoint.append([time,price])

    def liquidate(self, time, price):
        self.Cash += self.Share * price
        print("sell:", time, self.Share, price, self.Cash)
        self.Share = 0
        self.sellpoint.append([time,price])

    def checklimit(self, price):
        stop_profit = self.stop_profit
        stop_loss = self.stop_loss
        buyprice = self.buyprice
        if (price-buyprice)/buyprice > stop_profit:
            return 1
        if (price-buyprice)/buyprice < stop_loss:
            return -1
        else:
            return 0

    def stop_limit(self, time, price):
        self.Cash += self.Share * price
        print("limit：", time, self.Share, price, self.Cash)
        self.Share = 0
        self.sellpoint.append([time, price])

    def test(self, starttime, endtime):
        pass

    def train(self, token, starttime, endtime, times):
        pass

    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        pass

class _MACD(Strategy):
    slen = 0  # best parameter
    llen = 0
    bestAsset = 0
    _slen = Parameter() # train parameter
    _llen = Parameter()
    istrainready = 0
    def __init__(self):
        pass
    def setlen(self, slen=20, llen=60):
        self.slen = slen
        self.llen = llen

    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        self.cal(bar, BMbar)
        self.data = yf.Ticker(token)
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("PE:",round(self.data.info['forwardPE'],2))
        print("Market Cap:",round(self.data.info["marketCap"],2))
        print("Actions:",round(self.data.actions,2))
        print("Cash Flow:", self.data.cashflow)
        print("Balance Sheet:", self.data.balance_sheet)
        print("Financial Sheet:", self.data.financials)
        print("startCash:",self.startCash)
        print("Asset:",round(self.Asset,2))
        print("Cash:",round(self.Cash,2))
        print("Share:",round(self.Share,2))
        print("Alpha:",round(self.Alpha,2))
        print("Sharpe",round(self.Sharpe,2))
        print("buy:",self.buypoint)
        print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        print("Beta:", round(self.Beta,2)))
        print('-----------------------------------------------')
        bar = bar[pd.to_datetime(starttime) <= bar['Date']]
        bar = bar[bar['Date'] <= pd.to_datetime(endtime)]
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bar['Date'], bar['Close'])
        plt.plot(self.pdata['Date'], self.pdata['SMA'], label='SMA')
        plt.plot(self.pdata['Date'], self.pdata['LMA'], label='LMA')
        plt.legend()
        plt.ylabel("Price(USD)")
        plt.subplot(2, 1, 2)
        plt.ylabel("Asset(USD)")
        plt.plot(bar['Date'], self.pdata['Asset'])
        plt.tight_layout()
        if saveimg:
            plt.savefig('./Result/'+token+dt.datetime.now().strftime('%H%M%S'))


    def test(self, bar, starttime, endtime, istrain = 0):
        self.clear()
        stop_profit = self.stop_profit
        stop_loss = self.stop_loss
        if istrain:
            llen = self._llen.val
            slen = self._slen.val
        else:
            llen = self.llen
            slen = self.slen
        for i in range(llen + 1, len(bar)):
            if not (pd.to_datetime(starttime) <= bar.Date[i] <= pd.to_datetime(endtime)):continue
            self.pdata.loc[i, ['Date']] = bar.loc[i, ['Date']]
            sma_pre = bar.Close[i - slen:i].mean()
            sma_now = bar.Close[i - slen + 1:i + 1].mean()
            lma_pre = bar.Close[i - llen:i].mean()
            lma_now = bar.Close[i - llen + 1:i + 1].mean()
            self.pdata.loc[i, ['SMA']] = sma_now # record , get ready for plot
            self.pdata.loc[i, ['LMA']] = lma_now
            if self.Share == 0:
                if sma_pre < lma_pre and sma_now > lma_now:
                    self.buy(bar.Date[i], bar.Close[i], self.Cash / bar.Close[i])  # Buy with all cash
                    buyprice = bar.Close[i]
            else:
                self.Asset += self.Share * (bar.Close[i] - bar.Close[i - 1])  ##
                if self.checklimit(bar.Close[i]) == 1:
                    self.stop_limit(bar.Date[i], buyprice*(1+stop_profit))
                    self.Asset = self.Cash
                elif self.checklimit(bar.Close[i]) == -1:
                    self.stop_limit(bar.Date[i], buyprice*(1+stop_loss))
                    self.Asset = self.Cash
                else:
                    if sma_pre > lma_pre and sma_now < lma_now:
                        self.liquidate(bar.Date[i], bar.Close[i])  # Sell all shares
            self.pdata.loc[i, ['Asset']] = self.Asset
        if istrain and self.Asset > self.bestAsset:
            self.bestAsset = self.Asset
            self.llen = self._llen.val
            self.slen = self._slen.val

    def traininit(self, _slen, _llen):
        self._slen = _slen
        self._llen = _llen
        self.istrainready = 1
        self.bestAsset = -np.inf

    def train(self, bar, starttime, endtime, times):
        if self.istrainready != 1:
            print('Parameters\' range is not set.')
            sys.exit(0)
            return
        for i in range(times):
            self._slen.genint()
            self._llen.genint()
            while self._slen.val == self._llen.val:
                self._slen.genint()
                self._llen.genint()
            if self._slen.val>self._llen.val:
                self._slen.val, self._llen.val = self._llen.val, self._slen.val
            self.test(bar, starttime, endtime, 1)

class _RBreaker(Strategy):
    ylen = 0  # best parameter
    mlen = 0
    bestAsset = 0
    _ylen = Parameter()  # train parameter
    _mlen = Parameter()
    istrainready = 0

    def __init__(self):
        pass

    def setlen(self, ylen=1, mlen=3):
        self.ylen = ylen
        self.mlen = mlen

    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        self.cal(bar, BMbar)
        self.data = yf.Ticker(token)
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("PE:",round(self.data.info['forwardPE'],2))
        print("Market Cap:",round(self.data.info["marketCap"],2))
        print("Actions:",round(self.data.actions,2))
        print("Cash Flow:", self.data.cashflow)
        print("Balance Sheet:", self.data.balance_sheet)
        print("Financial Sheet:", self.data.financials)
        print("startCash:",self.startCash)
        print("Asset:",round(self.Asset,2))
        print("Cash:",round(self.Cash,2))
        print("Share:",round(self.Share,2))
        print("Alpha:",round(self.Alpha,2))
        print("Sharpe",round(self.Sharpe,2))
        print("buy:",self.buypoint)
        print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        print("Beta:", round(self.Beta,2))
        print('-----------------------------------------------')
        bar = bar[pd.to_datetime(starttime) <= bar['Date']]
        bar = bar[bar['Date'] <= pd.to_datetime(endtime)]
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bar['Date'], bar['Close'])
        plt.plot(self.pdata['Date'], self.pdata['Pivot'])
        plt.plot(self.pdata['Date'], self.pdata['Break'])
        label = ["Close","Pivot","Break"]
        plt.legend(label)
        plt.ylabel("Price(USD)")
        plt.subplot(2, 1, 2)
        plt.ylabel("Asset(USD)")
        plt.plot(bar['Date'], self.pdata['Asset'])
        plt.tight_layout()
        if saveimg:
            plt.savefig('./Result/'+token+dt.datetime.now().strftime('%H%M%S'))


    def test(self, bar, starttime, endtime, istrain=0):
        self.clear()
        if istrain:
            ylen = self._ylen.val
            mlen = self._mlen.val
        else:
            ylen = self.ylen
            mlen = self.mlen
        for i in range(mlen + 1, len(bar)):
            if not (pd.to_datetime(starttime) <= bar.Date[i] <= pd.to_datetime(endtime)):continue
            self.pdata.loc[i, ['Date']] = bar.loc[i, ['Date']]
            # cal Pivot
            mhigh = bar.High[i - mlen:i].mean()
            mlow = bar.Low[i - mlen:i].mean()
            mclose = bar.Close[i - mlen:i].mean()
            pivot = (mhigh + mlow + mclose) / 3
            bBreak = mhigh + 2 * (pivot - mlow)
            sSetup = pivot + (mhigh - mlow)
            sEnter = 2 * pivot - mlow
            bEnter = 2 * pivot - mhigh
            bSetup = pivot - (mhigh - mlow)
            sBreak = pivot - 2 * (mhigh - mlow)
            p = bar.Close[i]
            yhigh = bar.High[i - ylen:i].mean()
            ylow = bar.Low[i - ylen:i].mean()
            self.pdata.loc[i, ['Pivot']] = pivot  # record , get ready for plot
            self.pdata.loc[i, ['Close']] = mclose
            self.pdata.loc[i, ['Break']] = bBreak
            if self.Share == 0:
                if p > bBreak or ylow < bSetup and p > bEnter:
                    # Buy with all cash
                    self.buy(bar.Date[i], bar.Close[i],
                             self.Cash / bar.Close[i])

            else:
                self.Asset += self.Share * (bar.Close[i] - bar.Close[i - 1])
                if yhigh > sSetup and p < sEnter:
                    # Sell all shares
                    self.liquidate(bar.Date[i], bar.Close[i])

            bar.loc[i, ['Asset']] = self.Asset
            self.pdata.loc[i, ['Asset']] = self.Asset




        if istrain and self.Asset > self.bestAsset:
            self.bestAsset = self.Asset
            self.ylen = self._ylen.val
            self.mlen = self._mlen.val

    def traininit(self, _ylen, _mlen):
        self._ylen = _ylen
        self._mlen = _mlen
        self.istrainready = 1
        self.bestAsset = -np.inf

    def train(self, bar, starttime, endtime, times):
        if self.istrainready != 1:
            print('Parameters\' range is not set.')
            return
        for i in range(times):
            self._ylen.genint()
            self._mlen.genint()
            while self._ylen.val == self._mlen.val:
                self._ylen.genint()
                self._mlen.genint()
            if self._ylen.val>self._mlen.val:
                self._ylen.val, self._mlen.val = self._mlen.val, self._ylen.val
            self.test(bar, starttime, endtime, 1)

class _bld(Strategy):
    k = 0  # best parameter
    t = 0
    bestAsset = 0
    _k = Parameter()  # train parameter
    _t = Parameter()
    istrainready = 0

    def __init__(self):
        pass

    def setlen(self, k=1, t=10):
        print(self)
        self.k = k
        self.t = t

    def show(self, token, bar, BMbar, starttime, endtime, saveimg=1):
        self.cal(bar, BMbar)
        self.data = yf.Ticker(token)
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("PE:",round(self.data.info['forwardPE'],2))
        print("Market Cap:",round(self.data.info["marketCap"],2))
        print("Actions:",round(self.data.actions,2))
        print("Cash Flow:", self.data.cashflow)
        print("Balance Sheet:", self.data.balance_sheet)
        print("Financial Sheet:", self.data.financials)
        print("startCash:", self.startCash)
        print("Asset:", round(self.Asset, 2))
        print("Cash:", round(self.Cash, 2))
        print("Share:", round(self.Share, 2))
        print("Alpha:",round(self.Alpha,2))
        print("Sharpe",round(self.Sharpe,2))
        print("buy:",self.buypoint)
        print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        print("Beta:", round(self.Beta,2))
        print('-----------------------------------------------')
        bar = bar[pd.to_datetime(starttime) <= bar['Date']]
        bar = bar[bar['Date'] <= pd.to_datetime(endtime)]
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bar['Date'], bar['Close'])
        plt.plot(self.pdata['Date'], self.pdata['UP'])
        plt.plot(self.pdata['Date'], self.pdata['DOWN'])
        label = ["Close", "UP","DOWN"]
        plt.legend(label)
        plt.ylabel("Price(USD)")
        plt.subplot(2, 1, 2)
        plt.ylabel("Asset(USD)")
        plt.plot(bar['Date'], self.pdata['Asset'])
        plt.tight_layout()
        if saveimg:
            plt.savefig('./Result/' + token + dt.datetime.now().strftime('%H%M%S'))



    def test(self, bar, starttime, endtime, istrain=0):
        self.clear()
        stop_profit = self.stop_profit
        stop_loss = self.stop_loss
        if istrain:
            k = self._k.val
            t = self._t.val
        else:
            k = self.k
            t = self.t
        for i in range(k + 1, len(bar)):
            if not (pd.to_datetime(starttime) <= bar.Date[i] <= pd.to_datetime(endtime)):
                continue
            self.pdata.loc[i, ['Date']] = bar.loc[i, ['Date']]
            # cal MA
            ma = bar.Close[i - t:i].mean()
            up = (ma + k * bar.Close.std()) / 1.3
            down = (ma - k * bar.Close.std()) * 1.3
            p = bar.Close[i - 1]
            self.pdata.loc[i, ['UP']] = up  # record , get ready for plot
            self.pdata.loc[i, ['DOWN']] = down
            if self.Share == 0:
                if p < down:
                    # Buy with all cash
                    self.buy(bar.Date[i], bar.Close[i], self.Cash / bar.Close[i])
                    buyprice = bar.Close[i]
            else:
                self.Asset += self.Share * (bar.Close[i] - bar.Close[i - 1])  # Hold
                if self.checklimit(bar.Close[i]) == 1:
                    self.stop_limit(bar.Date[i], buyprice * (1 + stop_profit))
                    self.Asset = self.Cash
                elif self.checklimit(bar.Close[i]) == -1:
                    self.stop_limit(bar.Date[i], buyprice * (1 + stop_loss))
                    self.Asset = self.Cash
                else:
                    if p > up:
                        # Sell all shares
                        self.liquidate(bar.Date[i], bar.Close[i])
            bar.loc[i, ['Asset']] = self.Asset
            self.pdata.loc[i, ['Asset']] = self.Asset
        if istrain and self.Asset > self.bestAsset:
            self.bestAsset = self.Asset
            self.k = self._k.val
            self.t = self._t.val

    def traininit(self, _k, _t):
        self._k = _k
        self._t = _t
        self.istrainready = 1
        self.bestAsset = -np.inf

    def train(self, bar, starttime, endtime, times):
        if self.istrainready != 1:
            print('Parameters\' range is not set.')
            return
        for i in range(times):
            self._k.genint()
            self._t.genint()
            self.test(bar, starttime, endtime, 1)

class _LSTM(Strategy):
    def __init__(self):
        pass

    def LSTM_pretreatment(self, bar):
        bar.set_index(bar['Date'], inplace=True)
        bar = bar.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
        return bar

    def Stock_Price_LSTM_Data_Processing(self, bar, the_mem_days=15, pre_days=10):
        bar = self.LSTM_pretreatment(bar)
        bar.dropna(inplace=True)
        bar.sort_index(inplace=True)
        bar['label'] = bar['Close'].shift(-pre_days)

        scaler = StandardScaler()
        sca_X = scaler.fit_transform(bar.iloc[:, :-1])

        # mem_his_days = 10 #记忆天数

        deq = deque(maxlen=the_mem_days)

        X = []
        for i in sca_X:
            deq.append(list(i))
            if len(deq) == the_mem_days:
                X.append(list(deq))

        X = X[:-pre_days]

        y = bar['label'].values[the_mem_days - 1:-pre_days]

        X = np.array(X)
        y = np.array(y)

        return X, y

    def get_LSTM_model(self, bar, the_mem_days=15, the_lstm_layers=1, the_dense_layers=1, the_units=32):
        self.clear()
        print('It may take some time to fit the model, please wait.^_^')
        filepath = '.\\models\\'+f'men_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_units}'
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            monitor='val_mape',
            mode='min',
            save_best_only=True)

        X, y = self.Stock_Price_LSTM_Data_Processing(bar, the_mem_days=the_mem_days, pre_days=10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)


        model = Sequential()
        model.add(LSTM(the_units, input_shape=X.shape[1:], activation='relu', return_sequences=True))
        model.add(Dropout(0.1))

        for i in range(the_lstm_layers):
            model.add(LSTM(the_units, activation='relu', return_sequences=True))
            model.add(Dropout(0.1))

        model.add(LSTM(the_units, activation='relu'))
        model.add(Dropout(0.1))

        for i in range(the_dense_layers):
            model.add(Dense(the_units, activation='relu'))
            model.add(Dropout(0.1))

        model.add(Dense(1))

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mape'])

        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

    def pred_LSTM_price(self, bar):
        best_model = load_model('.//models//men_15_lstm_1_dense_1_unit_32')
        X, y = self.Stock_Price_LSTM_Data_Processing(bar, the_mem_days=15, pre_days=5)
        y_test = y
        bar_time = bar.index[-len(y_test):]
        # best_model.evaluate(X_test, y_test)
        pre = best_model.predict(X)
        return bar_time, y_test, pre

    def LSTM_buy_judge(self, bar):
        bar_time, y_test, pre = self.pred_LSTM_price(bar)
        bar1 = pd.DataFrame(pre, index=bar_time, columns=['pre'])
        bar2 = pd.DataFrame(y_test, index=bar_time, columns=['Close'])
        bar_pre = pd.merge(bar1, bar2, left_index=True, right_index=True)
        bar_pre.set_index(bar_time)
        bar_pre['pred_profit'] = bar_pre['pre'] - bar_pre['pre'].shift(1)
        bar_pre.iloc[0, -1] = 0
        return bar_pre, bar_time

    def LSTM_backtest(self, bar, starttime, endtime):
        self.clear()
        bar_pre, bar_time = self.LSTM_buy_judge(bar)
        bar_pre = bar_pre[pd.to_datetime(starttime) <= bar_pre.index]
        bar_pre = bar_pre[bar_pre.index <= pd.to_datetime(endtime)]
        bar_time = bar_pre.index
        stop_profit = self.stop_profit
        stop_loss = self.stop_loss
        buyprice = 0
        for i in range(len(bar_time)):
            pred_profit = bar_pre.loc[bar_time[i], 'pred_profit']
            if self.Share == 0:
                if pred_profit > 0:
                    self.buy(bar_pre.index[i], bar_pre.Close[i], self.Cash / bar_pre.Close[i])
                    buyprice = bar_pre.Close[i]
            else:
                self.Asset += self.Share * (bar_pre.Close[i] - bar_pre.Close[i - 1])
                if self.checklimit(bar_pre.Close[i]) == 1:
                    self.stop_limit(bar_pre.index[i], buyprice * (1 + stop_profit))
                    self.Asset = self.Cash
                elif self.checklimit(bar_pre.Close[i]) == -1:
                    self.stop_limit(bar_pre.index[i], buyprice * (1 + stop_loss))
                    self.Asset = self.Cash
                else:
                    if pred_profit < 0:
                        self.liquidate(bar_pre.index[i], bar_pre.Close[i])
            self.pdata.loc[i, ['Asset']] = self.Asset
            self.bestAsset = self.Asset
        self.pdata = self.pdata.set_index(bar_time)


    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 0):
        self.cal(bar, BMbar)
        self.data = yf.Ticker(token)
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("PE:",round(self.data.info['forwardPE'],2))
        print("Market Cap:",round(self.data.info["marketCap"],2))
        print("Actions:",round(self.data.actions,2))
        print("Cash Flow:", self.data.cashflow)
        print("Balance Sheet:", self.data.balance_sheet)
        print("Financial Sheet:", self.data.financials)
        print("startCash:", self.startCash)
        print("Asset:", round(self.Asset, 2))
        print("Cash:", round(self.Cash, 2))
        print("Share:", round(self.Share, 2))
        #print("Alpha:",round(self.Alpha,2))
        #print("Sharpe",round(self.Sharpe,2))
        #print("buy:",self.buypoint)
        #print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        #print("Beta:", round(self.Beta,2))
        print('-----------------------------------------------')
        #plot
        bar_pre, bar_time = self.LSTM_buy_judge(bar)
        #bar = bar_pre.reset_index()
        #bar['Date'] = pd.to_datetime(bar['Date'])
        bar = bar_pre
        bar = bar[pd.to_datetime(starttime) <= bar.index]
        bar = bar[bar.index <= pd.to_datetime(endtime)]
        bar = bar.reset_index()
        self.pdata = self.pdata.reset_index()
        self.pdata = pd.merge(self.pdata, bar, how='right', left_on='Date', right_on='Date')
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self.pdata['Date'], self.pdata['pre'], color='red', label='prediction')
        plt.plot(self.pdata['Date'], self.pdata['Close'], color='blue', label='real price')
        plt.legend()
        plt.ylabel("Price(USD)")
        plt.subplot(2, 1, 2)
        plt.ylabel("Asset(USD)")
        plt.plot(self.pdata['Date'], self.pdata['Asset'])
        plt.tight_layout()
        #plt.show()
        if saveimg:
            plt.savefig('./Result/'+token+dt.datetime.now().strftime('%H%M%S'))


class StrLib:
    MACD = _MACD()
    RBreaker = _RBreaker()
    bld = _bld()
    LSTM = _LSTM()

    def __init__(self):
        pass


class Stock(StrLib):
    token = ''
    data = None
    bar = None
    Beta = np.nan

    '''def __init__(self, token, timedelta, starttime, endtime):
        self.token = token
        self.data = yf.Ticker(token)
        starttime = pd.to_datetime(starttime)
        starttime -= pd.to_timedelta('400 days')
        self.bar = self.data.history(interval=timedelta, start=starttime)
        self.bar.to_csv('./stocks' + str(self.token) + '.csv')
        self.bar = pd.read_csv('./stocks' + str(self.token) + '.csv')
        self.bar['Date'] = pd.to_datetime(self.bar['Date'])'''
    def __init__(self, token, timedelta, starttime, endtime):
        self.token = token
        self.data = yf.Ticker(token)
        self.bar = pd.read_csv(str(self.token) + '.csv')
        self.bar['Date'] = pd.to_datetime(self.bar['Date'])


class SelResult:
    Rank = []
    def __init__(self):
        pass


class SelStrategy(SelResult):

    def __init__(self):
        pass

    def train(self, Stocks, starttime, endtime):
        pass

    def test(self, Stocks, starttime, endtime):
        pass

    def show(self, Stocks, starttime, endtime):
        pass

class _MF(SelStrategy):
    def __init__(self):
        pass

class _SR(SelStrategy):
    def __init__(self):
        pass

class SelStrLib(SelStrategy):
    SR = _SR()
    MF = _MF()

class StockLib:
    Stocks = []
    BM = None
    SelStrLib = SelStrLib()
    isBMready = 0

    def __init__(self):
        pass

    def addstock(self, Stock):
        if self.isBMready == 0:
            print('BenchMark has not set yet.')
            sys.exit(0)
        self.Stocks.append(Stock)

    def setBM(self, Stock):
        self.BM = Stock
        self.isBMready = 1
