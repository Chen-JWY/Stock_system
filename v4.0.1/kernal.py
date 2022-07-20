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
        x = pd.Series.to_numpy(bar['Close'].pct_change()[1:])
        y = pd.Series.to_numpy(BMbar['Close'].pct_change()[1:])
        self.Beta = (np.cov(x,y)[0,1]) / np.var(y)

    def show(self, token, bar, BMbar, starttime, endtime, saveimg = 1):
        pass

class Strategy(Result,Parameter):
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
        # print("buy: ",time,shares,price)
        self.Cash -= price * shares
        self.Share += shares
        self.buypoint.append([time,price])

    def liquidate(self, time, price):
        self.Cash += self.Share * price
        # print("sell:", time, self.Share, price, self.Cash)
        self.Share = 0
        self.sellpoint.append([time,price])

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
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("startCash:",self.startCash)
        print("Asset:",round(self.Asset,2))
        print("Cash:",round(self.Cash,2))
        print("Share:",round(self.Share,2))
        print("Alpha:",self.Alpha)
        print("Sharpe",self.Sharpe)
        print("buy:",self.buypoint)
        print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        print("Beta:",round(self.Beta))
        print('-----------------------------------------------')
        bar = bar[pd.to_datetime(starttime) <= bar['Date']]
        bar = bar[bar['Date'] <= pd.to_datetime(endtime)]
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bar['Date'], bar['Close'])
        plt.plot(self.pdata['Date'], self.pdata['SMA'])
        plt.plot(self.pdata['Date'], self.pdata['LMA'])
        #plt.legend()
        plt.ylabel("Price(USD)")
        plt.subplot(2, 1, 2)
        plt.ylabel("Asset(USD)")
        plt.plot(bar['Date'], self.pdata['Asset'])
        plt.tight_layout()
        if saveimg:
            plt.savefig('./Result/'+token+dt.datetime.now().strftime('%H%M%S'))


    def test(self, bar, starttime, endtime, istrain = 0):
        self.clear()
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
            else:
                self.Asset += self.Share * (bar.Close[i] - bar.Close[i - 1]) ##
                if sma_pre > lma_pre and sma_now < lma_now:
                    self.liquidate(bar.Date[i],bar.Close[i])  # Sell all shares
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
        print('-------------------RESULT----------------------')
        print("Stock:", token)
        print("startCash:",self.startCash)
        print("Asset:",round(self.Asset,2))
        print("Cash:",round(self.Cash,2))
        print("Share:",round(self.Share,2))
        print("Alpha:",self.Alpha)
        print("Sharpe",self.Sharpe)
        print("buy:",self.buypoint)
        print("sell:",self.sellpoint)
        print("Return: {:.2%}".format(self.Return))
        print("Beta:", round(self.Beta))
        print('-----------------------------------------------')
        bar = bar[pd.to_datetime(starttime) <= bar['Date']]
        bar = bar[bar['Date'] <= pd.to_datetime(endtime)]
        self.pdata = self.pdata[pd.to_datetime(starttime) <= self.pdata['Date']]
        self.pdata = self.pdata[self.pdata['Date'] <= pd.to_datetime(endtime)]
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(bar['Date'], bar['Close'])
        plt.plot(self.pdata['Date'], self.pdata['Pivot'])
        label = ["Close","Pivot"]
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
            if self.Share == 0:
                if p > bBreak:
                    self.buy(bar.Date[i], bar.Close[i],
                             self.Cash / bar.Close[i])# Buy with all cash
            else:
                self.Asset += self.Share * (bar.Close[i] - bar.Close[i - 1])
                if yhigh > sSetup and p < sEnter:
                    # Sell all shares
                    self.liquidate(bar.Date[i], bar.Close[i])
                elif ylow < bSetup and p > bEnter:
                    # Buy with all cash
                    self.buy(bar.Date[i], bar.Close[i],
                             self.Cash / bar.Close[i])
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



class StrLib:
    MACD = _MACD()
    RBreaker = _RBreaker()
    

    def __init__(self):
        pass


class Stock(StrLib):
    token = ''
    data = None
    bar = None
    Beta = np.nan

    def __init__(self, token, timedelta, starttime, endtime):
        self.token = token
        self.data = yf.Ticker(token)
        starttime = pd.to_datetime(starttime)
        starttime -= pd.to_timedelta('400 days')
        self.bar = self.data.history(interval=timedelta, start=starttime)
        self.bar.to_csv('./stocks' + str(self.token) + '.csv')
        self.bar = pd.read_csv('./stocks' + str(self.token) + '.csv')
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

