from kernal import *

timedelta = '1d'
starttime = '2021-01-01'
endtime = '2022-01-01'
now = '20' + dt.datetime.now().strftime('%y-%m-%d')
SLib = StockLib()


def init():
    SLib.setBM(Stock('SPY', timedelta, starttime, now))
    SLib.addstock(Stock('AAPL', timedelta, starttime, now))   #from yahoo finance
    SLib.addstock(Stock('TSLA', timedelta, starttime, now))
    SLib.addstock(Stock('IBM', timedelta, starttime, now))
    SLib.addstock(Stock('SPY', timedelta, starttime, now))


def main():
    init()
    for stock in SLib.Stocks:

        if stock.token == 'AAPL':
            obj = stock.MACD
            obj.setlen(15, 60)  # set parameters of MACD
            obj.setcash(100000)  # 100000USD for each stock
            obj.set_limit(stop_profit=0.15, stop_loss=-0.1) # set maximum return rate and pullback
            obj.test(stock.bar, starttime, now)  # use default parameter to run MACD
            obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
            obj.traininit(Parameter(5, 50, 5), Parameter(20, 200, 5))  # SMA in [5:50:5]; LMA in [20:200:5]
            obj.train(stock.bar, starttime, endtime, 10)  # use past data, train 10 times to find the better parameters
            print(stock.MACD.slen, stock.MACD.llen)  # print the parameters after training
            obj.test(stock.bar, starttime, now)  # use the parameter to test
            obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
            stock.show(stock.token,stock.bar,SLib.BM.bar)

        if stock.token == 'TSLA':
            obj = stock.RBreaker
            obj.setlen(1, 3)  # set parameters of RBreaker
            obj.setcash(100000) # 100000USD for each stock
            obj.set_limit(stop_profit=0.15, stop_loss=-0.1) # set maximum return rate and pullback
            obj.test(stock.bar, starttime, now)  # use default parameter to run RBreaker
            obj.traininit(Parameter(1, 10, 1), Parameter(3, 20, 3))  # ylen in [1:10:1]; mlen in [3:20:3]
            obj.train(stock.bar, starttime, endtime, 5)  # use past data, train 5 times to find the better parameters
            print(stock.RBreaker.ylen, stock.RBreaker.mlen)  # print the parameters after training
            obj.test(stock.bar, starttime, now)  # use the parameter to test
            obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
            stock.show(stock.token,stock.bar,SLib.BM.bar)

        if stock.token == 'IBM':
            obj = stock.bld    #Boll Channel Strategy
            obj.setlen(1,10)
            obj.setcash(100000)
            obj.set_limit(stop_profit=0.15, stop_loss=-0.1)
            obj.test(stock.bar, starttime, now)
            obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
            #obj.traininit(Parameter(1, 15, 1), Parameter(10, 50, 5))
            #obj.train(stock.bar, starttime, endtime, 5)
            #print(stock.bld.k, stock.bld.t)
            #obj.test(stock.bar, starttime, now)
            #obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
            stock.show(stock.token,stock.bar,SLib.BM.bar)

        if stock.token == 'SPY':
            obj = stock.LSTM
            obj.setcash(100000)  # 100000USD for each stock
            obj.set_limit(stop_profit=0.15, stop_loss=-0.1)
            obj.get_LSTM_model(stock.bar, endtime=starttime)  # train lstm model with default parameter and save model
            obj.LSTM_backtest(stock.bar, starttime, endtime)
            obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)  # show and save plot
            stock.show(stock.token,stock.bar,SLib.BM.bar)


if __name__ == '__main__':
    main()
