from kernal import *

timedelta = '1d'
starttime = '2021-01-01'
endtime = '2022-01-01'
now = '20' + dt.datetime.now().strftime('%y-%m-%d')
SLib = StockLib()

def init():
    SLib.setBM(Stock('SPY', timedelta, starttime, now))
    SLib.addstock(Stock('AAPL', timedelta, starttime, now))
    #SLib.addstock(Stock('TSLA', timedelta, starttime, now))
    #SLib.addstock(Stock('IBM', timedelta, starttime, now))


def main():
    init()
    for stock in SLib.Stocks:
        obj = stock.MACD
        obj.setlen(15, 60)  # set parameters of MACD
        obj.setcash(100000)  # 100000USD for each stock
        obj.test(stock.bar, starttime, now)  # use default parameter to run MACD
        obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
        obj.traininit(Parameter(5, 50, 5), Parameter(20, 200, 5))  # SMA in [5:50:5]; LMA in [20:200:5]
        obj.train(stock.bar, starttime, endtime, 10)  # use past data, train 10 times to find the better parameters
        print(stock.MACD.slen, stock.MACD.llen)  # print the parameters after training
        obj.test(stock.bar, starttime, now)  # use the parameter to test
        obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)

        obj = stock.RBreaker
        obj.setlen(1, 3)  # set parameters of RBreaker
        obj.setcash(100000)  # 100000USD for each stock
        obj.test(stock.bar, starttime, now)  # use default parameter to run RBreaker
        obj.traininit(Parameter(1,10,1),Parameter(3,20,3))  # ylen in [1:5:10]; mlen in [3:10:20]
        obj.train(stock.bar, starttime, endtime, 5)  # use past data, train 5 times to find the better parameters
        print(stock.RBreaker.ylen, stock.RBreaker.mlen)  # print the parameters after training
        obj.test(stock.bar, starttime, now)  # use the parameter to test
        obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)

      
        obj = stock.bld
        obj.setlen(1)
        obj.setcash(100000)
        obj.test(stock.bar, starttime, now)
        obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)
        obj.traininit(Parameter(1, 15, 1), Parameter(10, 50, 5))
        obj.train(stock.bar, starttime, endtime, 5)
        print(stock.bld.k, stock.bld.t)
        obj.test(stock.bar, starttime, now)
        obj.show(stock.token, stock.bar, SLib.BM.bar, starttime, now, 1)

if __name__ == '__main__':
    main()
