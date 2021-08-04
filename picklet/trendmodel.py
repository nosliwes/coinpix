import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import warnings
from datetime import date 
from datetime import timedelta 
import robin_stocks.robinhood as r
import robin_stocks.tda as tda

warnings.filterwarnings("ignore")

# helper method to calculate EMA
def get_ema(df,span):
    sma = df.rolling(window=span, min_periods=span).mean()[:span]
    rest = df[span:]
    ema = pd.concat([sma, rest]).ewm(span=span, adjust=False).mean()
    return ema

def rsi (data, time_window):
    # diff in one field(one day)
    diff = data.diff(1).dropna()        

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def smooth_signals(x):
    shift = 3
    window = 15
    s = pd.Series(x)
    smooth0 = np.where((s == 1) | ((s == 0) & (s.shift(shift) == 1) & (s.shift(-shift) == 1)), 1, 0)
    smooth1 = np.where((s == 0) | ((s == 1) & (s.shift(shift) == 0) & (s.shift(-shift) == 0)), 0, 1)
    smooth2 = s.rolling(window, center=True).mean().round().fillna(s).astype(int).values
    return smooth2

def get_model_signals(symbol, price_df, model):
    short_window = 10
    model_df = price_df.copy()
    model_df.columns = ['high','low','open','price','volume','adj close']
    model_df['sma10'] = model_df['price'].rolling(window=short_window).mean()
    model_df['ema20'] = get_ema(model_df['price'],20)
    model_df['ema30'] = get_ema(model_df['price'],30)
    model_df['rsi'] = rsi(price_df['Close'],14)
    model_df['macd'] = model_df['sma10'] - model_df['ema20']
    model_df = model_df.dropna()

    columns = ['sma10','ema20','ema30','price', 'open', 'high', 'low','macd','rsi'] 
    
    # predict signals
    model_df['signal'] = model.predict(model_df[columns]) 

    # smooth signals
    model_df['signal'] = smooth_signals(model_df['signal'])
    
    # trading orders
    model_df['positions'] = model_df['signal'].diff()
    
    # save prices for optimization
    # filepath = 'prices/' + symbol + '.csv'
    # model_df.to_csv(filepath)

    return model_df

def plot_signals(signals, title):
    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(111,ylabel='Price')

    ax1.plot(signals['price'], color='black')
    ax1.plot(signals['sma10'], color='orange')
    ax1.plot(signals['ema20'], color='blue')
    ax1.plot(signals['ema30'], color='green')
    
    ax1.title.set_text(title)

    # plot buys
    ax1.plot(signals.loc[signals.positions == 1.0].index, \
             signals.sma10[signals.positions == 1.0], '^', markersize=20, color='g')

    # plot sells
    ax1.plot(signals.loc[signals.positions == -1.0].index, \
             signals.sma10[signals.positions == -1.0], 'v', markersize=20, color='r')
       
    plt.show()
    
def get_historical(symbol, start_date, end_date):
    price_df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    return price_df

def backtest_signals(signals, method):
    # create empty positions
    positions = pd.DataFrame(index=signals.index).fillna(0.0)

    # define number of shares on each buy
    num_shares = 100  

    positions['price'] = signals['price']
    positions['shares'] = signals['signal'] * num_shares
    positions['net shares'] = positions['shares'].diff()

    # add day 1 purchase if needed
    if signals['signal'].iloc[0] == 1.0:
        positions['net shares'].iloc[0] = num_shares       

    # calculate net cash
    positions['net cash'] = (-1) * positions['net shares'] * positions['price']

    # close any open positions on last day       
    if positions['shares'].iloc[-1] == num_shares:
        positions['net cash'].iloc[-1] = positions['net cash'].iloc[-1] + \
                                        (positions['shares'].iloc[-1] * positions['price'].iloc[-1])
       
    # calculate return
    initial_cash = positions['price'].iloc[0] * num_shares
    profit = positions['net cash'].sum()
    total_return = profit / initial_cash * 100    

    # return results
    return total_return

def get_historical_robinhood(symbol, interval):
       
    # get price data from robinhood
    price_df = pd.DataFrame(r.stocks.get_stock_historicals(symbol,interval))

    # rearrange dataframe columns since model expects yahoo price data and not robinhood

    # reorder columns
    columns = ["begins_at","high_price","low_price","open_price","close_price", "volume", "close_price"]
    price_df = price_df[columns]

    # rename columns
    price_df.columns = ["Date","High","Low","Open","Close","Volume","Adj Close"]

    # format date and re-index
    # price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.strftime('%Y-%m-%d')
    price_df.set_index("Date", inplace=True)
    
    # convert data columns to floats
    columns = ["High","Low","Open","Close","Volume","Adj Close"]
    price_df[columns] = price_df[columns].astype(float)

    return price_df 

def get_historical_tda(symbol, period_type, frequency_type, frequency):
    # get price data from tda
    price_history=tda.stocks.get_price_history(symbol, period_type, frequency_type, frequency, jsonify=True)
    price_df = pd.DataFrame.from_dict(price_history[0]['candles'])
    
    columns = ["datetime","high","low","open","close", "volume", "close"]
    price_df = price_df[columns]

    # rename columns
    price_df.columns = ["Date","High","Low","Open","Close","Volume","Adj Close"]

    # format date and re-index
    price_df['Date'] = pd.to_datetime(price_df['Date'], unit='ms')
    price_df.set_index("Date", inplace=True)

    columns = ["High","Low","Open","Close","Volume","Adj Close"]
    price_df[columns] = price_df[columns].astype(float)
    
    return price_df

def process_symbols(symbols, intervals, source):
    
    # load predictive model
    model_file = 'model.sav'
    model = joblib.load(model_file)
    
    # declare list to hold results
    results = []

    # process symbols
    for symbol in symbols:
        print('analyzing',symbol)
        
        for interval in intervals:   
            # load price history
            if (source == "yahoo" or interval == "day"):
                start = date.today() - timedelta(days = 365) 
                end = date.today()
                price_df = get_historical(symbol, start, end)
            if (source == "robinhood"):
                price_df = get_historical_robinhood(symbol, interval)
            if (source == "tda"):
                period_type = 'day' # Valid values are day, month, year, or ytd. default is day
                frequency_type = 'minute'
                frequency = '5'
                period = '5'
                price_df = get_historical_tda(symbol, period_type, frequency_type, frequency)                
            
            # run model experiment
            model_signals = get_model_signals(symbol, price_df, model)
            model_return = backtest_signals(model_signals, 1)

            # add signal to price df
            price_df['signal'] = model_signals['signal']

            # current signal
            if (model_signals['signal'].tail(1).iloc[0] == 0):
                action = 'sell'
            else:
                action = 'buy'

            # assemble result
            result = symbol, interval, model_return, action

            # append results
            results.append(result)  

            # plot model signals
            plot_signals(model_signals, symbol + ' ' + interval)

            # print tail of prices
            display(price_df.tail(5))

            # display output for buys
            if(action == 'buy'):

                # get current price
                current_price = r.stocks.get_latest_price(symbol)[0]
                stop_price = float(current_price) * .995

                print("current price", current_price, "stop loss", stop_price)

                print(result)
            
    return results