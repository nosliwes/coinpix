# This file contains functions to manage robinhood trading account

import pandas as pd
import robin_stocks.robinhood as r
   
def get_watchlist_symbols():
    """
    Returns: the symbol for each stock in your watchlist as a list of strings
    """
    my_list_names = []
    symbols = []
    for name in r.get_all_watchlists(info='name'):
        my_list_names.append(name)
    for name in my_list_names:
        list = r.get_watchlist_by_name(name)
        for item in list:
            instrument_data = r.get_instrument_by_url(item.get('instrument'))
            symbol = instrument_data['symbol']
            symbols.append(symbol)
    return symbols

def get_portfolio_symbols():
    """
    Returns: the symbol for each stock in your portfolio as a list of strings
    """
    symbols = []
    positions_data = r.get_current_positions()
    for item in positions_data:
        if not item:
            continue
        instrument_data = r.get_instrument_by_url(item.get('instrument'))
        symbol = instrument_data['symbol']
        symbols.append(symbol)
    return symbols

def execute_trades(action, args):
    
    for symbol in args.symbols:
        
        # get positions
        positions = get_positions(symbol)

        if action == 'sell':
            # sell positions
            sell_positions(symbol)

        if action == 'buy':
            # buy positions
            buy_positions(symbol)
        
def get_positions(symbol):
    """
    Returns: the current positions held for a given symbol in robinhood account
    """  
    positions_data = r.account.get_open_stock_positions()
    for item in positions_data:
        if not item:
            continue
        instrument_data = r.stocks.get_instrument_by_url(item.get('instrument'))
        instrument_symbol = instrument_data['symbol']
        if (instrument_symbol == symbol):
            buy_price = item['average_buy_price']
            quantity = item['quantity']
            current_price = r.stocks.get_latest_price(symbol)[0]
            position = (symbol, buy_price, quantity,current_price)
    
    return position

def sell_positions(symbol):
    
    print("sell", symbol)
    
def buy_positions(symbol):
    
    print("buy ", symbol)