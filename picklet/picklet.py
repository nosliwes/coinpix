import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keyring
from datetime import datetime
import time

#import trend model functions
from picklet_model import *

# import robinhood functions
import robin_stocks.robinhood as r
from rh_account import *

def run_model(args=None):

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    symbols = args.symbols
    intervals = args.intervals
        
    # connect to robinhood account
    login = r.login(keyring.get_password("robinhood","username"),keyring.get_password("robinhood","password"))
    
    # process symbols
    results = process_symbols(symbols, intervals, source = "robinhood")
    
    # show results
    current_time = datetime.now()
    current_interval = results[0][1]
    current_symbol = results[0][0]
    current_action = results[0][3]
    
    print(current_time, current_symbol, current_action)  
    
    return(current_action)
    
def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Trend Trading Bot', fromfile_prefix_chars='@')
    parser.add_argument('-symbols', nargs='+', type=str, default=['AAPL'], help='list of ticker symbols') 
    parser.add_argument('-intervals', nargs='+', type=str, default=['5minute'], help='interval to trade')
    parser.add_argument('-refresh', type=int, default=60, help='refresh rate in seconds')
    parser.add_argument('-broker', type=str, default='robinhood', help='broker to use')
    parser.add_argument('-cash', type=int, default=0, help='cash to trade')    
    parser.add_argument('-live', type=bool, default=False, help='execute trades')
    
    return parser

if __name__ == "__main__":
    last_action = None
    parser = create_parser()
    args = parser.parse_args()
    print('picklet settings', args)
    
    while True:
        current_action = run_model(args)
        
        if (current_action != last_action):
            # if live execute trades
            if args.live:
                execute_trades(current_action, args)
            last_action = current_action
        
        time.sleep(args.refresh)

    