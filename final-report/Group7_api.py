# CS 5593 CoinPIX API
# Group 7
#
# This is the main API for CoinPIX.  It is a flask application written in python.
#    It begins by exposing a route to /api/clusters for the Angular User Interface
#    1) The initial /api/clusters endpoint returns the coins labeled by cluster that were 
#       determined using the custom kmeans.py code.
#    2) It then adds the current price trend using the custom random_forest.py
#    3) Next it adds the price predictions for one day, one week, and one year using the 
#       custom regression.py
#    4) Finally, it finds the optimum portfolio percentage using the custom genetic algorithm
#       in the file optimizer.py

# imports
import flask
from flask import request, Response
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from datetime import date
import pickle
from random_forest import *
from optimizer import *

# initialize application
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# default route not used
@app.route('/', methods=['GET'])
def home():
    return "<p>no endpoint. use api/clusters instead.</p>"

# route starting the application and returning the data for coins by cluster.
@app.route('/api/clusters', methods=['GET'])
def api_clusters():
    # get cluster id
    query_parameters = request.args
    id = query_parameters.get('id')

    # load models for classification and regression
    classifier_filename = 'models/classification.sav'
    regressor_filename = 'models/SVR.sav'
    clf = pickle.load(open(classifier_filename, 'rb'))
    regressor = pickle.load(open(regressor_filename, 'rb'))

    # get clusters from kmeans
    df = pd.read_csv("cluster-data/clusters.csv")
    risk_df = df[df.cluster==int(id)].head(10)

    # get current price for display
    prices = []
    for index, row in risk_df.iterrows():
        symbol = row['coin'] + '-USD'
        try:
            current_price = get_current_price(symbol)[0]
            amount = "${:,.2f}".format(current_price)
            prices.append(amount)
        except:
            prices.append(0)
    risk_df['price'] = prices

    # get trend for classification
    trends = []
    for index, row in risk_df.iterrows():
        symbol = row['coin']
        col_names = ['symbol','Date','Close','pc_sma15c_sma30c', 'pc_open_close', 'pc_high_low', 'pc_low_close']
        file_name = 'price-data/' + symbol + '.csv'
        df = pd.read_csv(file_name)[col_names]

        # get last row for prediction
        x = df.iloc[-1]

        # predict trend
        if (clf.predict(x)):
            trend = 'up'
        else:
            trend = 'down'
 
        trends.append(trend)

    risk_df['trend'] = trends

    # get predicted prices for regression
    predicted_prices = []
    for index, row in risk_df.iterrows():
        symbol = row['coin']
        col_names = ['Date','Close']
        file_name = 'price-data/' + symbol + '.csv'
        df = pd.read_csv(file_name)[col_names]
        df = df.set_index(pd.DatetimeIndex(df["Date"].values))
        try:
            days = [1,7,365]
            preds = []
            for d in days:
                pred_price = get_predicted_price(df, regressor, d)
                amount = "${:,.2f}".format(pred_price)
                preds.append(amount)
            predicted_prices.append(preds)
        except:
            predicted_prices.append([0,0,0])

    pred_prices_df = pd.DataFrame(predicted_prices, columns=['oneday','oneweek','oneyear'])
    pred_prices_df['coin'] = risk_df['coin'].values
    final_df = pd.merge(risk_df,pred_prices_df)

    # run portfolio optimization
    # load porfolio data for optimization
    df = pd.read_csv('optimizer-data/annualreturns.csv') 
    R = pd.read_csv('optimizer-data/modelreturns.csv')

    coin_list = risk_df['coin'].values
    df = df[df.coin.isin(coin_list)]
    R = R[R.coin.isin(coin_list)]

    # set GA parameters
    populationSize = 10 #size of GA population
    generations = 1   #number of GA generations
    crossOverRate = 0.8  #crossover rate
    mutationRate = 0.2   #mutation rate
    elitismCount = 0   #number of parents to keep in each generation, 0 for no elitism

    # set initial population
    population = initializePopulation(R, df, populationSize)

    # loop generations
    for j in range(generations): 

        # genetic algorithm
        mates=tournamentSelection(population,3,populationSize)
        offspring = breeding(mates,populationSize,crossOverRate,mutationRate,R,df)
        population = insert(population,offspring,populationSize,elitismCount)

    # assign optimum portfolio
    final_df['optimum'] = population[len(population)-1][0]

    # return all data for coins in a cluster
    return Response(final_df.to_json(orient="records"), mimetype='application/json')

# get current prices by coin, used in User Interface
def get_current_price(symbol):
    yf.pdr_override()
    data = pdr.get_data_yahoo(symbol, start=date.today() - datetime.timedelta(1), end=date.today())
    return data['Close'].values

# get historical prices by coin, used in classification 
def get_historical(symbol):
    end_date = date.today()
    start_date = end_date - datetime.timedelta(365)
    yf.pdr_override()
    data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    return data

# get predicted prices by coin, used in regression
def get_predicted_price(df, model, days):
    X = np.array(df[["Close"]])
    X = X[:df.shape[0] - days]
    try:
        preds = model.predict(X)
        result = preds[len(preds)-1]
    except:
        result = 0
    return result

# run the app
app.run()