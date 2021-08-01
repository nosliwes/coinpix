# CS 5593 - HW 4
# Andre' Robinson
#
# Custom ranodm forest algorithm
#
# The program can be run at the command line with the following arguments.
#
#         -data            : the full path to the transaction data
#         -keep_cols       : the columns to keep in the dataset
#         -label_col       : the column which represents the class label

# import libraries
import argparse
import numpy as np
import pandas as pd
import random

class Decision_Tree:
    def __init__(self, data=None):
        if not (data == None):
            self.fit(data)

# custom k-means class
class Random_Forest:
    def __init__(self):
        pass
        
    def fit(self,data):
        pass
    # method to save the model
    def save_model(self, filename):
        pass
# method to load data from file
def load_data(filepath, keep_cols, label_col = None):   
    # read csv data
    df = pd.read_csv(filepath)
    
    # select only keep columns
    df = df[keep_cols]

    # drop missing rows
    df.dropna(inplace=True)
    
    return df.to_numpy()
            
# parse command-line arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='dow_jones_index.data', help='Data filename')
    parser.add_argument('-keep_cols', nargs='+', type=str, default=['days_to_next_dividend', 'percent_change_price'], help='Column names to keep') 
    parser.add_argument('-label_col', type=str, default=None, help='Column which contains the labels')
    args = parser.parse_args()    
    return args

# main method to run program from command line
if __name__ == '__main__':

    # get arguments
    args = create_parser()  
    
    # load data and preprocess
    X = load_data(filepath=args.data, keep_cols=args.keep_cols, label_col=args.label_col)
    
    # custom k-means
    clf = K_Means(k=args.k, tol=args.tol, max_iter=args.max_iter)
    clf.fit(X)
    
    # show results
    print("k =", args.k)
    clf.display_clusters()