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

# get gini impurity of a df. only works with boolean labels
def gini_impurity(df, label_col):
    true_labels = df[df['lable_col'] == True]
    false_labels = df[df['label_col'] == False]
    freq_true = len(true_labels.index)
    freq_false = len(false_labesl.index)
    denom = freq_true + freq_false
    gini = 2 * freq_true * freq_false / denom
    return gini

def gini_on_split(df, label_col, split_col, split_val, split_type = 'num'):
    gini = 0.5
    l_df = None
    r_df = None
    if split_type == 'num':
        l_df = df[df[split_col] < split_val]
        r_df = df[df[split_col] >= split_val]
    else:
        l_df = df[df[split_col] == split_val]
        r_df = df[df[split_col] != split_val]
    l_wgt = len(l_df.index)
    r_wgt = len(r_df.index)
    l_gini = gini_impurity(l_df, label_col)
    r_gini = gini_impurity(r_df, label_col)
    gini = (l_wgt * l_gini + r_wgt * r_gini) / (l_wgt + r_wgt)
    return gini
    

class Decision_Tree:
    def __init__(self, data=None, label_col == None, min_records = 5):
        self.min_records = min_records
        self.prediction = labels.mode()
        self.left = None
        self.right = None
        self.label = None
        if data != None and labels != None:
            self.fit(data, labels)
        
    def fit(self,data=self.data, label_col=self.label_col):
        self.data = data
        self.label_col = label_col
        if len(data.index) > min_records:
            self.left, self.right, self.split_col, self.split_val 
                = self.find_best_split()
            self.left.fit()
            self.right.fit()
        else:
            # extract the mode label
            self.label = self.data[label_col].mode().iloc[0][label_col]
    
    def predict(self, datum, label_col = self.label_col):
        if self.label == None:
            if  datum[label_col] >= datum[self.split_col]:
                return self.right.predict(datum, label_col)
            else:
                return self.left.predict(datum, label_col)
        else:
            return self.label
    
    def find_best_fit(self):
        
        

# custom random forest class
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