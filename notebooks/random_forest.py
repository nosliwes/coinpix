# CS 5593 - Coinpix
# Andre' Robinson
#
# Custom random forest algorithm
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
import pickle

# get gini impurity of a df. only works with boolean labels
def gini_impurity(df, label_col):
    true_labels = df[df[label_col] == True]
    false_labels = df[df[label_col] == False]
    freq_true = len(true_labels)
    freq_false = len(false_labels)
    denom = freq_true + freq_false
    if denom==0:
        gini = 1.0
    else:
        gini = (2 * freq_true * freq_false) / float(denom**2)
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
    l_wgt = len(l_df)
    r_wgt = len(r_df)
    l_gini = gini_impurity(l_df, label_col)
    r_gini = gini_impurity(r_df, label_col)
    gini = (l_wgt * l_gini + r_wgt * r_gini) / (l_wgt + r_wgt)
    return gini
    

class Decision_Tree:
    def __init__(self, data=None, label_col = None, min_records = 5, gini_cutoff = 0.01):
        self.min_records = min_records
        self.left = None
        self.right = None
        self.label = None
        self.gini_cutoff = gini_cutoff
        self.data = data
        self.label_col = label_col
        
    def fit(self,data=None, label_col=None, keep_data = True):
        if data == None:
            data = self.data
        if label_col == None:
            label_col = self.label_col
        self.data = data
        self.label_col = label_col
        if (len(data.index) > self.min_records) and (gini_impurity(self.data, self.label_col) > self.gini_cutoff):
            self.left, self.right, self.split_col, self.split_val = self.find_best_split()
            if not keep_data:
                self.data = None
            self.left.fit(keep_data = False)
            self.right.fit(keep_data = False)
        else:
            # extract the mode label
            self.label = bool(self.data[label_col].mode().iloc[0])
    
    def predict(self, datum, label_col = None):
        if label_col == None:
            label_col = self.label_col
        if self.label == None:
            if  datum[self.split_col] >= self.split_val:
                return self.right.predict(datum, label_col)
            else:
                return self.left.predict(datum, label_col)
        else:
            return self.label
    
    def find_best_split(self):
        column = 'No_column'
        value = 0
        gini = 1.0
        l = None
        r = None
        # X is the dataframe excluding labels
        X = self.data[self.data.columns.difference([self.label_col])]
        # find best split column and value
        for idx, row in X.iterrows():
            for col in X.columns:
                l_gini = gini_on_split(self.data, self.label_col, col, row[col])
                if l_gini < gini:
                    gini = l_gini
                    value = row[col]
                    column = col
        #generate left and right children from those values
        l = Decision_Tree(self.data[self.data[column] >= value], self.label_col)
        r = Decision_Tree(self.data[self.data[column] < value], self.label_col)
        return l, r, column, value
    
    # delete dataframes from model to save memory
    def clear_dataframes(self):
        if self.left != None:
            self.left.clear_dataframes()
        if self.right != None:
            self.right.clear_dataframes()
        if self.data != None:
            self.data = None
# custom random forest class
class Random_Forest:
    def __init__(self, data, label_column, min_records=5, n_trees=100, row_prop = 0.1, max_row_samples = 100, column_prop = 0.50):
        self.trees = []
        self.data = data
        self.label_column = label_column
        self.min_records = min_records
        self.n_trees = n_trees
        self.row_prop = row_prop
        self.max_row_samples = max_row_samples
        self.column_prop = column_prop
        pass
        
    def fit(self,data=None):
        if data is None:
            data = self.data
        self.trees.clear()
        for i in range(self.n_trees):
            # get a random sample of rows
            random_sample = data.sample(frac = self.row_prop, replace=True)
            sample_labels = [label for label in random_sample[self.label_column]]
            random_sample = random_sample.sample(frac = self.column_prop, axis='columns')
            random_sample[self.label_column] = sample_labels
            random_tree = Decision_Tree(data = random_sample, label_col = self.label_column, min_records = self.min_records)
            random_tree.fit(keep_data = False)
            self.trees.append(random_tree)
            print('tree', i, 'of', self.n_trees,)
    
    def predict(self, datum):
        vote_count = 0
        for tree in self.trees:
            if tree.predict(datum):
                vote_count = vote_count + 1
        threshold = self.n_trees / 2
        return vote_count > threshold
    
    # method to save the model
    def save_model(self, filename):
        # model file would be quite large otherwise
        for tree in self.trees:
            tree.clear_dataframes()
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        
# method to load data from file
def load_data(filepath, keep_cols, label_col = None):   
    # read csv data
    df = pd.read_csv(filepath)
    
    # select only keep columns
    df = df[keep_cols]

    # drop missing rows
    df.dropna(inplace=True)
    
    return df
            
# parse command-line arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='dow_jones_index.data', help='Data filename')
    parser.add_argument('-keep_cols', nargs='+', type=str, default=['days_to_next_dividend', 'percent_change_price'], help='Column names to keep') 
    parser.add_argument('-label_col', type=str, default=None, help='Column which contains the labels. Should also be contained in keep_cols')
    parser.add_argument('-model_file', type=str, default=None, help='Name of file to save the model to')
    args = parser.parse_args()    
    return args

# main method to run program from command line
if __name__ == '__main__':

    # get arguments
    args = create_parser()  
    
    # load data and preprocess
    X = load_data(filepath=args.data, keep_cols=args.keep_cols, label_col=args.label_col)
    
    print(X.head())
    
    # custom random forest
    clf = Random_Forest(X, args.label_col)