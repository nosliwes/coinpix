# CS 5593
# Steven Wilson
#
# Custom k-means algorithm
#
# The program can be run at the command line with the following arguments.
#
#         -data            : the full path to the transaction data
#         -k               : the number of clusters k 
#         -tol             : the centroid change tolerance between iterations before stopping
#         -max_iter        : the maximum iterations if tolerance is not reached
#         -keep_cols       : the columns to keep in the dataset

# import libraries
import argparse
import numpy as np
import pandas as pd
import random
import json

# custom k-means class
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=100):
        self.k = k                # k value
        self.tol = tol            # tolerance 
        self.max_iter = max_iter  # maximum iterations
        random.seed(22)           # set seed so clusters stay the same
        
    def fit(self,labels,data):
        # initialize centroids dictionary
        self.centroids = {}
        
        # upper bounds for random points of initial centroids
        data_length = len(data)

        # set starting centroids for each k
        for i in range(self.k):
            
            # get random index for starting centroids
            random_index = random.randint(0, len(data)-1)
            
            # set starting centroid for each value of k
            self.centroids[i] = data[random_index]
        
        # loop until max iterations reached or tolerance met
        for i in range(self.max_iter):
            
            # initilize empty clusters 
            self.clusters = {}
            self.sse = {}
            self.ids = {}
            self.names = {}
                
            for i in range(self.k):
                self.clusters[i] = []
                self.sse[i] = []
                self.ids[i] = []
                self.names[i] = []
            
            # find distances for each point from centroid
            for idx, x in enumerate(data):
                distances = [np.linalg.norm(x-self.centroids[centroid]) for centroid in self.centroids]
                
                # get cluster index for closest centroid
                cluster_index = distances.index(min(distances))
                
                # calculate sum squared error for point
                sse = min(distances)**2  
                
                # assign point and sse to cluster
                self.clusters[cluster_index].append(x)
                self.sse[cluster_index].append(sse)
                self.ids[cluster_index].append(idx)
                self.names[cluster_index].append(labels[idx])

            # copy current centroids to previous centroids variable to check tolerance
            prev_centroids = dict(self.centroids)

            # move centroid to center of each cluster
            for index in self.clusters:
                self.centroids[index] = np.average(self.clusters[index],axis=0)

            # initialize done flag
            done = True

            # check centroids for movement tolerance
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                
                # if centroid movement larger than tolerance mark done as false
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    done = False

            # otherwise, centroid movement is less than tolerance and stop
            if done:
                self.total_sse = 0
                for index in self.clusters:
                    self.total_sse += np.sum(self.sse[index])
                break   
                
        self.results = []
        for i in range(self.k):
            for name in self.names[i]:
                self.results.append((name, i))

    
    # method to display results
    def display_clusters(self):
        for idx, cluster in enumerate(self.ids):
            print("Cluster: ", idx)
            print("---------------------------------")
            print("Total Sum of Squared Error: ", np.sum(self.sse[idx]))
            print("Cluster Mean: ", self.centroids[cluster])
            print("Ids in Cluster: ", self.ids[cluster])
            print("Names in Cluster: ", self.names[cluster])     

# method to load data from file
def load_data(filepath, id_col, keep_cols):   
    # read csv data
    df = pd.read_csv(filepath)
    
    # get id columns
    ids = df[id_col]
    
    # select only keep columns
    df = df[keep_cols]

    # drop missing rows
    df.dropna(inplace=True)
    
    return ids.to_numpy(), df.to_numpy()            
            
# parse command-line arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=2, help="k value")
    parser.add_argument('-tol', type=float, default=0.001, help="tolerance level")
    parser.add_argument('-max_iter', type=float, default=100, help='maximum iterations')
    parser.add_argument('-data', type=str, default='returns.csv', help='Data filename')
    parser.add_argument('-id_col', type=str, default='coin')
    parser.add_argument('-keep_cols', nargs='+', type=str, default=['days_to_next_dividend', 'percent_change_price'], help='Column names to keep') 
    args = parser.parse_args()    
    return args

# main method to run program from command line
if __name__ == '__main__':

    # get arguments
    args = create_parser()  
    
    # load data and preprocess
    ids, X = load_data(filepath=args.data, id_col=args.id_col, keep_cols=args.keep_cols)
    
    # custom k-means
    clf = K_Means(k=args.k, tol=args.tol, max_iter=args.max_iter)
    clf.fit(ids, X)
    
    # show results
    # print("k =", args.k)
    # clf.display_clusters()
    
    print(clf.results)
    # assemble results
    df = pd.DataFrame(clf.results, columns=['coin','cluster'])
    
    # return results as json
    print(df.to_json()) 