# CS 5593
# Steven Wilson
#
# Genetic Algorithm used for portfolio optimization

# import libraries
import pandas as pd
import numpy as np
from random import Random
import matplotlib.pyplot as plt
from progressbar import ProgressBar

# random seed for repeatable results
seed = 4621
myPRNG = Random(seed)

# create a continuous valued chromosome 
def createChromosome(d, lBnd, uBnd):   
    x = []
    for i in range(d):
        x.append(myPRNG.uniform(lBnd,uBnd))   #creating a randomly located solution
    
    # normalize values for Markowitz constraint
    for i in range(len(x)):
        x[i] = x[i] / sum(x)
      
    return x

# create initial population
def initializePopulation(): 
    population = []
    populationFitness = []
    
    print("Initializing Population")
    pbar = ProgressBar()
    for i in pbar(range(populationSize)):
        population.append(createChromosome(R['coin'].count(),0,1))
        populationFitness.append(evaluate(population[i]))
        
    tempZip = zip(population, populationFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])
    
    return popVals   

# helper method to calculate covariance used in sharpe ratio
def covariance(x, y):
    symbol_x = df['coin'].unique()[x]
    symbol_y = df['coin'].unique()[y]   
    val_x = df[df['coin']==symbol_x]['return'].values
    val_y = df[df['coin']==symbol_y]['return'].values
    cov = (val_x * val_y).mean() - (val_x.mean() * val_y.mean())
    return cov

# function to evaluate the Sharpe ratio of the portfolio
def evaluate(w):     
    
    Rp = 0 # portfolio return 
    Rf = 1.0 # risk free return
    sigma = 0 # portfolio variance
    
     # calculate portfolio return
    for i in range(len(w)):
        Rp = Rp + R['return'].iloc[i] * w[i]
        
    # calculate portfolio variance
    for i in range(len(w)):
        for j in range(len(w)):
            sigma = sigma + (covariance(i,j) * w[i] * w[j])
            
    sharpe = (Rp - Rf) / sigma
    
    frontierData.append((Rp,sigma,sharpe))
    
    return sharpe   

# performs tournament selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
# function returns the mating pool with size equal to the initial population
def tournamentSelection(pop,k):
    
    #randomly select k chromosomes; the best joins the mating pool
    matingPool = []
    
    print("Tournament Selection")
    while len(matingPool)<populationSize:
        
        ids = [myPRNG.randint(0,populationSize-1) for i in range(k)]
        competingIndividuals = [pop[i][1] for i in ids]
        bestID=ids[competingIndividuals.index(max(competingIndividuals))]
        matingPool.append(pop[bestID][0])

    return matingPool

def breeding(matingPool):
    children = []
    childrenFitness = []
    
    print("Breeding Offspring")
    pbar = ProgressBar()
    for i in pbar(range(0,populationSize-1,2)):

        child1,child2=crossover(matingPool[i],matingPool[i+1])
        
        child1=mutate(child1)
        child2=mutate(child2)
        
        # normalize values for Markowitz constraint
        for i in range(len(child1)):
            child1[i] = child1[i] / sum(child1)
            child2[i] = child2[i] / sum(child2)
        
        children.append(child1)
        children.append(child2)
        
        childrenFitness.append(evaluate(child1))
        childrenFitness.append(evaluate(child2))
        
    tempZip = zip(children, childrenFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])
    
    return popVals

# implement a linear crossover
def crossover(x1,x2):
    # randomly generate probability of crossover
    probability = myPRNG.uniform(0, 1)
      
    if probability <= crossOverRate: 

        #choose the crossover point so that at least 1 element of parent is copied
        crossOverPt = myPRNG.randint(1,len(x1)-1) 

        beta = myPRNG.random()  #random number between 0 and 1

        #create the linear combination of the solutions
        new1 = list(np.array(x1) - beta*(np.array(x1)-np.array(x2))) 
        new2 = list(np.array(x2) + beta*(np.array(x1)-np.array(x2)))

        #perfrom the crossover between the original solutions "x1" and "x2" and the "new1" and "new2" solutions
        if crossOverPt < len(x1)/2:    
            offspring1 = x1[0:crossOverPt] + new1[crossOverPt:len(x1)]
            offspring2 = x2[0:crossOverPt] + new2[crossOverPt:len(x1)]
        else:
            offspring1 = new1[0:crossOverPt] + x1[crossOverPt:len(x1)]
            offspring2 = new2[0:crossOverPt] + x2[crossOverPt:len(x1)]        
    else:
        offspring1 = x1
        offspring2 = x2
      
    return offspring1, offspring2

# mutate function 
def mutate(x):
    # randomly generate probability of mutation
    probability = myPRNG.uniform(0, 1)
    
    # mutate if less than mutationRate
    if probability <= mutationRate:
        # random position to mutate
        index = myPRNG.randint(0, len(x)-1)

        # solution feasbile flag (0 infeasible, 1 feasible)
        flag = 0 

        # loop until feasible mutation found
        while flag == 0:
            x[index] = myPRNG.uniform(0, 1)
            if (evaluate(x) > 0):
                flag = 1
    return x

# insertion step using elitism strategy
def insert(pop,kids):
    
    # sort parents for elitism strategy
    popVals = sorted(pop, key=lambda pop: pop[1])
    
    # sort kids for elistim strategy
    kidVals = sorted(kids, key=lambda kids: kids[1])
   
    # combine top parents with top kids for next generation
    population = popVals[:elitismCount] + kidVals[:(populationSize-elitismCount)]
    
    return population 

# display fitness of population
def summaryFitness(pop):
    a=np.array(list(zip(*pop))[1])
    return np.max(a), np.mean(a), np.var(a)

# display best solution
def bestSolutionInPopulation(pop):
    print ("Best Portfolio: " , pop[len(population)-1])