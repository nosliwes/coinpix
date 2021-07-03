# CoinPIX

CS 5593 Data Mining Project
Summer 2021

This project selects cryptocurrency coins based on a chosen risk tolerance.  This is accomplished by using the following methodology.

Stage 1: Kmeans clustering is used to group coins into 4 different risk clusters.  A user then chooses a risk profile they would like to invest from the following categories.
1. High
2. Somewhat High
3. Somewhat Low
4. Low

Stage 2: Following the choice of risk profile, a random forest classification model is applied that labels each coin in the cluster as moving either up or down in price.

Stage 3: A regression model is then applied to predict the number of days left in the current trend.

Stage 4: The last step uses a genetic algorithm to find the optimum porfolio of coins from the cluster to maximize returns and minimize variance.  



