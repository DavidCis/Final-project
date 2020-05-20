![Oil](https://github.com/DavidCis/Final-project/tree/master/Images/854081161001_6154929188001_6154933434001-vs.jpg)

# Crude oil price prediction 2020 (Final Project)

**David Alfredo Cisneros Diaz**

**Data Analytics Bootcamp**


## Overview

My goal with this project was using supervised machine learning to predcit the price of the crude oil WTI (West Texas Intermediate) this stock is used to stablish the oil price.

I set this goal because of the uncertainty of the stock market with the COVID-19, a virus that has stopped the world economy. On April 24th we saw the biggest fall of this stock since 2009.The stock went down -309.97%. The stock market opened that day with a price of 17.73 usd and at the end of the day it was -37.63 usd.

My first idea was to analyze how this stock behaved in another pandemic.The closest one that afected the world economy like COVID-19 was H1N1 (January 2009-July 2010).


##
## Data

* [Historical Data](https://mx.investing.com/commodities/crude-oil-historical-data)

I got 90% of my data from investing.com, is a website where you can watch all the stock marcket in real time, it has the data from commodities to criptocurrency.
All data were clean but i wanted just the prices columns and dates to start running and testing the time-series regresions.


##
## AutoRegressive model (Ar)

An autoregressive (AR) model predicts future behavior based on past behavior. It’s used for forecasting when there is some correlation between values in a time series and the values that precede and succeed them.
This was my first model with very good results, with an error fo 10.43 but the last prediction had a 34.58 point of difference with the real one.

![AR](https://github.com/DavidCis/Final-project/tree/master/Images/ar.png)


##
## SARIMAX model (Seasonal AutoRegressive Integrated Moving Average Exogenous)

Then i used SARIMAX because the pandemics are seasonal and the stocks are volatile. This model is useful in cases we suspect that residuals may exhibit a seasonal trend or pattern.
But the results were similar to AR.

![SARIMAX](https://github.com/DavidCis/Final-project/tree/master/Images/sarimax.png)

##
## Prophet

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
Prophet could had been a very good option but my data is monthly, so i couldnt use prophet as its best.

Green dots are the real data, the blue line is prophet precdcitions and the soft blue around it is the area of error.

![Prophet](https://github.com/DavidCis/Final-project/tree/master/Images/prophet.png)


##
## GBR model (Gradient Boosting Regressors)

Gradient boosting regressors are a type of inductively generated tree ensemble model. At each step, a new tree is trained against the negative gradient of the loss function, which is analogous to (or identical to, in the case of least-squares error) the residual error.
With this model i used HyperOpt to fit the model with the best hyper-parameters for my data.
The MSE (Mean Squared Error) of this model is 6.9 and the train and test r2 are: 0.999904 , 0.9912.

![gbr1](https://github.com/DavidCis/Final-project/tree/master/Images/tail_gbr.png)

The mean error difference is close to 0.

![gbr2](https://github.com/DavidCis/Final-project/tree/master//Images/describe_gbr.png)


##
## Visualisation (Tableu)

Pandemics from 2002 to 2020 with oil prices at the start and at the end of the pandemic.


![Pandemics](https://github.com/DavidCis/Final-project/tree/master/Images/price_pandemics.png)


This is how price behaved during H1N1.


![H1N1](https://github.com/DavidCis/Final-project/tree/master/Images/h1n1.png)


Acurracy of my machine learning model. (Predicted data vs real data)


![Error](https://github.com/DavidCis/Final-project/tree/master/Images/realvserror.png)


This are my predictions for the price of the crude oil WTI for 2020.


![Predict](https://github.com/DavidCis/Final-project/tree/master/Images/prediction2020.png)


##
## Today_oil_price

Is a function to get the real time data of the price of the crude oil WTI and the prices of the stock from the last month.
