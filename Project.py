import pandas as pd
import numpy as np
from datetime import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from sklearn.ensemble import GradientBoostingRegressor as GBR
from functools import reduce
import warnings
warnings.simplefilter('ignore')

#Opening all data from scrapping
crude=pd.read_csv('data/Cushing, OK WTI Spot Price FOB (Dollars per Barrel).csv')
refiner=pd.read_csv('data/U.S. Crude Oil Composite Acquisition Cost by Refiners (Dollars per Barrel).csv')
usd=pd.read_csv('data/usd.csv')
pan=pd.read_csv('data/Pandemics.csv')
stocks=pd.read_csv('data/stocks.csv')
preds=pd.read_csv('data/preds.csv')

def clean(df,x): #This function cleans data and replace months from spanish to english.

    df.drop(['Apertura','Máximo','Mínimo','Vol.','% var.'],axis=1,inplace=True)

    df.rename(columns={'Fecha': 'Date', 'Cierre': x}, inplace=True)

    df['Date']=df.Date.apply(lambda x:x.replace('Ene','Jan'))
    df['Date']=df.Date.apply(lambda x:x.replace('Dic','Dec'))
    df['Date']=df.Date.apply(lambda x:x.replace('Abr','Apr'))
    df['Date']=df.Date.apply(lambda x:x.replace('Ago','Aug'))
    df['Date']=pd.to_datetime(df.Date)

    df.sort_values(by='Date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    df=df.set_index('Date')

clean(stocks,'Stocks_price_usd')
clean(usd,'usd')
clean(preds,'Futuro')

def clean2(df,x): 
    df.rename(columns={df.columns[1]: x}, inplace=True)
    df['Date']=pd.to_datetime(df.Date)
    df= df.set_index('Date')   

clean2(crude,'Crude_oil_price_usd')
clean2(refiner,'Refiners_Cost_usd')

pan=pan.fillna('non') #This is a boolean dataset, i had to change the data to numbers
pan['Pandemics'] = np.where(pan['Pandemics']=='non',0,1)
pan['Date']=pd.to_datetime(pan.Date)
pan= pan.set_index('Date')

dfs= [crude,refiner,usd,stocks,pan] #Merge all the dataframes into one
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],how='outer'), dfs)
df_merged.drop(crude.tail(1).index,inplace=True)
df_merged=df_merged.set_index('Date')

train, test = df_merged.Stocks_price_usd[:-10], df_merged.Stocks_price_usd[-10:] #Using SARIMAX and AR for time series regresion

modelo=SARIMAX(train, order=(20, 1, 5)).fit(disp=False)
pred=modelo.predict(len(train), len(df_merged.Stocks_price_usd)-1)     
error=(pred-test).abs().sum()/len(pred) 
res=pd.DataFrame({'real':test, 'pred':pred, 'diff':pred-test})


modelo=AR(train).fit()
pred=modelo.predict(len(train), len(df_merged.Stocks_price_usd)-1)
error=(pred-test).abs().sum()/len(pred)
res=pd.DataFrame({'real':test, 'pred':pred, 'diff':pred-test})


dffinal=df_merged.copy() #Making a copy for the final product

df_merged.reset_index(level=0, inplace=True) #Transforming all data into numbers
df_merged['Month']=df_merged.Date.dt.month
df_merged['Year']=df_merged.Date.dt.year
df_merged.drop(['Date'],axis=1,inplace=True)

X=df_merged.drop('Stocks_price_usd', axis=1)
y=df_merged.Stocks_price_usd
X_train, X_test, y_train, y_test=tts(X, y)

#Hyper parameters of GBR
hyper={
    'n_estimators':hp.quniform('n_estimators', 10, 10000, 10),
    
    'learning_rate':hp.uniform('learning_rate', 0.0001, 1.0),
    
    'subsample':hp.uniform('x_subsample', 0.5, 1),
    
    'alpha':hp.uniform('x_alpha', 0.5, 0.9),
    
    'validation_fraction':hp.uniform('x_validation_fraction', 0.1, 0.4)
}

def goal(hyper):
    
    modelo=GBR(
        n_estimators=int(hyper['n_estimators']),
        learning_rate=hyper['learning_rate'],
        subsample=hyper['subsample'],
        alpha=hyper['alpha'],
        validation_fraction=hyper['validation_fraction']
    
    )
    
    eval_set=[(X_train, y_train), (X_test, y_test)]
    
    modelo.fit(X_train, y_train)
    
    y_pred=modelo.predict(X_test)
    
    rmse=mse(y_test, y_pred)**0.5
    
    return {'loss':rmse, 'status':STATUS_OK}

trials_reg=Trials() 
best=fmin(fn=goal, space=hyper, algo=tpe.suggest, max_evals=50, trials=Trials())    

modelo=GBR(
    n_estimators=int(best['n_estimators']),
    learning_rate=best['learning_rate'],
    subsample= best['x_subsample'],
    alpha=best['x_alpha'],
    validation_fraction=best['x_validation_fraction'],
    )

modelo.fit(X_train, y_train) #MSE
y_pred=modelo.predict(X_test)
print('Mean squared error',mse(y_test, y_pred))    

train_score=modelo.score(X_train, y_train) #R2
test_score=modelo.score(X_test, y_test)
print ('train R2:',train_score, '-- test R2:', test_score)

pred=df_merged.drop('Stocks_price_usd', axis=1) #Using trained model to predict data
res=modelo.predict(pred)

dffinal.drop(['Refiners_Cost_usd','usd','Stocks_price_usd','Pandemics'],axis=1,inplace=True) # 3 column df to show results
dffinal.rename(columns={'Crude_oil_price_usd': 'Real'}, inplace=True)
dffinal['Predicitions']=res
dffinal['Difference']=dffinal.Predicitions-dffinal.Real

preds.insert(1, "Refiners_Cost_usd", [32.72, 25.80, 29.1], True) # creating df to predict prices in the future
preds.insert(2, "usd", [99.92, 99.028, 99.48], True)
preds.insert(3, "Pandemics", [1, 1, 1], True)
preds.insert(0, "Crude_oil_price_usd", [29.21, 15.10, 26.5], True)
preds.insert(4, "Month", [3, 4, 5], True)
preds.insert(5, "Year", [2020, 2020, 2020], True)
df2 = pd.DataFrame({"Crude_oil_price_usd":[28.4,30.7,32.9,34.5,37.1,38.1,38.5,37.1,38.3], 
                    "Refiners_Cost_usd":[31.4,32.7,34.9,36.5,39.1,40.1,38.5,38.1,40.3],
                    "usd":[99.4,99.62,99.51,99.6,99.5,99.85,99.53,99.62,99.81],
                    "Month":[6,7,8,9,10,11,12,1,2],
                    "Year":[2020,2020,2020,2020,2020,2020,2020,2021,2021],
                    "Pandemics":[1,1,1,1,1,1,1,1,1],}) 
preds.drop('Date',axis=1,inplace=True)
preds=preds.append(df2, ignore_index = True)
futuro=preds.Futuro
preds.drop('Futuro',axis=1,inplace=True)
res=modelo.predict(preds)
preds['Predicitions']=res

print(dffinal)
print(preds)