# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:21:26 2022

@author: smitha.s1_oob
"""

# AQI Prediction model
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor

data=pd.read_csv(r'C:\Users\Smitha.s1_oob\OneDrive - GEMS Education\Desktop\city_day.csv')
pickle.dump(data,open('data.pkl','wb'))
print('data pickled')

data.loc[:,'PM2.5':'Xylene']=data.groupby('City').transform(lambda x:x.fillna(x.mean()))
data.loc[:,'PM2.5':'Xylene']=data.fillna(data.mean())
print('Missing data filled')

data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].apply(lambda x:x.year)
data['month'] = data['Date'].apply(lambda x:x.month)
data['day'] = data['Date'].apply(lambda x:x.day)
data=data.drop(['Date','station','StationName','Status'],axis=1)
print('Initial data reduction')

#PM10 sub_index calculation
def get_PM10_subindex(x):
    if x<=50:
        return x
    elif x>50 and x<=100:
        return x
    elif x>100 and x<=250:
        return 100+(x-100)*100/150
    elif x>250 and x<=350:
        return 200+(x-250)
    elif x>350 and x<=430:
        return 300+(x-350)*100/80
    elif x>430:
        return 400+(x-430)*100/80
    else:
        return 0
data['PM10_subindex']=data['PM10'].astype(int).apply(lambda x:get_PM10_subindex(x))
print('PM10 si calculated')

def get_PM25_subindex(x):
    if x<=30:
        return x*50/30
    elif x>30 and x<=60:
        return 50+(x-30)*50/30
    elif x>60 and x<=90:
        return 100+(x-60)*100/30
    elif x>90 and x<=120:
        return 200+(x-90)*100/30
    elif x>120 and x<=250:
        return 300+(x-120)*100/130
    elif x>250:
        return 400+(x-250)*100/130
    else:
        return 0
data['PM2.5_subindex']=data['PM2.5'].astype(int).apply(lambda x:get_PM25_subindex(x))
print('PM5 si calculated')

def get_SO2_subindex(x):
    if x<=40:
        return x*50/40
    elif x>40 and x<=80:
        return 50+(x-40)*50/40
    elif x>80 and x<=380:
        return 100+(x-80)*100/300
    elif x>380 and x<=800:
        return 200+(x-380)*100/420
    elif x>800 and x<=1600:
        return 300+(x-800)*100/800
    elif x>1600:
        return 400+(x-1600)*100/800
    else:
        return 0
data['SO2_subindex']=data['SO2'].astype(int).apply(lambda x:get_SO2_subindex(x))
print('so2 si calculated')

def get_NOx_subindex(x):
    if x<=40:
        return x*50/40
    elif x>40 and x<=80:
        return 50+(x-40)*50/40
    elif x>80 and x<=180:
        return 100+(x-80)*100/100
    elif x>180 and x<=280:
        return 200+(x-180)*100/100
    elif x>280 and x<=400:
        return 300+(x-280)*100/120
    elif x>400:
        return 400+(x-400)*100/120
    else:
        return 0
data['NOx_subindex']=data['NOx'].astype(int).apply(lambda x:get_NOx_subindex(x))
print('NOx si calculated')

def get_NH3_subindex(x):
    if x<=200:
        return x*50/200
    elif x>200 and x<=400:
        return 50+(x-200)*50/200
    elif x>400 and x<=800:
        return 100+(x-400)*100/400
    elif x>800 and x<=1200:
        return 200+(x-800)*100/400
    elif x>1200 and x<=1800:
        return 300+(x-1200)*100/600
    elif x>1800:
        return 400+(x-1800)*100/600
    else:
        return 0
data['NH3_subindex']=data['NH3'].astype(int).apply(lambda x:get_NH3_subindex(x))
print('NH3 si calculated')

def get_CO_subindex(x):
    if x<=1:
        return x*50/1
    elif x>1 and x<=2:
        return 50+(x-1)*50/1
    elif x>2 and x<=10:
        return 100+(x-2)*100/8
    elif x>10 and x<=17:
        return 200+(x-10)*100/7
    elif x>17 and x<=34:
        return 300+(x-17)*100/17
    elif x>34:
        return 400+(x-34)*100/17
    else:
        return 0
data['CO_subindex']=data['CO'].astype(int).apply(lambda x:get_CO_subindex(x))
print('co si calculated')

def get_O3_subindex(x):
    if x<=50:
        return x*50/50
    elif x>50 and x<=100:
        return 50+(x-50)*50/50
    elif x>100 and x<=168:
        return 100+(x-100)*100/68
    elif x>168 and x<=208:
        return 200+(x-168)*100/40
    elif x>208 and x<=748:
        return 300+(x-208)*100/539
    elif x>748:
        return 400+(x-400)*100/539
    else:
        return 0
print(data.head())
data['O3_subindex']=data['O3'].astype(int).apply(lambda x:get_O3_subindex(x))
print('o3 calculated')

data['AQI']=data['AQI'].fillna(round(data[['PM2.5_subindex','PM10_subindex','SO2_subindex','NOx_subindex','NH3_subindex','CO_subindex','O3_subindex']].max(axis=1)))
print('AQI calculated')
data.to_csv('city_day1.csv')

for features in ['Benzene','CO','NH3','NO','NO2','PM2.5','PM10','NOx','O3','SO2','Toluene','Xylene','AQI','CO_subindex','NH3_subindex','NOx_subindex','O3_subindex','PM10_subindex','PM2.5_subindex','SO2_subindex']:
         OF_Q1=data[features].quantile(0.25)
         OF_Q2=data[features].quantile(0.50)
         OF_Q3=data[features].quantile(0.75)
         OF_IQR=OF_Q3-OF_Q1
         OF_low_limit=OF_Q1-1.5*OF_IQR
         OF_up_limit=OF_Q3+1.5*OF_IQR
         OF_outlier=data[(data[features]<OF_low_limit)|(data[features]>OF_low_limit)]
         data[features]=data[features].clip(OF_up_limit,OF_low_limit)

data['State'].replace({'Gujarat':0, 'Mizoram':1, 'Andhra Pradesh':2, 'Punjab':3, 'Karnataka':4,\
                       'Madhya Pradesh':5, 'Odisha':6, 'Chandigarh':7, 'Tamilnadu':8, 'Delhi':9,\
                       'Kerala':10, 'Haryana':11, 'Assam':12, 'Telengana':13, 'Rajasthan':14,\
                       'Jharkhand':15, 'West Bengal':16, 'Uttar Pradesh':17,'Maharashtra':18,\
                       'Bihar':19, 'Meghalaya':20},inplace=True)
data['City'].replace({'Ahmedabad':0, 'Aizawl':1, 'Amaravati':2, 'Amritsar':3, 'Bengaluru':4,\
                      'Bhopal':5, 'Brajrajnagar':6, 'Chandigarh':7, 'Chennai':8, 'Coimbatore':9,\
                      'Delhi':10, 'Ernakulam':11, 'Gurugram':12, 'Guwahati':13, 'Hyderabad':14,\
                      'Jaipur':15, 'Jorapokhar':16, 'Kochi':17, 'Kolkata':18, 'Lucknow':19, 'Mumbai':20,\
                      'Patna':21, 'Shillong':22, 'Talcher':23,'Thiruvananthapuram':24,'Visakhapatnam':25},inplace=True)
data['month_sin'] = np.sin(data['month']*(2.*np.pi/12))
data['month_cos'] = np.cos(data['month']*(2.*np.pi/12))
#data['hour_sin'] = np.sin(data['hour']*(2.*np.pi/24))
#data['hour_cos'] = np.cos(data['hour']*(2.*np.pi/24))

y=data['AQI']
X = data[['State','City','year', 'month_sin','month_cos']]
#X=X.drop(['year','PM10_subindex'], axis=1)
print(X.columns)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)    
model_cat=CatBoostRegressor(verbose=0)
model_cat.fit(X_train, y_train)
#model_lgbm=LGBMRegressor()
#model_lgbm.fit(X_train,y_train)
#Saving the model to disk
pickle.dump(model_cat,open('model.pkl','wb'))
print('Model pickled')

