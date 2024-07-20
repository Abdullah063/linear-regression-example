# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:42:33 2023

@author: ABDULLAH
"""
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Audi.csv')
print(df.head(5))
print(df.columns)

#sütün silme işlemleri 
newFrame  = df.drop(columns=['index','Score', 'PPYRank','PriceRank','href','MileageRank'])

print(newFrame)
print(newFrame.columns)

#-----------------------format değişştirme işlemleri 
newFrame.info()
#replace yapma 
print(newFrame.head())
newFrame['Engine']=newFrame['Engine'].str.replace('L','')
#to_numineric 
newFrame['Engine']=pd.to_numeric(newFrame['Engine'])

# kategorilendirme 
print(newFrame.columns)
newFrame= pd.get_dummies(newFrame,columns=['Type','Transmission','Fuel'],drop_first=True)# evet hayır için yeni sütünlerr olıuşacak
newFrame.info()
print(newFrame.columns)


# y = ax +b
y = newFrame['Price(£)']
x = newFrame.drop("Price(£)",axis=1)

lm = LinearRegression()
model = lm.fit(x, y)
newFrame.info()
tahmin = model.predict([[2018,44000,1.6,110,1,2600,0,1]])
print(tahmin)

print(model.score(x, y)) # başarı oranı 