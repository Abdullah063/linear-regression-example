import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Student_Marks.csv')
print(df.head(5))
print(df.columns)
df=df.rename(columns={"number_courses": "sinif", "time_study": "saat","Marks":"puan"})
print(df.columns)

y = df[['puan']]
x= df[['sinif','saat']]

lm = LinearRegression()
model = lm.fit(x, y)

# Y = a1x1 + a2x2 + c

print(model.coef_)#katsayı  
print(model.intercept_)#sabit sayı

#a= model.predict([[4,5]])

a=((3*1.86405074) + (4.5*5.39917879) + -7.45634623 )
print(a)
print('*'*50)
df.info()

#import matplotlib.pyplot as plt
#plt.plot(df[['puan']])
