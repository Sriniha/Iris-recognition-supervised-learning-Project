# Supervised-Machine-Learning-Project

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head(10)

df.shape

train = df[0:15]

test = df[16:]

test.head()

x_train = train.drop('Scores',axis=1)

y_train = train['Scores']

x_test = test.drop('Scores',axis=1)

y_test = test['Scores']

from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

lr.score(x_test,y_test)

lr.score(x_test,pred)

lr.score(x_train,y_train)

rmse_test = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(pred)),2)))
rmse_test

rmse_train = np.sqrt(np.mean(np.power((np.array(y_train) - np.array(lr.predict(x_train))),2)))
rmse_train

df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})  
df 

