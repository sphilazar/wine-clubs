import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class KNN:
    def __init__(self,n=3): 
        self.model = KNeighborsRegressor(n_neighbors=n)
        self.X = None
        self.y = None
        self.labels = None # Just for fun, real target is continuous
        self.df = None
    
    def fit_model(self,df):
        self.df = df
        # self.labels = self.df.pop('Target')
        self.y = self.df.pop('clubLength')
        self.X = self.df.values
        self.model.fit(self.X,self.y)

    def get_predictions(self,df_test):
        # df_test = df_test.drop(['Target','clubLength'],axis=1)
        df_test = df_test.drop('clubLength',axis=1)
        return self.model.predict(df_test.values)

    def get_score(self,df_test):
        # df_test = df_test.drop(['Target'],axis=1)

        y = df_test.pop('clubLength')
        return self.model.score(df_test.values,y)

class LinReg:
    def __init__(self): 
        self.model = LinearRegression()
        # self.model = Lasso(alpha=1)
        self.X = None
        self.y = None
        self.labels = None # Just for fun, real target is continuous
        self.df = None

    def fit_model(self,df):
        self.df = df
        # print(self.df.shape)

        # self.labels = self.df.pop('Target')
        self.y = self.df.pop('clubLength')
        self.X = self.df.values
        # print(self.X.shape)

        # ss = StandardScaler()
        # ss.fit(self.X)
        # X = ss.transform(self.X)

        self.model.fit(self.X,self.y)
        
    def get_predictions(self,df_test):
        # df_test = df_test.drop(['Target','clubLength'],axis=1)
        df_test = df_test.drop('clubLength',axis=1)
        # print(df_test.shape)
        return self.model.predict(df_test.values)

    def get_score(self,df_test):
        # df_test = df_test.drop(['Target'],axis=1)
        y = df_test.pop('clubLength')

        return self.model.score(df_test.values,y)


    
