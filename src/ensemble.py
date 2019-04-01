import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve,roc_auc_score
from  sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class EnsembleChurnModel:
    def __init__(self,columns):
        self.model = None
        self.columns = columns
        self.labels = None

    # Split customers into group A or B (how many distinct clusters?)

    def split_customers(self,df,n=2):
        pass

    # If customer clustered into group A

    def fit_model_A(self,df):
        '''
        Logistic regression across multiple variables
        '''
        X = df[self.columns].values
        y = df["Target"].values.astype(int)
        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=16)
        model = LogisticRegression(penalty='l1')
        self.model = model
        self.model.fit(X,y)

    # If customer clustered into group B

    def fit_model_B(self,df):
        '''
        Logistic regression across multiple variables
        '''
        X = df[self.columns].values
        y = df["Target"].values.astype(int)
        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=16)
        model = LogisticRegression(penalty='l1')
        self.model = model
        self.model.fit(X,y)

    # How to reconcile A predictions/ B predictions? Mask customers 

    def get_predictions(self,df_test,df_train):
        X_test = df_test[self.columns].values
        self.labels = df_test["Target"].values.astype(int) #y_test
        X_train = df_train[self.columns].values
        y_train = df_train["Target"].values.astype(int)

        yhat = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)
        score = self.model.score(X_test,self.labels)
        cv_scores = cross_val_score(self.model,X_train,y_train,cv=5)
        return yhat,probas,score,cv_scores