import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve,roc_auc_score
from  sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class ChurnModel:
    def __init__(self,columns):
        self.model = None
        self.columns = list(set(columns)-set("clubLength"))
        self.labels = None

    def fit_dumb_model(self,df):
        X = df["Club Length"].values.reshape(-1,1)
        y = df["Target"].values.astype(int)
        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=16)
        model = LogisticRegression(solver='lbfgs')
        self.model =  model
        self.model.fit(X,y)

    def fit_a_model(self,df):
        '''
        Logistic regression across multiple variables
        '''
        X = df[self.columns].values
        y = df["Target"].values.astype(int)
        

        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=16)
        model = LogisticRegression(penalty='l1')
        self.model = model
        self.model.fit(X,y)

    def fit_random_forest(self,df):
        X = df[self.columns].values
        y = df.pop("clubLength")
        # print(X)
        # print(self.columns)

        #flip OOB score to true
        rf = RandomForestRegressor()
        self.model = rf
        self.model.fit(X,y)


    def fit_gradient_boosted_forest(self,df,n_estimators=100,learning_rate=0.005):
        X = df[self.columns].values
        y = df.pop("clubLength")
        # print(X)
        
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=4,
                                    learning_rate=learning_rate,
                                    random_state=64)
        self.model.fit(X,y)

    def get_predictions(self,df_test,df_train):
        X_test = df_test[self.columns].values
        self.labels = df_test["clubLength"].values.astype(int) #y_test
        X_train = df_train[self.columns].values
        y_train = df_train["clubLength"].values.astype(int)

        yhat = self.model.predict(X_test)
        predictions = self.model.predict(X_test)
        score = self.model.score(X_test,self.labels)
        cv_scores = cross_val_score(self.model,X_train,y_train,cv=5)
        return yhat,predictions,score,cv_scores

    def get_roc_curve(self,probabilities,n=100):

        #just in case labels are 1's and 0's
        # self.labels = self.labels.astype(bool)
        auc = roc_auc_score(self.labels,probabilities)
        TPRs = []
        FPRs = []
        Ts = []
        for threshold in np.linspace(0,1,n):

            #set threshold boolean mask
            y_predict = probabilities > threshold

            #calculate confusion matrix stuff
            tp = (y_predict & self.labels).sum()
            fp = (y_predict & ~self.labels).sum()
            tn = (~y_predict & ~self.labels).sum()
            fn = (~y_predict & self.labels).sum()

            #calc tpr and fpr for ROC curve
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            #add tpr and fpr to lists
            TPRs.append(tpr)
            FPRs.append(fpr)
            Ts.append(threshold)


        plt.figure()
        plt.plot(FPRs,TPRs,color='r')
        plt.plot(Ts,Ts, 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for wine club subscription attrition (Gradient Boosting)')
        plt.grid(True)
        plt.show()

        return auc,np.array(TPRs), np.array(FPRs), np.array(Ts)

import pickle

def get_pickle(model):
    pickle.dump(model, open('model.p', 'wb'))

