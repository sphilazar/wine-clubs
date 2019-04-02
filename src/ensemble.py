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
    def __init__(self,columns,Xs,ys):
        self.model = None
        self.columns = columns
        self.labels = []
        self.Xs = Xs
        self.ys = ys
        self.n_models = len(self.Xs)
        self.log_models = []
        self.rf_models = []
        self.boosted_models = []

    def fit_models(self):
        '''
        Logistic regression across multiple variables
        '''
        for n in range(self.n_models):
            X = self.Xs[n][self.columns].values
            y = self.ys[n].values.astype(int)
            # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=16)

            if y.sum() == 0:
                self.log_models.append(None)
                self.rf_models.append(None)
                self.boosted_models.append(None)

            else:            
                log_model = LogisticRegression(penalty='l1')
                log_model.fit(X,y)
                self.log_models.append(log_model)

                rf_model = RandomForestClassifier()
                rf_model.fit(X,y)
                self.rf_models.append(rf_model)

                boosted_model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=1, loss='exponential',
                                        random_state=64)
                boosted_model.fit(X,y)
                self.boosted_models.append(boosted_model)
                

    # How to reconcile A predictions/ B predictions? Mask customers 

    def get_predictions(self,df_tests,df_trains):
        best_models = []
        for n in range(self.n_models):
            X_test = df_tests[n][self.columns].values
            self.labels.append(df_tests[n]["Target"].values.astype(int)) #y_test
            X_train = df_trains[n][self.columns].values
            y_train = df_trains[n]["Target"].values.astype(int)

            best_model_cv = []
            model_results = []

            yhat = self.log_models[n].predict(X_test)
            probas = self.log_models[n].predict_proba(X_test)
            score = self.log_models[n].score(X_test,self.labels)
            cv_score = np.array( cross_val_score(self.log_models[n],X_train,y_train,cv=5) ).mean()

            best_model_cv.append(cv_score)
            model_results.append( (yhat,probas,score,cv_score) )

            yhat = self.rf_models[n].predict(X_test)
            probas = self.rf_models[n].predict_proba(X_test)
            score = self.rf_models[n].score(X_test,self.labels)
            cv_score = np.array( cross_val_score(self.rf_models[n],X_train,y_train,cv=5) ).mean()

            best_model_cv.append(cv_score)
            model_results.append( (yhat,probas,score,cv_score) )
            
            yhat = self.boosted_models[n].predict(X_test)
            probas = self.boosted_models[n].predict_proba(X_test)
            score = self.boosted_models[n].score(X_test,self.labels)
            cv_score = np.array( cross_val_score(self.boosted_models[n],X_train,y_train,cv=5) ).mean()

            best_model_cv.append(cv_score)
            model_results.append( (yhat,probas,score,cv_score) )

            best_model = np.argmax(best_model_cv)
            best_models.append( model_results[best_model] )

        return best_models