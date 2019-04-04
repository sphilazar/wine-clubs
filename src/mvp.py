# mvp

import pandas as pd
import numpy as np
from src import format
import string
from src.model import ChurnModel
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src import model as m
from src.clusters import KMeans
from src.ensemble import EnsembleChurnModel

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import csv
import random
import pickle

'''
Clean up data, produce DataFrame
'''
clubs = format.clean_data()
format.get_test_train_set(clubs)

clubs_train = pd.read_csv('../data/train_set.csv')
clubs_test = pd.read_csv('../data/test_set.csv')

print(clubs_train.head())

cols = ['Age','ASP','Club Length','Quarter Case','isPickup','Time Since Last Order','Orders Total','Last Order Amount']
cm = ChurnModel(cols)

cm.fit_a_model(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

#score
#cv_scores
auc = cm_test.get_roc_curve(probas[:,1])
#auc[0]

'''
Random Forest
'''
cm.fit_random_forest(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

#score
#cv_scores

'''
GradientBoostingClassifier
'''
cols = ['Age','ASP','Club Length','Quarter Case','isPickup','Time Since Last Order','Orders Total','Last Order Amount']
dropcols = ['Club Tier','Cancel Reason','Customer ID','State', 'Zip','Club Status','Above Mean Club Length']
cols_gd = list(set(list(clubs_train.columns)) - set(dropcols))

clubs_train_gd = clubs_train.drop(dropcols,axis=1)
clubs_test_gd = clubs_test.drop(dropcols,axis=1)

cm = ChurnModel(cols)
cm.fit_gradient_boosted_forest(clubs_train_gd,n_estimators=100)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test_gd,clubs_train_gd)

#score
#cv_scores

'''
Partial Dependence Plots
'''
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

c = np.array(clubs_train_gd.columns).reshape(-1,1).T
features = []
for col in cols:
    features.append(np.argwhere(c==col)[0])
features = list(np.array(features)[:,1])
c = list(clubs_train_gd[cols].columns)
features = np.arange(0,len(c))

fig, axs = plot_partial_dependence(cm.model, clubs_train_gd[cols].values, features,
                                   feature_names=clubs_train_gd[cols].columns,
                                   n_jobs=3, grid_resolution=50,figsize=(14,9))

cols = ['Age','ASP','Club Length','Quarter Case','isPickup','Time Since Last Order','Orders Total','Last Order Amount']
cluster_cols = ["Customer ID","Target"]+cols

cluster_df = clubs_train[cluster_cols].reset_index()
cluster_df_test = clubs_test[cluster_cols].reset_index()

'''
Clustering
'''
km = KMeans(cluster_df,cols)
km.fit(min_converge=1,max_iter=4) # min_converge=1, max_iter=4
km.assign_test_clusters(cluster_df_test)

'''
Ensemble Modeling
'''
ecm = EnsembleChurnModel(cols,km.clusters,km.targets)
ecm.fit_models()
print(ecm.score()) #0.90

'''
Customer lifetime regression
'''
