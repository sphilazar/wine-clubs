
'''
To Do:

Make ROCs for all models
Make plot of clusters


'''

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
from src.customerlifecycle import OrderAnalysis

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

oa = OrderAnalysis()
oa.get_orders()
oa.get_clubs()

clubs_train = pd.read_csv('../train_set.csv')
clubs_test = pd.read_csv('../test_set.csv')

prior_orders = oa.merge_tables()
prior_orders = prior_orders[['Customer Number','Website Log Price Total Before','POS Log Price Total Before']].reset_index()
prior_orders["Prior Orders"] = [(x + y) if not (x=="nan" or y=="nan" or x=="inf" or y=="inf")  else 0 for x,y in zip(prior_orders['Website Log Price Total Before'],prior_orders['POS Log Price Total Before']) ]
prior_orders = prior_orders[['Customer Number','Prior Orders']]

clubs_train = clubs_train.merge(prior_orders,how="left",left_on="Customer ID",right_on="Customer Number")
clubs_test = clubs_test.merge(prior_orders,how="left",left_on="Customer ID",right_on="Customer Number")

clubs_train = clubs_train[~clubs_train["Prior Orders"].isna()]
clubs_test = clubs_test[~clubs_test["Prior Orders"].isna()]

# print(prior_orders['Prior Orders'].unique())
# print(clubs_train.head())

'''
Logistic
'''
cols = ['Age','ASP','Quarter Case','Half Case',  'Full Case','isPickup','Last Order Amount','Average Transaction','Prior Orders']
cm = ChurnModel(cols)

cm.fit_a_model(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)
print(cm.model.coef_)
print(score)
print(cv_scores)

auc = cm_test.get_roc_curve(probas[:,1])
print("AUC Logistic: ",auc[0])
plt.show()

'''
Random Forest
'''
cm.fit_random_forest(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

print(score)
print(cv_scores)

auc = cm_test.get_roc_curve(probas[:,1])
print("AUC RF: ",auc[0])
plt.show()
'''
GradientBoostingClassifier
'''
# cols = ['Age','ASP','Quarter Case','isPickup','Last Order Amount','Average Transaction']
cols = ['Age','ASP','Quarter Case','Half Case', 'Full Case','isPickup','Last Order Amount','Average Transaction','Prior Orders']
dropcols = ['Club Tier','Cancel Reason','Customer ID','State', 'Zip','Club Status','Above Mean Club Length']
cols_gd = list(set(list(clubs_train.columns)) - set(dropcols))

clubs_train_gd = clubs_train.drop(dropcols,axis=1)
clubs_test_gd = clubs_test.drop(dropcols,axis=1)

cm = ChurnModel(cols)
cm.fit_gradient_boosted_forest(clubs_train_gd,n_estimators=50)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test_gd,clubs_train_gd)

print(score)
print(cv_scores)

auc = cm_test.get_roc_curve(probas[:,1])
print("AUC GB: ",auc[0])
plt.show()

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

# cols = ['Age','ASP','Quarter Case','isPickup','Last Order Amount','Average Transaction']
cols = ['Age','ASP','Quarter Case','Half Case',  'Full Case','isPickup','Last Order Amount','Average Transaction','Prior Orders']
cluster_cols = ["Customer ID","Target"]+cols

cluster_df = clubs_train[cluster_cols].reset_index()
cluster_df_test = clubs_test[cluster_cols].reset_index()

'''
Elbow
'''
# k_range = np.arange(2,8)
# sses = []
# plt.figure()
# for k in k_range:
#     print(k)
#     cluster_df = clubs_train[cluster_cols].reset_index()
#     cluster_df_test = clubs_test[cluster_cols].reset_index()
#     km = KMeans(cluster_df,cluster_cols,k=k)
#     km.fit(min_converge=1,max_iter=4)
#     sses.append(km.sse())
# print(sses)
# plt.scatter(k_range,np.array(sses))
# plt.show()

'''
Clustering
'''
km = KMeans(cluster_df,cluster_cols,k=6)
km.fit(min_converge=1,max_iter=4) # min_converge=1, max_iter=4
km.assign_test_clusters(cluster_df_test)

'''
Ensemble Modeling
'''
ecm = EnsembleChurnModel(cols,km.clusters,km.targets)
ecm.fit_models()
ecm.get_predictions(km.ensemble_Xs,km.ensemble_ys)
print(ecm.score()) #0.80


