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
from src.customerlifecycle import OrderAnalysis
from src.regression import LinReg, KNN

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import csv
import random
import pickle

'''
Customer Lifecycle
'''
oa = OrderAnalysis()
data = oa.get_orders()
clubs = oa.get_clubs()
final_table = oa.merge_tables()
# print(final_table.sample(20))

final_table['Log ASP'] = [x/y if y > 0 else 0 for x,y in zip(final_table['Log Price Total'],final_table['Number Of Transactions'])]

oa.get_test_train_set(final_table)

cols = ['Customer Number', 'Bill Zip',  'isPickup',  'clubLength',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case','Quantity','Log Spending Per Year','Log ASP'  ,'Log Price Total',  'OrderCompletedDate',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after',  'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before', 'Number Of Transactions']

# Note add back Customer Number when models improve

dropcols = ['Customer Number','OrderCompletedDate','Shipments','Quantity','Log Spending Per Year','Log Price Total','POS Log Price Total', 'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after', 'Club Log Price Total Before','Number Of Transactions']

# Removed target from each

clubs_train = pd.read_csv('../reg_train_set.csv')
clubs_test = pd.read_csv('../reg_test_set.csv')

cols = list(set(list(cols)) - set(dropcols))

clubs_train = clubs_train[cols]
clubs_test = clubs_test[cols]

print("Hey",clubs_train.columns)
print("Hey",clubs_test.columns)

'''
Linear Regression
'''

lr = LinReg()
lr.fit_model(clubs_train)
predictions = lr.get_predictions(clubs_test)
score = lr.get_score(clubs_test)

print("Linear Regression:"+str(score))

'''
kNN
'''

# print(clubs_train.columns)


cols = ['Customer Number', 'Bill Zip',  'isPickup',  'clubLength',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case','Quantity','Log Spending Per Year','Log ASP'  ,'Log Price Total',  'OrderCompletedDate',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after',  'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before', 'Number Of Transactions']

# Note add back Customer Number when models improve

dropcols = ['Customer Number','OrderCompletedDate','Shipments','Quantity','Log Spending Per Year','Log Price Total','POS Log Price Total', 'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after', 'Club Log Price Total Before','Number Of Transactions']

clubs_train = pd.read_csv('../reg_train_set.csv')
clubs_test = pd.read_csv('../reg_test_set.csv')

cols = list(set(list(cols)) - set(dropcols))

clubs_train = clubs_train[cols]
clubs_test = clubs_test[cols]

knn = KNN()
knn.fit_model(clubs_train)
predictions = knn.get_predictions(clubs_test)
score = knn.get_score(clubs_test)

print("KNN score:"+str(score))

'''
Random Forest
'''

cols = ['Customer Number', 'Bill Zip',  'isPickup',  'clubLength',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case','Quantity','Log Spending Per Year','Log ASP'  ,'Log Price Total',  'OrderCompletedDate',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after',  'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before', 'Number Of Transactions']

# Note add back Customer Number when models improve

dropcols = ['Customer Number','OrderCompletedDate','Shipments','Quantity','Log Spending Per Year','Log Price Total','POS Log Price Total', 'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after', 'Club Log Price Total Before','Number Of Transactions']

clubs_train = pd.read_csv('../reg_train_set.csv')
clubs_test = pd.read_csv('../reg_test_set.csv')

cols = list(set(list(cols)) - set(dropcols))

clubs_train = clubs_train[cols]
clubs_test = clubs_test[cols]

# clubs_train = pd.read_csv('../reg_train_set.csv')
# clubs_test = pd.read_csv('../reg_test_set.csv')
print(clubs_train.columns)

cm = ChurnModel(cols)
cm.fit_random_forest(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,predictions,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

print("Random Forest:"+str(score))
print(cv_scores)

'''
GradientBoostingRegressor
'''

cols = ['Customer Number', 'Bill Zip',  'isPickup',  'clubLength',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case','Quantity','Log Spending Per Year','Log ASP'  ,'Log Price Total',  'OrderCompletedDate',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after',  'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before', 'Number Of Transactions']

# Note add back Customer Number when models improve

dropcols = ['Customer Number','OrderCompletedDate','Shipments','Quantity','Log Spending Per Year','Log Price Total','POS Log Price Total', 'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after', 'Club Log Price Total Before','Number Of Transactions']

clubs_train = pd.read_csv('../reg_train_set.csv')
clubs_test = pd.read_csv('../reg_test_set.csv')

cols = list(set(list(cols)) - set(dropcols))

clubs_train = clubs_train[cols]
clubs_test = clubs_test[cols]

# clubs_train = pd.read_csv('../reg_train_set.csv')
# clubs_test = pd.read_csv('../reg_test_set.csv')

cm = ChurnModel(cols)
cm.fit_gradient_boosted_forest(clubs_train,n_estimators=5000)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,predictions,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

print("Gradient Boosted:"+str(score))
print("Gradient Boosted:"+str(cv_scores))

'''
Partial Dependence Plots
'''
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

pdp_cols = ['Log ASP', 'Full Case', 'Age', 'Website Log Price Total Before', 'POS Log Price Total Before', 'Bill Zip', 'Quarter Case', 'isPickup', 'Half Case']
c = np.array(clubs_train.columns).reshape(-1,1).T
features = []
for col in pdp_cols:
    features.append(np.argwhere(c==col)[0])
features = list(np.array(features)[:,1])
c = list(clubs_train[cols].columns)
features = np.arange(0,len(c))

plt.figure()
fig, axs = plot_partial_dependence(cm.model, clubs_train[cols].values, features,
                                   feature_names=clubs_train[cols].columns,
                                   n_jobs=3, grid_resolution=50,figsize=(14,9))
# how do I get this to show?




'''
Clustering
'''
clubs_train = pd.read_csv('../reg_train_set.csv')
clubs_test = pd.read_csv('../reg_test_set.csv')
# print(clubs_train.columns)

'''
Elbow Method
Result: 6 clusters optimal

k_range = np.arange(2,12)
sses = []
for k in k_range:
    km = KMeans(clubs_train,cols,k=k)
    km.fit(min_converge=0.005,max_iter=10)
    sses.append(km.sse())

plt.figure()
plt.scatter(k_range,np.array(sses))
plt.show()
'''

km = KMeans(clubs_train,cols)
km.fit(min_converge=1,max_iter=4) # min_converge=1, max_iter=4
km.assign_test_clusters(clubs_test)

'''
Ensemble Modeling
'''
dropcols = ["clubLength"]
ecmcols = list(set(list(cols)) - set(dropcols))

ecm = EnsembleChurnModel(ecmcols,km.clusters,km.targets)
ecm.fit_models()

best_models = ecm.get_predictions(km.ensemble_Xs,km.ensemble_ys)
score = ecm.score()
print("Ensemble:"+str(score)) #Current: 0.90

print(ecm.cv_scores)
print("Columns: ",ecm.columns,"Number: ",len(ecm.columns))

