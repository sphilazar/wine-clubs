
'''
This script produces all graphs, calculations, and model scores as needed.
'''

'''
Below are all Python packages used in communicating and visualizing the results of this analysis.
'''

import pandas as pd
import numpy as np
import string
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import csv
import random
import pickle

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

'''
Below are all the scripts utilized by this script. Descriptions below:

model.py

This script contains a ChurnModel class that can take on the form of any ML model and contains a function for storing the model in a pickle.

customerlifecycle.py

This script contains a OrderAnalysis class that stores all necessary dataframes for analysis and contains all methods necessary for manipulating the dataframes. 

'''

from src.model import ChurnModel
from src import model as m
from src.customerlifecycle import OrderAnalysis

'''
Clean up data, produce DataFrames
'''
oa = OrderAnalysis()
oa.clean_data()
oa.get_order_history()
oa.merge_tables()

'''
Load train, test datasets
'''
oa.get_test_train_set(oa.clubs)

clubs_train = pd.read_csv('../train_set.csv')
clubs_test = pd.read_csv('../test_set.csv')

'''
Note: clubs_train and clubs_test each have the following columns (choose among these columns as you wish):

['Customer Number',  'Bill Zip',  'isPickup',  'Club Length',  'Shipments',         'Age',  'Quarter Case',  'Half Case',  'Full Case',  'Quantity',  'Log Spending Per Year',  'POS Log Price Total Before'  ,'Club Log Price Total Before',  'Website Log Price Total Before',  'Number Of Transactions_y'  ,'AverageDaysSince',  'TotalWineBefore',  'OrdersBeforeJoin']
'''

'''
Logistic Regression
'''
cols = ['isPickup', 'Age',  'Quarter Case',  'Half Case',  'Full Case',  'POS Log Price Total Before'  ,'Club Log Price Total Before',  'Website Log Price Total Before' ,  'TotalWineBefore',  'OrdersBeforeJoin','ASP','Average Transaction','AverageDaysSince','Log Spending Per Year'] 

cm = ChurnModel(cols)
cm.fit_a_model(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

auc = cm_test.get_roc_curve("Logistic Regression",probas[:,1])

'''
Random Forest
'''
cm.fit_random_forest(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

auc = cm_test.get_roc_curve("Random Forest",probas[:,1])

'''
GradientBoostingClassifier
'''

cm = ChurnModel(cols)
cm.fit_gradient_boosted_forest(clubs_train,n_estimators=50)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

auc = cm_test.get_roc_curve("Gradient Boosting",probas[:,1])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curves")
plt.legend(['Logistic','Random Forest','Gradient Boosting'])
plt.grid(True)

'''
Profit Curve
'''

def standard_confusion_matrix(y_true, y_predict):
    """
    INPUT: NUMPY ARRAY (n_observations,) (bool), NUMPY ARRAY (n_observations,) (bool)
    OUTPUT: NUMPY ARRAY (2,2) in the form:
        [[tp, fp], 
         [fn, tn]]
    
    """
    y_true = np.array(y_true).astype(bool)
    y_predict = np.array(y_predict).astype(bool)
    
    tp = (y_true & y_predict).sum()
    fp = (~y_true & y_predict).sum()
    fn = (y_true & ~y_predict).sum()
    tn = (~y_true & ~y_predict).sum()

    con_mat = np.array([[tp, fp], [fn, tn]])
    
    recall = (tp / (tp + fn))
    precision = (tp / (tp + fp))
    F1 = (2 * ((recall * precision) / (recall + precision)))


    return con_mat,F1,recall,precision

def profit_curve(cost_benefit, predicted_probs, labels):
    
    labels = np.array(labels).astype(bool)
    predicted_probs = np.array(predicted_probs)
    cost_benefit = np.array(cost_benefit)
    cost_list = []
    
    idx_sort = np.argsort(predicted_probs)
    
    labels = np.append(0,labels[idx_sort])         
    predicted_probs = np.append(0,predicted_probs[idx_sort]) 
    
    F1s = []
    recalls = []
    precisions = []
    
    for threshold in predicted_probs:
        y_predict = (predicted_probs > threshold)
        con_mat,F1,recall,precision = standard_confusion_matrix(labels, y_predict)
        net_cb = (con_mat * cost_benefit).sum()
        cost_list.append(net_cb)
        F1s.append(F1)
        recalls.append(recall)
        precisions.append(precision)

    print("Max recall: ",np.array(recalls).max(),"At threshold: ",predicted_probs[np.argmax(recalls)])
        
    return np.array(cost_list), predicted_probs

retention_profit = 525
discount_cost = (-150)

cost_benefit = np.array([[(retention_profit + discount_cost),discount_cost], [0, 0]])

profits,thresholds = profit_curve(cost_benefit,probas[:,1],cm_test.labels)

print("Max Profit: ",profits.max(),"Corresponding threshold: ",thresholds[np.argmax(profits)])

optimal_threshold = thresholds[np.argmax(profits)]

plt.figure()
plt.plot(thresholds,profits,c='g')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Profit $")
plt.title("Profit Curve for Retaining Wine Club Members")
plt.axvline(x=optimal_threshold,linestyle=":",color="b")

def get_optimal_profit_curve_performance(threshold,predicted_probs,labels):
    y_predict = (predicted_probs > threshold)
    con_mat,F1,recall,precision = standard_confusion_matrix(labels, y_predict) #[[tp, fp], [fn, tn]]
    return (con_mat,F1,recall,precision)

stats = get_optimal_profit_curve_performance(optimal_threshold,probas[:,1],cm_test.labels)

'''
Partial Dependence Plots
'''
c = np.array(clubs_train.columns).reshape(-1,1).T
features = []
for col in cols:
    features.append(np.argwhere(c==col)[0])
features = list(np.array(features)[:,1])
c = list(clubs_train[cols].columns)
features = np.array([8,10,11]) # Interested in indices corresponding to columns ['TotalWineBefore', 'ASP','Average Transaction']

fig, axs = plot_partial_dependence(cm.model, clubs_train[cols].values, features,
                                   feature_names=clubs_train[cols].columns,n_cols=3,
                                   n_jobs=3, grid_resolution=50,figsize=(9,24))
plt.show()
