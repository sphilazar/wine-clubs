
'''
To Do:

Make ROCs for all models - done
Make plot of clusters - done
Average Transaction --> Order Total --> LTV - done
ASP --> Order Total --> LTV - done
Make sure generlizable for rest of POS data - done
Make sure generalizable for other wineries - done
kNN? -  maybe
Do a grid search - nah

Profit curve - done
Calculate Recall, Precision scores for model
Check F1 scores
Plots plots plots plots - cluster
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

from mpl_toolkits.mplot3d import Axes3D 

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import csv
import random
import pickle


'''
Clean up data, produce DataFrame
'''
# clubs = format.clean_data()

# '''
# Current Columns

# ['Club', 'Club Status',  'Customer Number', 'Bill Birth Date',       'Bill City', 'Bill State Code', 'Bill Zip', 'Ship Birth Date','Ship City', 'Ship State Code', 'Ship Zip', 'Pickup Location', 'Signup Date', 'Cancel Date','Cancel Reason','Shipments', 'Last Processed Date','Lifetime Value','Last Order Date','City', 'State', 'Zip Code',  'Transactions',  'Last Order Amount', 'Date Added', 'Last Modified Date',      'LTV','Age','isPickup','Club Length','Last Order Date','Time Since Last Order','Quantity','Total', 'ASP',  'Target',  'Quarter Case',  'Half Case',  'Full Case', 'Average Transaction']
# '''

# filter_cols = ['Club', 'Club Status',  'Customer Number', 'Zip Code',  'Transactions',  'Last Order Amount', 'Date Added', 'Last Modified Date',      'LTV','Age','isPickup','Club Length','Time Since Last Order','Quantity','Total', 'ASP',  'Target',  'Quarter Case',  'Half Case',  'Full Case', 'Average Transaction']
# clubs = clubs[filter_cols]

# print(clubs.sample(5))
# print(clubs.shape)


'''
Basic Stats:

Mean Club Length:  1.5738357843137254 
std club length:  0.8831259196975829
Cancelled vs. Not Cancelled: 0.642014
Class balance: 0.500393
'''

oa = OrderAnalysis()
oa.clean_data()
oa.get_order_history()
oa.merge_tables()
print("Balance (Positive Class): ",(oa.clubs["Target"].astype(int).sum() / len(oa.clubs)))
# clubs = oa.bootstrap(oa.clubs)
oa.get_test_train_set(oa.clubs)

clubs_train = pd.read_csv('../train_set.csv')
clubs_test = pd.read_csv('../test_set.csv')

# prior_orders = oa.merge_tables()
# prior_orders = prior_orders[['Customer Number','Website Log Price Total Before','POS Log Price Total Before']].reset_index()
# prior_orders["Prior Orders"] = [(x + y) if not (x=="nan" or y=="nan" or x=="inf" or y=="inf")  else 0 for x,y in zip(prior_orders['Website Log Price Total Before'],prior_orders['POS Log Price Total Before']) ]
# prior_orders = prior_orders[['Customer Number','Prior Orders']]


# clubs_train = clubs_train[~clubs_train["Prior Orders"].isna()]
# clubs_test = clubs_test[~clubs_test["Prior Orders"].isna()]

print(clubs_test.sample(5))

# # print(prior_orders['Prior Orders'].unique())
# # print(clubs_train.head())

'''
Current cols:

cols = ['Customer Number',  'Bill Zip',  'isPickup',  'Club Length',  'Shipments',         'Age',  'Quarter Case',  'Half Case',  'Full Case',  'Quantity',  'Log Spending Per Year',  'POS Log Price Total Before'  ,'Club Log Price Total Before',  'Website Log Price Total Before',  'Number Of Transactions_y'  ,'AverageDaysSince',  'TotalWineBefore',  'OrdersBeforeJoin']

Unsure: 'Log Spending Per Year','AverageDaysSince'
'''

'''
Logistic
'''
cols = ['isPickup', 'Age',  'Quarter Case',  'Half Case',  'Full Case',  'POS Log Price Total Before'  ,'Club Log Price Total Before',  'Website Log Price Total Before' ,  'TotalWineBefore',  'OrdersBeforeJoin','ASP','Average Transaction','AverageDaysSince','Log Spending Per Year'] 

cm = ChurnModel(cols)

cm.fit_a_model(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)
print(cm.model.coef_)
print("Logistic: ",score)
print(cv_scores)
print(clubs_test.columns)

auc = cm_test.get_roc_curve("Logistic Regression",probas[:,1])
print("AUC Logistic: ",auc[0])
# plt.show()

'''
Random Forest
'''
cm.fit_random_forest(clubs_train)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

print(cm.columns)
print("RF: ",score)
print(cv_scores)

auc = cm_test.get_roc_curve("Random Forest",probas[:,1])
print("AUC RF: ",auc[0])
# plt.show()

'''
GradientBoostingClassifier
'''
# # cols = ['Age',,'Quarter Case','isPickup','Last Order Amount','Average Transaction']
# cols = ['Age','Quarter Case','Half Case', 'Full Case','isPickup','Last Order Amount','Average Transaction','Prior Orders']
# dropcols = ['Club Tier','Cancel Reason','Customer ID','State', 'Zip','Club Status','Above Mean Club Length']
# cols_gd = list(set(list(clubs_train.columns)) - set(dropcols))

# clubs_train_gd = clubs_train.drop(dropcols,axis=1)
# clubs_test_gd = clubs_test.drop(dropcols,axis=1)

cm = ChurnModel(cols)
cm.fit_gradient_boosted_forest(clubs_train,n_estimators=50)
m.get_pickle(cm)
cm_test = pickle.load(open('model.p','rb'))

yhat,probas,score,cv_scores = cm_test.get_predictions(clubs_test,clubs_train)

print(cm.columns)
print("GB: ",score)
print(cv_scores)

auc = cm_test.get_roc_curve("Gradient Boosting",probas[:,1])
print("AUC GB: ",auc[0])
# plt.show()

# plt.plot(auc[3],auc[3],c="b:")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curves")
plt.legend(['Logistic','Random Forest','Gradient Boosting'])
plt.grid(True)
plt.show()

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

# Max Profit:  161.17021276595744 Corresponding threshold:  0.5105838099377613

def profit_curve(cost_benefit, predicted_probs, labels):
    
    labels = np.array(labels).astype(bool)
    predicted_probs = np.array(predicted_probs)
    cost_benefit = np.array(cost_benefit)
    cost_list = []
    
    idx_sort = np.argsort(predicted_probs)
    
    labels = np.append(0,labels[idx_sort])          # sort by prob and append 1
    predicted_probs = np.append(0,predicted_probs[idx_sort]) #sort by prob and append 1
    
    F1s = []
    recalls = []
    precisions = []
    
    for threshold in predicted_probs:
        y_predict = (predicted_probs > threshold)
        con_mat,F1,recall,precision = standard_confusion_matrix(labels, y_predict) #[[tp, fp], [fn, tn]]
        # net_cb = (con_mat * cost_benefit).sum() / (con_mat.sum() - 1)
        net_cb = (con_mat * cost_benefit).sum()
        cost_list.append(net_cb)
        F1s.append(F1)
        recalls.append(recall)
        precisions.append(precision)

    print("Max recall: ",np.array(recalls).max(),"At threshold: ",predicted_probs[np.argmax(recalls)])
        
    return np.array(cost_list), predicted_probs

retention_profit = 525
discount_cost = (-150) #free shipment (rtake into account spread of memberships - this assumes 4 bottles going out)

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
plt.show()

def get_optimal_profit_curve_performance(threshold,predicted_probs,labels):
    y_predict = (predicted_probs > threshold)
    con_mat,F1,recall,precision = standard_confusion_matrix(labels, y_predict) #[[tp, fp], [fn, tn]]
    return (con_mat,F1,recall,precision)

stats = get_optimal_profit_curve_performance(optimal_threshold,probas[:,1],cm_test.labels)

print("Conmat: ",stats[0],"F1: ",stats[1],"Recall: ",stats[2],"Precision: ",stats[3])

'''
Partial Dependence Plots
'''
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

c = np.array(clubs_train.columns).reshape(-1,1).T
features = []
for col in cols:
    features.append(np.argwhere(c==col)[0])
features = list(np.array(features)[:,1])
c = list(clubs_train[cols].columns)
features = np.array([8,10,11]) #Need Orders before age, TOtalwineBefore, ASP, Average transaction

fig, axs = plot_partial_dependence(cm.model, clubs_train[cols].values, features,
                                   feature_names=clubs_train[cols].columns,n_cols=3,
                                   n_jobs=3, grid_resolution=50,figsize=(9,24))
plt.show()
# cols = ['Age',,'Quarter Case','isPickup','Last Order Amount','Average Transaction']
# cols = ['Age','Quarter Case','Half Case','Full Case','isPickup','Last Order Amount','Average Transaction','Prior Orders']

def get_feature_importances(model_class):
    feature_importance = model_class.model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, model_class.columns[sorted_idx])
    plt.xlabel('Relative Ismportance')
    plt.title('Variable Importance')
    plt.show()

# get_feature_importances(cm)

# '''
# Elbow
# '''
# # k_range = np.arange(2,8)
# # sses = []
# # plt.figure()
# # for k in k_range:
# #     print(k)
# #     cluster_df = clubs_train[cluster_cols].reset_index()
# #     cluster_df_test = clubs_test[cluster_cols].reset_index()
# #     km = KMeans(cluster_df,cluster_cols,k=k)
# #     km.fit(min_converge=1,max_iter=4)
# #     sses.append(km.sse())
# # print(sses)
# # plt.scatter(k_range,np.array(sses))
# # plt.show()

# '''
# Clustering
# '''
# plot_cols = ['ASP'  ,'Average Transaction',  'Log Spending Per Year'] #AverageDaysSince
# cluster_cols = ["Customer Number","Target"]+plot_cols

# cluster_df = clubs_train[cluster_cols].reset_index()
# cluster_df_test = clubs_test[cluster_cols].reset_index()

# km = KMeans(cluster_df,cluster_cols,k=3)
# km.fit(min_converge=0.00000001,max_iter=100) # min_converge=1, max_iter=4 fucked
# km.assign_test_clusters(cluster_df_test)

# km.plot_clusters(plot_cols)

# plt.show()

# '''
# Ensemble Modeling
# '''
# ecm = EnsembleChurnModel(plot_cols,km.clusters,km.targets)
# ecm.fit_models()
# ecm.get_predictions(km.ensemble_Xs,km.ensemble_ys)
# print("Ensemble score: ",ecm.score()) #0.80


