# Predicting Wine Club Subscriptions

## Phil Salazar

### Objective

A winery wants to determine which customers to recruit for their wine club.

### What is a wine club?

A wine club is a subscription-based model in which a subscriber receives 3, 6, or 12 bottles of wine 4-5 times per year.

### Motivation

A winery consistently meets quarterly new wine club subscription goals but misses quarterly revenue goals. A potential solution to this problem would be to recruit customers who are likely to remain subscribers for a longer period of time. I intend to predict customers who will stay in the winery's wine club subsciption for 2 years or more, with 2 years being a standard deviation above the mean lifespan of a subscriber. 

### Data

My project relies primarily on point of sale (POS) data from the winery's database management system, WineDirect. Data attributes include date, quantity, pricing, and categorization of purchases, including whether an order was completed online, as a club subscription shipment, or in the winery. 

Additionally, I heavily relied on a database tracking club subscriber attributes and history.

### Feature Engineering

Below is a subset of features that were instrumental in prediction accuracy:

```
wine_before
```

Integer value: the number of bottles of wine a subscriber purchased prior to signing up

```
online_total_before
```

Float value: the summed total of log(`price`) * quantity for each transaction with `order_type` as `Online`.

```
winery_total_before
```

Float value: the summed total of log(`price`) * quantity for each transaction with `order_type` as `POS` (transactions occurring within winery).

```
asp
```

Float value: the mean total of log(`price`) * quantity across all transactions

```
average_transaction
```

Float value: the mean of all transaction totals.

### Defining the Target

A wine club subscriber belongs to the positive class so long as their subscription spanned a minimum of 2 years, regardless of if the subscription is currently active or not. 

### Leakage

So as to prevent leakage, all time-based variables are immediately eliminated, as are subscription-specific purchases in the POS data (designated by `ClubOrder` or `ClubShipment` in `order_type` field).

### Evaluation

I will evaluate models by the following classification metrics:

**F1, ROC - AUC score, precision, recall**

In particular, I will

### Model Evaluation

I will evaluate the robustness of my model by plotting a Receiver Operating Characteristic (ROC) curve and evaluating the resulting Area Under Curve (AUC) score.

Given this model will ultimately fulfill a business application, of key importance for evaluation will be a profit curve. Note that the sensitive economics of the subscription model do not allow me to disclose the detials of the cost-benefit matrix below.

#### Cost-Benefit Matrix

| Financial Outcome for each Model Prediction | Actual Subscription 2+ Years      | Actual Subscription <2 Years  |
| :-------------: |:-------------:| :----:|
| Predicted Subscription 2+ Years   | Incremental shipment revenue* + Cost of intervention^ | Cost of intervention^ |
| Predicted Subscription <2 Years  | Incremental shipment revenue* | 0 |

Note that profit curves rely on the use of soft classifiers, or predicted probabilities that a given subscriber will stay for 2 years. Based on the maximum of the profit curve, we pick the optimal probability threshold for properly classifying each subscriber.

###### * Note this cannot be disclosed under NDA
###### ^ Negative value, note this cannot be disclosed under NDA

### Models 

#### I. Baseline Model

I define a baseline model as one that predicts every subscriber to stay for 2 years or more. Since my training data has a class balance of 57% of customers who stay for 2 years or more, I would expect the precision of this baseline model to come out to 0.57.

I will focus on precision as a key metric for the purpose of evaluating competing models, as precision is the measure of the proportion of the model's **true positives relative to true positives and false positives.** False positives in particualr are to be avoided, as we expect a subscriber to stay for 2 years when in reality they leave early.

#### II. Logistic Regression

A basic model that produces soft classifiers is logistic regression. I fit a logistic regression model using a variety of features:

(TODO)

This logistic regression model leveraged LASSO L1-regularuization to aid in feature selection and interpretability.

The largest coefficients (and therefore most important) included:

(TODO)

The scores of this model were as follows:


#### III. Random Forest Classifier

I then fit a random forest classification model using a variety of features:

The scores of this model were as follows:

| Soft Classifier Model | ROC-AUC Score |
|:-------------:| :----:|
| Logistic Regression | 0.70 |
| Random Forest | 0.79 |
| Gradient Boosting | **0.87** |

Although I will consider precision to be the ultimate indicator of model prediction performance, the AUC score is an important metric for evaluating the model versatility. As business conditions could change, 

#### IV. Gradient Boosting Classifier

I then fit a random forest classification model using a variety of features:

Notably, this model required a low learning rate to perform well.

The scores of this model were as follows:

### Model Results

Gradient boosting classifier achieved higher accuracy metrics than both the random forest classifier or the logistic regression models. Most notably, gradient boosting appeared to be the most versatile and reliable model for producing high shares of true positives relative to false positives, as is indicated by the below ROC Curve. 






Versatility is an important consideration in a business application, as business conditions may change.


### Profit Curve

The profit curve, leveraging the corresponding confusion matrix at every probability threshold and applying our cost-benefit matrix above, shows the probability threshold for classification producing maximum profit.








In this case, we see that a probability threshold of 0.51 yields the global maximum for profit.

#### Confusion Matrix for Gradient Boosting Classifier at 0.51 classification threshold for test data

| Test Data | Actual Subscription 2+ Years      | Actual Subscription <2 Years  |
| :-------------: |:-------------:| :----:|
| Predicted Subscription 2+ Years   | 46 | 14 |
| Predicted Subscription <2 Years  | 5 | 29 |

With these outcomes in mind, **our model produces a precision score of 0.77, which is 20 points higher than the 0.57 precision score of our baseline model.**

### Conclusion

**Good predictors of club length:** average sale price (ASP) (affinity for premium wines), number of bottles purchased prior to signing up (higher implicit value of being in wine club), average transaction (spending power).

### Next Steps

Validate results on complete winery dataset, generalize model for other wineries in portfolio. Predict continuous targets: club subscription lifespan, club subscription lifetime value ($).



