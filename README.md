**Background**

There is significant need for a winery to determine when members of their wine club (a subscription-based model in which a customer is sent either 3, 6, or 12 bottles of wine 4-5 times per year) will churn. Having this predictive power in an industry that is notoriously rooted in tradition and reluctant to adopt new technologies has serious implications for both competitive advantage in the industry and overall production volume of wines featured on club shipments. If a winery could intervene with customers who are at risk of leaving the wine club, then we might avoid increased operating costs from unhealthy inventory of perishable goods as well as lost subscription revenue, which are both consequences of reduced club memberships. If we were to know right when a member was able to churn, and perhaps the reason why, we would be able to intervene and perhaps keep that customer.

**Data**

Using retail sales data from 2 wineries, I propose to create a model that could predict when people will churn from a wine club. A baseline, minimum viable product model would be simply predicting a club member will churn after the mean club membership length. Current attributes present in the raw data include:

```
Data columns (total 40 columns):
OrderNumber             24 non-null int64
OrderCompletedDate      24 non-null object
orderstatus             24 non-null object
ordertype               24 non-null object
OrderCompletedDate.1    24 non-null object
BillCustomerNumber      22 non-null float64
ShipStateCode           24 non-null object
BillBirthDate           22 non-null object
ShipZipCode             24 non-null int64
BillFirstName           24 non-null object
BillLastName            24 non-null object
BillCompany             17 non-null object
BillAddress             24 non-null object
BillAddress2            0 non-null float64
BillCity                24 non-null object
BillZipCode             24 non-null int64
BillPhone               22 non-null object
BillEmail               24 non-null object
isPickup                24 non-null bool
ShipBirthDate           22 non-null object
ShipFirstName           24 non-null object
ShipLastName            24 non-null object
ShipCompany             0 non-null float64
ShipAddress             24 non-null object
ShipAddress2            0 non-null float64
ShipCity                24 non-null object
ShipPhone               22 non-null object
ShipEmail               24 non-null object
SalesAssociate          24 non-null object
shipmentCount           0 non-null float64
ClubName                0 non-null float64
clubShipmentName        0 non-null float64
Quantity                24 non-null int64
ProductType             24 non-null object
OrderNotes              0 non-null float64
ProductTitle            24 non-null object
BrandKey                24 non-null object
ProductSKU              24 non-null object
originalPrice           24 non-null float64
Price                   24 non-null float64
```

**Feature Engineering**

A robust models will almost certainly require featurization to identify customer behavior that is a potential marker indicating a club member is more likely to churn. Paths for exploration might include:

```
skipped_shipment
```
Numerical value indicating how many shipments this customer has skipped. May change to categorical if not many shipments are skipped.

```
online_order_ltv
```

Numerical value indicating how much money (Lifetime Value) this customer has spent in supplemental orders online outside of tasting room and club shipments.

```
tasting_room_ltv
```

Numerical value indicating how much money (Lifetime Value) this customer has spent in supplemental orders online inside of tasting room and club shipments.


```
has_merch
```

Categorical value indicating if a customer has purchased merchandise (clothing, wine accessories) from us.

```
tasting_room_visits
```

Numerical value indicating number of visits to our tasting room


```
overall_ltv
```

Numerical sum of LTV (Lifetime Value) or total revenue gained from this customer.


```
intimacy_score
```

Numerical score indicating how relaxed the tasting room was at the time of club sign-up. Inverse of total amount of ```SamplesPoured``` value in ```BillFirstName```

```
bottle_count
```

Volume of wine purchased and consumed by club member.


**Challenges**

Potential challenges might include leakage as time-based data is central to this project. There may also exist a survivorship bias where are core data is comprised only of people who have continued to engage with the winery and have not churned.

**Defining the Target**

A club member has churned if...

**Model**

For a first model, I will predict a club member to churn after the mean membership length has elapsed. As this is a classification problem, I will initially use a logistic regression classifier along this one feature to start.

```
avg_membership_length
```

Classification feature with 0 indicating club member has not hit mean membership duration (and we classify is still a member) and 1 indicating club member has hit mean membership duration (and we classify as no longer a member).

**Model Evaluation**

I will evaluate the robustness of my model by plotting a ROC curve and evaluating the resulting AUC score.

Of key importance for evaluation will be our profit curve. For now, I will assume the following values:

| Profit Curve Outcomes | Actual Positive      | Actual Negative  |
| :-------------: |:-------------:| :----:|
| Predicted Positive  | Next shipment revenue - (Cost of intervention (time, discount)) | -(Cost of intervention (time, discount)) |
| Predicted Negative  | | |




