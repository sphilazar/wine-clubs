{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Background**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is significant need for a winery to determine when members of their wine club (a subscription-based model in which a customer is sent either 3, 6, or 12 bottles of wine 4-5 times per year) will churn. Having this predictive power in an industry that is notoriously rooted in tradition and reluctant to adopt new technologies has serious implications for both competitive advantage in the industry and overall production volume of wines featured on club shipments. If a winery could intervene with customers who are at risk of leaving the wine club, then we might avoid increased operating costs from unhealthy inventory of perishable goods as well as lost subscription revenue, which are both consequences of reduced club memberships. If we were to know right when a member was able to churn, and perhaps the reason why, we would be able to intervene and perhaps keep that customer."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Data**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using retail sales data from 2 wineries, I propose to create a model that could predict when people will churn from a wine club. A baseline, minimum viable product model would be simply predicting a club member will churn after the mean club membership length. Current attributes present in the raw data include:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "Data columns (total 40 columns):\n",
    "OrderNumber             24 non-null int64\n",
    "OrderCompletedDate      24 non-null object\n",
    "orderstatus             24 non-null object\n",
    "ordertype               24 non-null object\n",
    "OrderCompletedDate.1    24 non-null object\n",
    "BillCustomerNumber      22 non-null float64\n",
    "ShipStateCode           24 non-null object\n",
    "BillBirthDate           22 non-null object\n",
    "ShipZipCode             24 non-null int64\n",
    "BillFirstName           24 non-null object\n",
    "BillLastName            24 non-null object\n",
    "BillCompany             17 non-null object\n",
    "BillAddress             24 non-null object\n",
    "BillAddress2            0 non-null float64\n",
    "BillCity                24 non-null object\n",
    "BillZipCode             24 non-null int64\n",
    "BillPhone               22 non-null object\n",
    "BillEmail               24 non-null object\n",
    "isPickup                24 non-null bool\n",
    "ShipBirthDate           22 non-null object\n",
    "ShipFirstName           24 non-null object\n",
    "ShipLastName            24 non-null object\n",
    "ShipCompany             0 non-null float64\n",
    "ShipAddress             24 non-null object\n",
    "ShipAddress2            0 non-null float64\n",
    "ShipCity                24 non-null object\n",
    "ShipPhone               22 non-null object\n",
    "ShipEmail               24 non-null object\n",
    "SalesAssociate          24 non-null object\n",
    "shipmentCount           0 non-null float64\n",
    "ClubName                0 non-null float64\n",
    "clubShipmentName        0 non-null float64\n",
    "Quantity                24 non-null int64\n",
    "ProductType             24 non-null object\n",
    "OrderNotes              0 non-null float64\n",
    "ProductTitle            24 non-null object\n",
    "BrandKey                24 non-null object\n",
    "ProductSKU              24 non-null object\n",
    "originalPrice           24 non-null float64\n",
    "Price                   24 non-null float64\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Feature Engineering**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A robust models will almost certainly require featurization to identify customer behavior that is a potential marker indicating a club member is more likely to churn. Paths for exploration might include:\n",
    "\n",
    "```\n",
    "skipped_shipment\n",
    "```\n",
    "Numerical value indicating how many shipments this customer has skipped. May change to categorical if not many shipments are skipped.\n",
    "\n",
    "```\n",
    "online_order_ltv\n",
    "```\n",
    "\n",
    "Numerical value indicating how much money (Lifetime Value) this customer has spent in supplemental orders online outside of tasting room and club shipments.\n",
    "\n",
    "```\n",
    "tasting_room_ltv\n",
    "```\n",
    "\n",
    "Numerical value indicating how much money (Lifetime Value) this customer has spent in supplemental orders online inside of tasting room and club shipments.\n",
    "\n",
    "\n",
    "```\n",
    "has_merch\n",
    "```\n",
    "\n",
    "Categorical value indicating if a customer has purchased merchandise (clothing, wine accessories) from us.\n",
    "\n",
    "```\n",
    "tasting_room_visits\n",
    "```\n",
    "\n",
    "Numerical value indicating number of visits to our tasting room\n",
    "\n",
    "\n",
    "```\n",
    "overall_ltv\n",
    "```\n",
    "\n",
    "Numerical sum of LTV (Lifetime Value) or total revenue gained from this customer.\n",
    "\n",
    "\n",
    "```\n",
    "intimacy_score\n",
    "```\n",
    "\n",
    "Numerical score indicating how relaxed the tasting room was at the time of club sign-up. Inverse of total amount of ```SamplesPoured``` value in ```BillFirstName```\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Challenges**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Potential challenges might include leakage as time-based data is central to this project. There may also exist a survivorship bias where are core data is comprised only of people who have continued to engage with the winery and have not churned."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
