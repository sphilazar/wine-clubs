import pandas as pd
import numpy as np
import string

def clean_data():
    path = '../club_members.csv'
    clubs = pd.read_csv(path)

    path = '../customer_list.csv'
    customers = pd.read_csv(path)

    # Pick important features in Clubs
    nans = ["Club Status","Bill Birth Date","Ship Birth Date","Ship City","Ship State Code","Ship Zip","Pickup Location","Cancel Date","Cancel Reason","Last Processed Date","Last Order Date"]
    delete = ["Bill City","Bill State Code","Bill Zip"]
    keep = ["Club","Customer Number","Signup Date","Shipments (note that these don't necessarily mean orders)","Lifetime Value"]

    clubs = clubs[nans + keep]

    # Create features
    clubs["ltv"] = [float("".join((ltv[1:].split(","))) ) for ltv in clubs["Lifetime Value"]]
    clubs.loc[clubs["Club Status"].isna() & clubs["Cancel Date"].isna(),"Club Status"] = "Active"
    clubs.loc[clubs["Club Status"].isna() & ~clubs["Cancel Reason"].isna(),"Club Status"] = "Cancelled"
    clubs.loc[ (clubs["Bill Birth Date"].isna() & ~clubs["Ship Birth Date"].isna()) , "Bill Birth Date"] = clubs["Ship Birth Date"]
    clubs.loc[ (~clubs["Bill Birth Date"].isna() & clubs["Ship Birth Date"].isna()) , "Ship Birth Date"] = clubs["Bill Birth Date"]
    clubs.loc[ (clubs["Bill Birth Date"].isna() ),"Bill Birth Date" ] = str(1/1/20)
    clubs["age"] = np.array( [(2019-int("19" + "".join(bd[-2::]))) for bd in clubs[~clubs["Bill Birth Date"].isna()]["Bill Birth Date"] ])
    clubs.loc[(clubs["age"]==99),"age"] = clubs.loc[~(clubs["age"]==99),"age"].values.mean()
    clubs.loc[clubs["Ship City"].isna(),"Ship City"] = "Calistoga"
    clubs.loc[clubs["Ship State Code"].isna(),"Ship State Code"] = "CA"
    clubs.loc[clubs["Ship Zip"].isna(),"Ship Zip"] = "94515"
    clubs["isPickup"] = ~clubs["Pickup Location"].isna()

    # Convert dates to time periods (float in years)

    clubs["cancel"] = np.array([str(cancel).split("/")[0::2] for cancel in clubs["Cancel Date"]])
    clubs["signup"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Signup Date"]]
    clubs["clubLength"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if (len(end)==2) else (19 - int(start[1])) +  ((( (3+12) - int(start[0]) ) % 12  ) / 12 ) for end,start in zip(clubs["cancel"],clubs["signup"]) ]

    clubs["last_order"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Last Order Date"]]
    clubs["last_process"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Last Processed Date"]]

    # If Last Order is nan, they have just signed up. Assume 0 Time since last order if new club member
    # Not if "Time Since Last Order" is < 0,  member ordered AFTER cancellation
    clubs["Time Since Last Order"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if ((len(end)==2) & (len(start)==2)) else 0 for end,start in zip(clubs["cancel"],clubs["last_order"]) ]

    # Merge Customers onto Clubs

    clubs = clubs.merge(customers[['Customer No.','Number Of Transactions','Lifetime Value','Last Order Date','Last Order Amount']],how="left",left_on="Customer Number",right_on="Customer No.")
    clubs["Last Order Amount"] = [float("".join(ltv[1:].split(","))) if not (ltv[0] == "(") else float("".join(ltv[2:-1].split(",")))  for ltv in clubs["Last Order Amount"]]

    # Aggregate totals by Customer ID

    path = '../order_history.csv'
    data = pd.read_csv(path)

    data["Total"] = data["Quantity"] * data["Price"]
    customer_orders = data.groupby("Customer Number").sum()[["Quantity","Total"]]

    customer_orders = customer_orders[customer_orders["Quantity"]!=0]
    customer_orders["ASP"] = customer_orders["Total"] / customer_orders["Quantity"]

    clubs = clubs.merge(customer_orders,how="left",left_on="Customer Number",right_on="Customer Number") # Note there will be NaNs left
    
    # Eliminate NaNs

    clubs.loc[clubs["Quantity"].isna(),"Quantity"] = 0
    clubs.loc[clubs["Total"].isna(),"Total"] = 0
    clubs.loc[clubs["ASP"].isna(),"ASP"] = 0

    # Return final dataframe features for fitting
    '''
    #Columns before filtering

    current_columns = ['Club Status', 'Bill Birth Date', 'Ship Birth Date', 'Ship City',
       'Ship State Code', 'Ship Zip', 'Pickup Location', 'Cancel Date',
       'Cancel Reason', 'Last Processed Date', 'Last Order Date_x', 'Club',
       'Customer Number', 'Signup Date',
       "Shipments (note that these don't necessarily mean orders)",
       'Lifetime Value_x', 'ltv', 'age', 'isPickup', 'cancel', 'signup',
       'clubLength', 'Customer No.', 'Number Of Transactions',
       'Lifetime Value_y', 'Last Order Date_y', 'Last Order Amount',
       'Quantity', 'Total','ASP']
    '''
    clubs = clubs[['Club Status',
       'Ship State Code', 'Ship Zip',
       'Cancel Reason', 'Club',
       'Customer Number',
       "Shipments (note that these don't necessarily mean orders)", 'ltv', 'age', 'isPickup',
       'clubLength', 'Number Of Transactions', 'Last Order Amount',"Time Since Last Order",'Quantity', 'Total','ASP']]

    # Change names for clarification
    clubs.columns = ['Club Status',
       'State', 'Zip',
       'Cancel Reason', 'Club Tier',
       'Customer ID',
       "Shipments", 'LTV', 'Age', 'isPickup',
       'Club Length', 'Transactions', 'Last Order Amount',"Time Since Last Order",
       'Quantity', 'Orders Total','ASP']

    # Change order of columns
    clubs = clubs[['Customer ID','Club Tier','State','Zip','Age','isPickup','Transactions','Shipments',
       'Quantity','Orders Total','ASP','LTV',"Time Since Last Order",'Last Order Amount',
       'Club Status','Club Length','Cancel Reason']]

    # Combine multiple entries on Clubs for a given Customer ID ("Switched Club Level")

    for clubid in clubs["Customer ID"].unique():
        if clubs[clubs["Customer ID"]==clubid].shape[0] > 1:
            summed = clubs[clubs["Customer ID"]==clubid].groupby("Customer ID").sum()[["Shipments","Club Length","Transactions"]]
            clubs.loc[((clubs["Customer ID"]==clubid) & (clubs["Club Status"]=="Active")),"Club Length"] = summed.loc[clubid,"Club Length"]
            clubs.loc[((clubs["Shipments"]==clubid) & (clubs["Club Status"]=="Active")),"Shipments"] = summed.loc[clubid,"Shipments"]
            clubs.loc[((clubs["Transactions"]==clubid) & (clubs["Club Status"]=="Active")),"Transactions"] = summed.loc[clubid,"Transactions"]

            clubs.drop(clubs[((clubs["Customer ID"]==clubid) & (clubs["Club Status"]=="Cancelled"))].index,axis=0,inplace=True)

    # Average transaction amount

    clubs["Average Transaction"] = [ x/y if x!=0 else 0 for x,y in zip(clubs["LTV"],clubs["Transactions"])]

    # Create above / below average club length feature for dumb model

    clubs["Above Mean Club Length"] = (clubs["Club Length"]>=clubs["Club Length"].mean()).astype(int)

    # Create explicit target. 1 for churn, 0 for not

    clubs["Target"] = ~((clubs["Club Status"]=="Active") | (clubs["Club Status"]=="OnHold"))

    # One hot encode membership tiers

    clubs["Quarter Case"] = ((clubs["Club Tier"]=="3-Bottle") | (clubs["Club Tier"]=="3-Bottle (Industry)")).astype(int)
    clubs["Half Case"] = (clubs["Club Tier"]=="6-Bottle").astype(int)
    clubs["Full Case"] = (clubs["Club Tier"]=="12-Bottle").astype(int)

    return clubs

import csv
import random
random.seed(32)

def get_test_train_set(df):
    TEST_SPLIT_RATIO = 0.1
    a = np.array(df.index.tolist())
    random.shuffle(a)

    split_index = int(len(a) * TEST_SPLIT_RATIO)
    df_test = df.loc[a][:split_index]
    df_train = df.loc[a][split_index:]

    with open('../test_set.csv', mode='w') as test:
        clubwriter = csv.writer(test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        clubwriter.writerow(df_test.columns)
        for row in df_test.values:
            clubwriter.writerow(row)
    
    with open('../train_set.csv', mode='w') as train:
        clubwriter = csv.writer(train, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        clubwriter.writerow(df_train.columns)
        for row in df_train.values:
            clubwriter.writerow(row)
    

    
