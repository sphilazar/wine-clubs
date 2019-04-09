import pandas as pd
import numpy as np
import string
import csv
import random
from datetime import datetime
random.seed(32)

TEST_SPLIT_RATIO = 0.1

# How should I handle importing data?
CLUB_PATH = '../club_members.csv'
CUSTOMER_PATH = '../customer_list.csv'
ORDER_PATH = '../order_history.csv'


def clean_data():

   # Import data
   clubs = pd.read_csv(CLUB_PATH)
   customers = pd.read_csv(CUSTOMER_PATH)
   data = pd.read_csv(ORDER_PATH)

   # Filter out customers missing from POS
   POS_customers = data["Customer Number"].unique() #4433

   clubs["inPOS"] = [True if (x in POS_customers) else False for x in clubs["Customer Number"]]
   customers["inPOS"] = [True if (x in POS_customers) else False for x in customers["Customer No."]]

   clubs = clubs[clubs["inPOS"]] #(1203, 20)
   customers = customers[customers["inPOS"]] #(4433, 11)

   # Filter out clubs
   clubs = clubs.merge(customers,how="left",left_on="Customer Number",right_on="Customer No.")

   cols = ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date', 'Bill City', 'Bill State Code', 'Bill Zip', 'Ship Birth Date', 'Ship City', 'Ship State Code', 'Ship Zip', 'Pickup Location', 'Signup Date', 'Cancel Date', 'Cancel Reason', "Shipments (note that these don't necessarily mean orders)", 'Last Processed Date', 'Lifetime Value_x', 'Last Order Date_x','City', 'State', 'Zip Code', 'Number Of Transactions', 'Last Order Amount', 'Date Added', 'Last Modified Date']
   clubs = clubs[cols]

   # Clean Features
   clubs["LTV"] = [float("".join((ltv[1:].split(","))) ) for ltv in clubs["Lifetime Value_x"]]

   clubs.loc[clubs["Club Status"].isna() & clubs["Cancel Date"].isna(),"Club Status"] = "Active"
   clubs.loc[clubs["Club Status"].isna() & ~clubs["Cancel Reason"].isna(),"Club Status"] = "Cancelled"

   clubs.loc[ (clubs["Bill Birth Date"].isna() & ~clubs["Ship Birth Date"].isna()) , "Bill Birth Date"] = clubs["Ship Birth Date"]
   clubs.loc[ (~clubs["Bill Birth Date"].isna() & clubs["Ship Birth Date"].isna()) , "Ship Birth Date"] = clubs["Bill Birth Date"]
   clubs.loc[ (clubs["Bill Birth Date"].isna() ),"Bill Birth Date" ] = str(1/1/1966) # Default 50 years old?
   clubs["Age"] = [(2019-int("19" + "".join(str(bd[-2:])))) for bd in clubs["Bill Birth Date"]]

   # Remove admin entries
   clubs.loc[(clubs["Age"]==99),"Age"] = clubs.loc[~(clubs["Age"]==99),"Age"].values.mean()
   clubs.loc[clubs["Ship City"].isna(),"Ship City"] = "Calistoga"
   clubs.loc[clubs["Ship State Code"].isna(),"Ship State Code"] = "CA"
   clubs.loc[clubs["Ship Zip"].isna(),"Ship Zip"] = "94515"

   # Create isPickup
   clubs["isPickup"] = ~clubs["Pickup Location"].isna()

   # Convert dates to time periods (float in years)
   clubs["Cancel Date"] = np.array([str(cancel).split("/")[0::2] for cancel in clubs["Cancel Date"]])
   clubs["Signup Date"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Signup Date"]]
   clubs["Club Length"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if (len(end)==2) else (19 - int(start[1])) +  ((( (3+12) - int(start[0]) ) % 12  ) / 12 ) for end,start in zip(clubs["Cancel Date"],clubs["Signup Date"]) ]
   
   clubs["Last Order Date"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Last Order Date_x"]]
   clubs["Last Processed Date"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Last Processed Date"]]

   # If Last Order is nan, they have just signed up. Assume 0 Time since last order if new club member
   # Not if "Time Since Last Order" is < 0,  member ordered AFTER cancellation
   clubs["Time Since Last Order"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if ((len(end)==2) & (len(start)==2)) else 0 for end,start in zip(clubs["Cancel Date"],clubs["Last Order Date"]) ]

   clubs["Last Order Amount"] = [float("".join(ltv[1:].split(","))) if not (ltv[0] == "(") else float("".join(ltv[2:-1].split(",")))  for ltv in clubs["Last Order Amount"]]

   # Log dollar amounts

   data["Total"] = data["Quantity"] * [np.log(x) if x > 0 else 0 for x in data["Price"]]

   # Aggregate totals
   customer_orders = data.groupby("Customer Number").sum()[["Quantity","Total"]]
   customer_orders = customer_orders[customer_orders["Quantity"]!=0]
   customer_orders["ASP"] = customer_orders["Total"] / customer_orders["Quantity"]
    
   clubs = clubs.merge(customer_orders,how="left",left_on="Customer Number",right_on="Customer Number") 
   # Note there will be NaNs left
    
   # Eliminate NaNs]
   clubs.loc[clubs["Quantity"].isna(),"Quantity"] = 0
   clubs.loc[clubs["Total"].isna(),"Total"] = 0
   clubs.loc[clubs["ASP"].isna(),"ASP"] = 0

   # Rename some columns

   clubs.columns = ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date', 'Bill City', 'Bill State Code', 'Bill Zip', 'Ship Birth Date', 'Ship City', 'Ship State Code', 'Ship Zip', 'Pickup Location', 'Signup Date', 'Cancel Date', 'Cancel Reason', "Shipments", 'Last Processed Date', 'Lifetime Value', 'Last Order Date', 'City', 'State', 'Zip Code', 'Transactions', 'Last Order Amount', 'Date Added', 'Last Modified Date', 'LTV', 'Age', 'isPickup', 'Club Length', 'Last Order Date', 'Time Since Last Order', 'Quantity', 'Total', 'ASP']

   # Combine multiple entries on Clubs for a given Customer ID ("Switched Club Level")

   for clubid in clubs["Customer Number"].unique():
      if clubs[clubs["Customer Number"]==clubid].shape[0] > 1:
         summed = clubs[clubs["Customer Number"]==clubid].groupby("Customer Number").sum()[["Shipments","Club Length","Transactions"]]
         clubs.loc[((clubs["Customer Number"]==clubid) & (clubs["Club Status"]=="Active")),"Club Length"] = summed.loc[clubid,"Club Length"]
         clubs.loc[((clubs["Shipments"]==clubid) & (clubs["Club Status"]=="Active")),"Shipments"] = summed.loc[clubid,"Shipments"]
         clubs.loc[((clubs["Transactions"]==clubid) & (clubs["Club Status"]=="Active")),"Transactions"] = summed.loc[clubid,"Transactions"]

         clubs.drop(clubs[((clubs["Customer Number"]==clubid) & (clubs["Club Status"]=="Cancelled"))].index,axis=0,inplace=True)

   # Average transaction amount

   #  clubs["Average Transaction"] = [ x/y if x!=0 else 0 for x,y in zip(clubs["LTV"],clubs["Transactions"])]
   
   # Adjust target - more than 2 years member = 1

   clubs = clubs[~((clubs["Club Status"]=="Active") & (clubs["Club Length"]<2)) ]
   clubs["Target"] = (clubs["Club Length"] >= 2)

   # One hot encode membership tiers

   clubs["Quarter Case"] = ((clubs["Club"]=="3-Bottle") | (clubs["Club"]=="3-Bottle (Industry)")).astype(int)
   clubs["Half Case"] = (clubs["Club"]=="6-Bottle").astype(int)
   clubs["Full Case"] = (clubs["Club"]=="12-Bottle").astype(int)

   clubs["Average Transaction"] = [ x/y if y>0 else 0 for x,y in zip(clubs["LTV"],clubs
   ["Transactions"])] # Changed from Orders to LTV

   clubs["Age"] = np.log(clubs["Age"]) # Log of age
  
   clubs["ASP"] = [x/y if y>0 else 0 for x,y in zip(clubs["LTV"],clubs["Shipments"])]

   # Basic stats

   mean = clubs[clubs['Club Status']=='Cancelled']['Club Length'].mean()
   std = np.sqrt( clubs[clubs['Club Status']=='Cancelled']['Club Length'].var() )
   print("Mean CLub Length: ",mean,"std club length: ",std)
   left = clubs[clubs['Club Status']=="Cancelled"].count()
   total = clubs.shape[0]
   class_cancelled = left/total
   print("Class balance of cancelled: ",class_cancelled)

   # What is loss function
   # What is balance: 0.60 cancelled memberships, 0.47 subscriptions over 2 years old

   length = clubs[clubs['Club Length']>=2].count()
   length_portion = length/total
   print("Class balance of long memberships: ",length_portion)

   # ROC curve, AUC better classifier

   return clubs

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def get_order_history():

   path = '../club_members.csv'
   clubs = pd.read_csv(path)

   path = '../customer_list.csv'
   customers = pd.read_csv(path)

   path = '../order_history.csv'
   data = pd.read_csv(path)

   data = data[(data["ordertype"]!='ClubOrder')&(data["ordertype"]!='ClubDaily') & (data["ProductType"]=="Wine")]
   data = data.sort_values(by=['Customer Number','OrderCompletedDate'],axis=0).reset_index()

   data['OrderCompletedDate'] = [x[:10] for x in data['OrderCompletedDate']]
   data['OrderCompletedYear'] = [x[:4] for x in data['OrderCompletedDate']]
   data['OrderCompletedMonth'] = [x[5:7] for x in data['OrderCompletedDate']]
   data['OrderCompletedDay'] = [x[8:] for x in data['OrderCompletedDate']]
   data['OrdersBeforeJoin'] = [0 for x in data['OrderCompletedDate']]

   data['DaysSince'] = [0 for x in data['OrderCompletedDate']]
   data['AverageDaysSince'] = [0 for x in data['OrderCompletedDate']]
   data['TotalWine'] = [0 for x in data['OrderCompletedDate']]

   for customer in data['Customer Number'].unique():
      if (len(data[data['Customer Number']==customer]) > 1):
         start = data[data["Customer Number"]==customer].index.values.astype(int).max()
         end = start-1
         for entry in range(len(data[data['Customer Number']==customer])-1):
               # Total Days if  not club order
               print((start-entry),customer,start,end)
               date_before = data.loc[(start-entry),"OrderCompletedDate"]
               date_after = data.loc[(start-entry-1),"OrderCompletedDate"]
               days = days_between(date_before,date_after)
               print("Days: ",days)
               end -= 1
               data.loc[(start-entry-1),"DaysSince"] = days
               
         # Total Wine
         total_wine = data.loc[((data.index >= end) & (data.index <= start)),"Quantity"].sum(axis=0)
         data.loc[((data.index >= end) & (data.index <= start)),"TotalWine"] = total_wine
         
         # Orders before join
         total_orders_before = data.loc[((data.index >= end) & (data.index <= start)),"OrderCompletedDate"].unique().shape[0]
         data.loc[((data.index >= end) & (data.index <= start)),'OrdersBeforeJoin'] = total_orders_before
         
         total_days = data.loc[((data.index >= end) & (data.index <= start)),"DaysSince"].sum(axis=0)
         total_count = data.loc[((data.index >= end) & (data.index <= start)),"DaysSince"].count()
         print("Total Days: ",total_days)
         avg_days_since = (total_days / total_count -1)
         data.loc[((data.index >= end) & (data.index <= start)),"AverageDaysSince"] = avg_days_since
         # Filter out negatives

   return data

def bootstrap(df):
   below_target = list(np.argwhere(~df["Target"]).ravel())
   above_target = list(np.argwhere(df["Target"]).ravel())

   while np.abs( len(below_target) - len(above_target) ) >= 2:
      if len(below_target) < len(above_target):
         below_target.append(np.random.choice(np.array(below_target),replace=True))
      else:
         above_target.append(np.random.choice(np.array(above_target),replace=True))

   return df.iloc[above_target + below_target]

def get_test_train_set(df):
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
    