
import pandas as pd
import numpy as np
import string
import csv
import random
from datetime import datetime
random.seed(32)

TEST_SPLIT_RATIO = 0.1

CLUB_PATH = '../club_members.csv'
CUSTOMER_PATH = '../customer_list.csv'
ORDER_PATH = '../order_history.csv'

TODAY_DATE = '04/01/19'

class OrderAnalysis:
    def __init__(self):
        self.clubs = pd.read_csv(CLUB_PATH)
        self.customers = pd.read_csv(CUSTOMER_PATH)
        self.data = pd.read_csv(ORDER_PATH)

    def filter_POS(self):
        '''
        This function filters out customers who don't appear in the POS data
        '''
        POS_customers = self.data["Customer Number"].unique() 

        self.clubs["inPOS"] = [True if (x in POS_customers) else False for x in self.clubs["Customer Number"]]
        self.customers["inPOS"] = [True if (x in POS_customers) else False for x in self.customers["Customer No."]]

        self.clubs = self.clubs[self.clubs["inPOS"]] 
        self.customers = self.customers[self.customers["inPOS"]]

    def combine_club_customer(self,cols):
        '''
        This function merges customer table onto clubs table and filters clubs to only the columns passed in
        '''
        self.clubs = self.clubs.merge(self.customers,how="left",left_on="Customer Number",right_on="Customer No.")
        self.clubs = self.clubs[cols]

    def clean_features(self):
        '''
        This function creates more useful features from existing data, including:
            LTV – LifeTime Value ($)
            Age
            isPickup – customer picks up shipments at the winery

        This function also removes NaN or clear administrative entries from the following columns:
            Club Status – Active or Cancelled
            Bill Birth Date, Ship Birth Date
            Age
        '''
        # Clean Features
        self.clubs["LTV"] = [float("".join((ltv[1:].split(","))) ) for ltv in self.clubs["Lifetime Value_x"]]
        self.clubs["Last Order Date"] = self.clubs["Last Order Date_x"]
        self.clubs["Shipments"] = self.clubs["Shipments (note that these don't necessarily mean orders)"]

        self.clubs.loc[self.clubs["Club Status"].isna() & self.clubs["Cancel Date"].isna(),"Club Status"] = "Active"
        self.clubs.loc[self.clubs["Club Status"].isna() & ~self.clubs["Cancel Reason"].isna(),"Club Status"] = "Cancelled"

        self.clubs.loc[ (self.clubs["Bill Birth Date"].isna() & ~self.clubs["Ship Birth Date"].isna()) , "Bill Birth Date"] = self.clubs["Ship Birth Date"]
        self.clubs.loc[ (~self.clubs["Bill Birth Date"].isna() & self.clubs["Ship Birth Date"].isna()) , "Ship Birth Date"] = self.clubs["Bill Birth Date"]
        self.clubs.loc[ (self.clubs["Bill Birth Date"].isna() ),"Bill Birth Date" ] = str(1/1/1966) # Default 50 years old?
        self.clubs["Age"] = [(2019-int("19" + "".join(str(bd[-2:])))) for bd in self.clubs["Bill Birth Date"]]

        # Remove admin entries
        self.clubs.loc[(self.clubs["Age"]==99),"Age"] = self.clubs.loc[~(self.clubs["Age"]==99),"Age"].values.mean()
        self.clubs.loc[self.clubs["Ship City"].isna(),"Ship City"] = "Calistoga"
        self.clubs.loc[self.clubs["Ship State Code"].isna(),"Ship State Code"] = "CA"
        self.clubs.loc[self.clubs["Ship Zip"].isna(),"Ship Zip"] = "94515"

        # Create isPickup
        self.clubs["isPickup"] = ~self.clubs["Pickup Location"].isna()



        # Convert dates to time periods (float in years)
        # clubs["Cancel Date"] = np.array([str(cancel).split("/")[0::2] for cancel in clubs["Cancel Date"]])
        # clubs["Signup Date"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Signup Date"]]
        self.clubs["Club Length"] = [self.calc_club_length(start,end) if type(end)==str else self.calc_club_length(start,TODAY_DATE) for end,start in zip(self.clubs["Cancel Date"],self.clubs["Signup Date"]) ]
        
        # self.clubs["Last Order Date"] = [list(str(signup).split("/")[0::2]) for signup in self.clubs["Last Order Date_x"]]
        # clubs["Last Processed Date"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Last Processed Date"]]

        # If Last Order is nan, they have just signed up. Assume 0 Time since last order if new club member
        # Not if "Time Since Last Order" is < 0,  member ordered AFTER cancellation
        self.clubs["Time Since Last Order"] =  [self.calc_club_length(last,end) if ((type(end)==str) & (type(last)==str)) else self.calc_club_length(start,TODAY_DATE) for end,last,start in zip(self.clubs["Cancel Date"],self.clubs["Last Order Date"],self.clubs["Signup Date"]) ]

        self.clubs["Last Order Amount"] = [float("".join(ltv[1:].split(","))) if not (ltv[0] == "(") else float("".join(ltv[2:-1].split(",")))  for ltv in self.clubs["Last Order Amount"]]

        self.clubs['Bill Zip'] = [int(str(x)[:5]) if str(x).isnumeric() else 94515 for x in self.clubs['Bill Zip']]

        # Average transaction amount

        self.clubs["Average Transaction"] = [ x/y if x!=0 else 0 for x,y in zip(self.clubs["LTV"],self.clubs["Number Of Transactions"])]

        # Aggregate totals
        self.data["Log Price Total"] = self.data["Quantity"] * [np.log(x) if x > 0 else 0 for x in self.data["Price"]]

        customer_orders = self.data.groupby("Customer Number").sum()[["Quantity","Log Price Total"]]
        customer_orders = customer_orders[customer_orders["Quantity"]!=0]
        customer_orders["ASP"] = customer_orders["Log Price Total"] / customer_orders["Quantity"]
            
        self.clubs = self.clubs.merge(customer_orders,how="left",left_on="Customer Number",right_on="Customer Number") 
        # Note there will be NaNs left
            
        # Eliminate NaNs]
        self.clubs.loc[self.clubs["Quantity"].isna(),"Quantity"] = 0
        self.clubs.loc[self.clubs["Log Price Total"].isna(),"Log Price Total"] = 0
        self.clubs.loc[self.clubs["ASP"].isna(),"ASP"] = 0



    def handle_club_switch(self):
        '''
        This function combine multiple entries on Clubs for a given Customer ID ("Switched Club Level")
        '''
        for clubid in self.clubs["Customer Number"].unique():
            if self.clubs[self.clubs["Customer Number"]==clubid].shape[0] > 1:
                summed = self.clubs[self.clubs["Customer Number"]==clubid].groupby("Customer Number").sum()[["Shipments","Club Length","Number Of Transactions"]]
                self.clubs.loc[((self.clubs["Customer Number"]==clubid) & (self.clubs["Club Status"]=="Active")),"Club Length"] = summed.loc[clubid,"Club Length"]
                self.clubs.loc[((self.clubs["Shipments"]==clubid) & (self.clubs["Club Status"]=="Active")),"Shipments"] = summed.loc[clubid,"Shipments"]
                self.clubs.loc[((self.clubs["Number Of Transactions"]==clubid) & (self.clubs["Club Status"]=="Active")),"Number Of Transactions"] = summed.loc[clubid,"Number Of Transactions"]

                self.clubs.drop(self.clubs[((self.clubs["Customer Number"]==clubid) & (self.clubs["Club Status"]=="Cancelled"))].index,axis=0,inplace=True)

    def make_target(self):
        '''
        This function creates our target, which we define as being a member for more than 2 years and filters the dataframe for entries that would lead too leakage
        '''
        self.clubs = self.clubs[~((self.clubs["Club Status"]=="Active") & (self.clubs["Club Length"]<2)) ]
        self.clubs["Target"] = (self.clubs["Club Length"] >= 2)

    def one_hot(self):
        self.clubs["Quarter Case"] = ((self.clubs["Club"]=="3-Bottle") | (self.clubs["Club"]=="3-Bottle (Industry)")).astype(int)
        self.clubs["Half Case"] = (self.clubs["Club"]=="6-Bottle").astype(int)
        self.clubs["Full Case"] = (self.clubs["Club"]=="12-Bottle").astype(int)

    def clean_data(self):

        self.filter_POS()

        cols = ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date', 'Bill City', 'Bill State Code', 'Bill Zip', 'Ship Birth Date', 'Ship City', 'Ship State Code', 'Ship Zip', 'Pickup Location', 'Signup Date', 'Cancel Date', 'Cancel Reason', "Shipments (note that these don't necessarily mean orders)", 'Last Processed Date', 'Lifetime Value_x', 'Last Order Date_x','City', 'State', 'Zip Code', 'Number Of Transactions', 'Last Order Amount', 'Date Added', 'Last Modified Date']

        self.combine_club_customer(cols)
        self.clean_features()

        # # Rename some columns
        # self.clubs.columns = ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date', 'Bill City', 'Bill State Code', 'Bill Zip', 'Ship Birth Date', 'Ship City', 'Ship State Code', 'Ship Zip', 'Pickup Location', 'Signup Date', 'Cancel Date', 'Cancel Reason', "Shipments", 'Last Processed Date', 'Lifetime Value', 'Last Order Date', 'City', 'State', 'Zip Code', 'Transactions', 'Last Order Amount', 'Date Added', 'Last Modified Date', 'LTV', 'Age', 'isPickup', 'Club Length', 'Last Order Date', 'Time Since Last Order', 'Quantity', 'Total', 'ASP']

        self.handle_club_switch()

        self.make_target()

        # One hot encode membership tiers
        self.one_hot()

        # # Basic stats -  delete
        # mean = self.[self.['Club Status']=='Cancelled']['Club Length'].mean()
        # std = np.sqrt( self.[self.['Club Status']=='Cancelled']['Club Length'].var() )
        # print("Mean CLub Length: ",mean,"std club length: ",std)
        # left = self.[self.['Club Status']=="Cancelled"].count()
        # total = self..shape[0]
        # class_cancelled = left/total
        # print("Class balance of cancelled: ",class_cancelled)

        # length = self.clubs[self.clubs['Club Length']>=2].count()
        # length_portion = length/total
        # print("Class balance of long memberships: ",length_portion)

        # ROC curve, AUC better classifier
        return self.clubs

    def bootstrap(self,df):
        '''
        This function takes in a DataFrame, counts total membership in each class by finding all indices of each class and balances out each class if necessary by randomly picking an index belonging to that class.
        Returns a dataframe containing all the original data plus bootstrapped data per the method above.
        '''

        below_target = list(np.argwhere(~df["Target"]).ravel())
        above_target = list(np.argwhere(df["Target"]).ravel())

        while np.abs( len(below_target) - len(above_target) ) >= 2:
            if len(below_target) < len(above_target):
                below_target.append(np.random.choice(np.array(below_target),replace=True))
            else:
                above_target.append(np.random.choice(np.array(above_target),replace=True))

        return df.iloc[above_target + below_target]

    def calc_club_length(self,d1,d2):
        '''
        Calculates the elapsed time in years between 2 Signup Date and Cancellation Date (strings)
        '''
        d1 = datetime.strptime(d1, "%m/%d/%y")
        d2 = datetime.strptime(d2, "%m/%d/%y")
        return ((d2 - d1).days / 365)

    def days_between(self,d1, d2):
        '''
        Calculates the elapsed time in days between 2 calendar dates (strings)
        '''
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return np.abs((d2 - d1).days)

    def get_order_history(self):

        # Featurize
        self.data['OrderCompletedDate'] = [x[:10] for x in self.data['OrderCompletedDate']]
        self.data = self.data.merge(self.clubs[["Customer Number","Signup Date"]],how="left",left_on="Customer Number",right_on="Customer Number")

        # Set up features for assignment 
        self.customers['OrdersBeforeJoin'] = [0 for x in self.customers['Customer No.']]
        self.customers['DaysSince'] = [0 for x in self.customers['Customer No.']]
        self.customers['AverageDaysSince'] = [0 for x in self.customers['Customer No.']]
        self.customers['TotalWineBefore'] = [0 for x in self.customers['Customer No.']]
        
        self.data['OrdersBeforeJoin'] = [0 for x in self.data['OrderCompletedDate']]
        self.data['DaysSince'] = [0 for x in self.data['OrderCompletedDate']]
        self.data['AverageDaysSince'] = [0 for x in self.data['OrderCompletedDate']]
        self.data['TotalWineBefore'] = [0 for x in self.data['OrderCompletedDate']]

        # Filter irrelevant entries
        mask_nonadmin = ~(self.data["Customer Number"]<5010)
        mask_bytype = ((self.data["ordertype"]!='ClubOrder')&(self.data["ordertype"]!='ClubDaily') & (self.data["ProductType"]=="Wine"))
        self.data = self.data[mask_nonadmin & mask_bytype]

        # Sort 
        self.data = self.data.sort_values(by=['Customer Number','OrderCompletedDate'],axis=0).reset_index()

        # Calculate aggregates by customer
        date_mask = [self.isbefore(x,y) if (type(y)==str) else False for x,y in zip(self.data["OrderCompletedDate"],self.data["Signup Date"])]
        
        for customer in self.data['Customer Number'].unique():
            # Filter out if after signup
            customer_mask = self.data['Customer Number']==customer
            if (len(self.data[customer_mask & date_mask]) > 1):
                start = self.data[customer_mask & date_mask].index.values.astype(int).max()
                end = start-1
                for entry in range(len(self.data[self.data['Customer Number']==customer])-1):
                    # Total Days if not club order
                    date_before = self.data.loc[(start-entry),"OrderCompletedDate"]
                    date_after = self.data.loc[(start-entry-1),"OrderCompletedDate"]
                    days = self.days_between(date_after,date_before)
                    end -= 1
                    self.data.loc[(start-entry-1),"DaysSince"] = days
                    
                # Assign Total Wine
                total_wine = self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"Quantity"].sum(axis=0)
                self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"TotalWineBefore"] = total_wine
                # Assign Orders before joining club
                total_orders_before = self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"OrderCompletedDate"].unique().shape[0]
                self.data.loc[((self.data.index >= end) & (self.data.index <= start)),'OrdersBeforeJoin'] = total_orders_before
                
                # Assign average days between transactions
                total_days = self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"DaysSince"].sum(axis=0)
                total_count = self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"DaysSince"].count()
                avg_days_since = (total_days / (total_count -1))
                self.data.loc[((self.data.index >= end) & (self.data.index <= start)),"AverageDaysSince"] = avg_days_since

                cust_index = self.customers[self.customers["Customer No."]==customer].index
                self.customers.loc[cust_index,"AverageDaysSince"] = avg_days_since
                self.customers.loc[cust_index,"TotalWineBefore"] = total_wine
                self.customers.loc[cust_index,"OrdersBeforeJoin"] = total_orders_before

        return self.data

    def isbefore(self,d1,d2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%m/%d/%y")
        return (((d2 - d1).days) >= 0)

    def merge_tables(self):
        # self.data = self.data.reset_index()

        # Create aggregate tables - check tables.
        orders_agg = self.data.groupby("Customer Number").sum().reset_index()[['Customer Number',"Quantity","Log Price Total"]]
        orders_count = self.data.groupby(["Customer Number","OrderCompletedDate"]).count().reset_index().groupby(['Customer Number']).count().reset_index()[['Customer Number','OrderCompletedDate']]
        orders_by_type = self.data.groupby(["Customer Number","ordertype"]).sum().reset_index()
        first_order_dates = self.data.groupby(["Customer Number","OrderCompletedDate"]).sum().reset_index().groupby(['Customer Number']).min().reset_index()[["Customer Number",'OrderCompletedDate']]
        first_order_dates.columns = ["Customer Number",'Earliest Order Date']
        first_order_dates["Earliest Order Date"] = [list(str(date).split("-")[1::-1]) for date in first_order_dates["Earliest Order Date"]]

        POStotal = orders_by_type[orders_by_type["ordertype"]=='POS'][['Customer Number','Log Price Total']]
        POStotal.columns = ['Customer Number','POS Log Price Total']
        clubtotal = orders_by_type[orders_by_type["ordertype"]=='ClubOrder'][['Customer Number','Log Price Total']]
        clubtotal.columns = ['Customer Number','Club Log Price Total']
        websitetotal = orders_by_type[orders_by_type["ordertype"]=='Website'][['Customer Number','Log Price Total']]
        websitetotal.columns = ['Customer Number','Website Log Price Total']

        orders_by_type['Reservations'] = (orders_by_type["ordertype"]=="Reservation")

        self.clubs = self.clubs.merge(orders_agg,how="left",left_on="Customer Number",right_on="Customer Number")
        self.clubs = self.clubs.merge(orders_count,how="left",left_on="Customer Number",right_on="Customer Number")

        self.clubs = self.clubs.merge(POStotal,how="left",left_on="Customer Number",right_on="Customer Number")
        self.clubs = self.clubs.merge(clubtotal,how="left",left_on="Customer Number",right_on="Customer Number")
        self.clubs = self.clubs.merge(websitetotal,how="left",left_on="Customer Number",right_on="Customer Number")
        self.clubs = self.clubs.merge(first_order_dates,how="left",left_on="Customer Number",right_on="Customer Number")

        # Join customer info on orders info

        orders_before = self.data.merge(self.clubs[["Customer Number","Signup Date"]],how="left",left_on="Customer Number",right_on="Customer Number")
        orders_after = self.data.merge(self.clubs[["Customer Number","Signup Date"]],how="left",left_on="Customer Number",right_on="Customer Number")
        orders_before["OrderCompletedDate"] = [date[:10]for date in orders_before["OrderCompletedDate"]]
        orders_after["OrderCompletedDate"] = [date[:10] for date in orders_after["OrderCompletedDate"]]

        # Eliminate customers who haven't actually signed up for club

        mask_nan_before = ~orders_before["Signup Date_x"].isna()
        mask_nan_after = ~orders_after["Signup Date_x"].isna()

        orders_after = orders_after[mask_nan_after]
        orders_before = orders_before[mask_nan_before]

        # Filter orders before and orders after

        mask_before = [self.isbefore(x,y) if (type(y)==str) else False for x,y in zip(orders_before["OrderCompletedDate"],orders_before["Signup Date_x"])]
        # mask_after = [~self.isbefore(x,y) if (type(y)==str) else False for x,y in zip(orders_after["OrderCompletedDate"],orders_after["Signup Date"])]

        orders_before = orders_before[mask_before]
        # orders_after  = orders_after[mask_after]

        # Get log order totals before joining club
        orders_before_grouped = orders_before.groupby(["Customer Number","ordertype"]).sum().reset_index()[["Customer Number","ordertype","Quantity","Log Price Total"]]
        POStotal = orders_before_grouped[orders_before_grouped["ordertype"]=='POS'][['Customer Number','Log Price Total']]
        POStotal.columns = ['Customer Number','POS Log Price Total Before']
        clubtotal = orders_before_grouped[orders_before_grouped["ordertype"]=='ClubOrder'][['Customer Number','Log Price Total']]
        clubtotal.columns = ['Customer Number','Club Log Price Total Before']
        websitetotal = orders_before_grouped[orders_before_grouped["ordertype"]=='Website'][['Customer Number','Log Price Total']]
        websitetotal.columns = ['Customer Number','Website Log Price Total Before']

        orders_before_grouped = orders_before_grouped.merge(POStotal,how="left",left_on="Customer Number",right_on="Customer Number")
        orders_before_grouped = orders_before_grouped.merge(clubtotal,how="left",left_on="Customer Number",right_on="Customer Number")
        orders_before_grouped = orders_before_grouped.merge(websitetotal,how="left",left_on="Customer Number",right_on="Customer Number")

        orders_agg= orders_agg.merge(orders_before_grouped,how="left",left_on="Customer Number",right_on="Customer Number")

        orders_agg = orders_agg[['Customer Number' ,'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before']]

        self.clubs = self.clubs.merge(orders_agg,how="left",left_on="Customer Number",right_on="Customer Number")

        # Spending per year
        self.clubs['Log Spending Per Year'] = [x / y if y > 0 else 0 for x,y in zip(self.clubs['Log Price Total_x'],self.clubs['Club Length'])]

        self.clubs = self.clubs.merge(self.customers[['Customer No.','Number Of Transactions']],how="left",left_on="Customer Number",right_on="Customer No.")

        self.clubs = self.clubs.merge(self.customers[['Customer No.','AverageDaysSince','TotalWineBefore','OrdersBeforeJoin']],how="left",left_on="Customer Number",right_on="Customer No.")

        cols = ['Customer Number', 'Bill Zip',  'isPickup',  'Club Length',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case',  'Target','Quantity_x','Log Spending Per Year' , 'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before','Number Of Transactions_y','AverageDaysSince','TotalWineBefore','OrdersBeforeJoin','ASP','Average Transaction']
        self.clubs = self.clubs[cols]
        self.clubs = self.clubs.fillna(0)
        return self.clubs

    def get_test_train_set(self,df):
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