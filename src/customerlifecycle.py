
import pandas as pd
import numpy as np 

import csv
import random
random.seed(32)

class OrderAnalysis:
    def __init__(self):
        self.orders = None
        self.clubs = None

    def get_orders(self):
        path = '../order_history.csv'
        data = pd.read_csv(path)

        # Make DataFrame

        cols = ['Customer Number', 'OrderCompletedDate', 'ordertype', 'BillZipCode', 'isPickup', 'Quantity', 'Price']
        data = data[cols]

        # Format columns as needed
        data["Customer Number"] = [int(x) for x in data["Customer Number"]]
        data["OrderCompletedDate"] = [x[:10] for x in data["OrderCompletedDate"]]

        # Remove unneeded info
        mask_ordertype = (data["ordertype"]=="Reservation") | (data["ordertype"]=="Website") | (data["ordertype"]=="ClubOrder") | (data["ordertype"]=="POS")
        mask_nonadmin = ~(data["Customer Number"]<5010)
        data = data[mask_ordertype & mask_nonadmin]

        # Sort by Customer, Order date
        data = data.sort_values(["Customer Number","OrderCompletedDate"])
        

        # Log price
        data['Log Price Total'] = data['Quantity'] * [np.log(x) if x > 0 else 0 for x in data['Price']]

        # Groupby Customer, Order date
        data_agg = data.groupby(["Customer Number","OrderCompletedDate","ordertype"]).sum()

        # # Featurize
        # data["ClubOnsite"] = (data["ordertype"]=="ClubOnsite")
        # data["ClubOrder"] = (data["ordertype"]=="ClubOrder")
        # data["POS"] = (data["ordertype"]=="POS")
        # data["Resevervation"] = (data["ordertype"]=="Reservation")
        self.orders = data_agg
        return data_agg

    def get_clubs(self):
        path = '../club_members.csv'
        clubs = pd.read_csv(path)

        # Important cols
        cols = ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date','Bill Zip','Pickup Location', 'Signup Date', 'Cancel Date', "Shipments (note that these don't necessarily mean orders)"]
        clubs = clubs[cols]
        clubs.columns =  ['Club', 'Club Status', 'Customer Number', 'Bill Birth Date','Bill Zip','Pickup Location', 'Signup Date', 'Cancel Date', 'Shipments']

        # Only use clubs after 2016-01-01
        clubs["Join Year"] = [int(x[-2:]) for x in clubs["Signup Date"]]
        clubs = clubs[clubs["Join Year"] > 15]

        # Only use clubs with birthday
        clubs = clubs[~clubs["Bill Birth Date"].isna()]

        # Featurize as needed
        clubs['Customer Number'] = [int(x) for x in clubs['Customer Number']]
        clubs["Age"] = [(2019-int("19" + "".join(bd[-2::]))) for bd in clubs["Bill Birth Date"]]
        clubs["Quarter Case"] = ((clubs["Club"]=="3-Bottle") | (clubs["Club"]=="3-Bottle (Industry)")).astype(int)
        clubs["Half Case"] = (clubs["Club"]=="6-Bottle").astype(int)
        clubs["Full Case"] = (clubs["Club"]=="12-Bottle").astype(int)
        clubs["Target"] = ~((clubs["Club Status"]=="Active") | (clubs["Club Status"]=="OnHold"))
        clubs["isPickup"] = ~clubs["Pickup Location"].isna()
        clubs["cancel"] = np.array([str(cancel).split("/")[0::2] for cancel in clubs["Cancel Date"]])
        clubs["signup"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Signup Date"]]
        clubs["clubLength"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if (len(end)==2) else (19 - int(start[1])) +  ((( (3+12) - int(start[0]) ) % 12  ) / 12 ) for end,start in zip(clubs["cancel"],clubs["signup"]) ]

        finalcols = ['Customer Number','Bill Zip','isPickup', 'Signup Date', 'signup','cancel','clubLength','Shipments','Age','Quarter Case','Half Case','Full Case','Target']

        # Filter clubs with incorrect birthday
        clubs = clubs[clubs["Age"]<90]
        self.clubs = clubs[finalcols]
        return clubs[finalcols]

    def merge_tables(self):
        self.orders = self.orders.reset_index()

        orders_agg = self.orders.groupby("Customer Number").sum().reset_index()[['Customer Number',"Quantity","Log Price Total"]]
        orders_count = self.orders.groupby(["Customer Number","OrderCompletedDate"]).count().reset_index().groupby(['Customer Number']).count().reset_index()[['Customer Number','OrderCompletedDate']]
        orders_by_type = self.orders.groupby(["Customer Number","ordertype"]).sum().reset_index()
        first_order_dates = self.orders.groupby(["Customer Number","OrderCompletedDate"]).sum().reset_index().groupby(['Customer Number']).min().reset_index()[["Customer Number",'OrderCompletedDate']]
        first_order_dates.columns = ["Customer Number",'Earliest Order Date']
        first_order_dates["Earliest Order Date"] = [list(str(date).split("-")[1::-1]) for date in first_order_dates["Earliest Order Date"]]

        POStotal = orders_by_type[orders_by_type["ordertype"]=='POS'][['Customer Number','Log Price Total']]
        POStotal.columns = ['Customer Number','POS Log Price Total']
        clubtotal = orders_by_type[orders_by_type["ordertype"]=='ClubOrder'][['Customer Number','Log Price Total']]
        clubtotal.columns = ['Customer Number','Club Log Price Total']
        websitetotal = orders_by_type[orders_by_type["ordertype"]=='Website'][['Customer Number','Log Price Total']]
        websitetotal.columns = ['Customer Number','Website Log Price Total']

        orders_by_type['Reservations'] = (orders_by_type["ordertype"]=="Reservation")

        clubs = self.clubs.merge(orders_agg,how="left",left_on="Customer Number",right_on="Customer Number")
        clubs = clubs.merge(orders_count,how="left",left_on="Customer Number",right_on="Customer Number")


        clubs = clubs.merge(POStotal,how="left",left_on="Customer Number",right_on="Customer Number")
        clubs = clubs.merge(clubtotal,how="left",left_on="Customer Number",right_on="Customer Number")
        clubs = clubs.merge(websitetotal,how="left",left_on="Customer Number",right_on="Customer Number")
        clubs = clubs.merge(first_order_dates,how="left",left_on="Customer Number",right_on="Customer Number")

        # Featurize

        clubs['Bill Zip'] = [int(str(x)[:5]) if str(x).isnumeric() else 94515 for x in clubs['Bill Zip']]

        orders_before = self.orders.merge(clubs[["Customer Number","signup"]],how="left",left_on="Customer Number",right_on="Customer Number")
        orders_after = self.orders.merge(clubs[["Customer Number","signup"]],how="left",left_on="Customer Number",right_on="Customer Number")
        orders_before["OrderCompletedDate"] = [list(str(date).split("-")[1::-1]) for date in orders_before["OrderCompletedDate"]]
        orders_after["OrderCompletedDate"] = [list(str(date).split("-")[1::-1]) for date in orders_after["OrderCompletedDate"]]

        mask_nan_before = ~orders_before["signup"].isna()
        mask_nan_after = ~orders_after["signup"].isna()

        orders_after = orders_after[mask_nan_after]
        orders_before = orders_before[mask_nan_before]

        mask_before = [(((int(x[1][-2:])) <= int(y[1])) & (int(x[0]) <= int(y[0]))) if not y=="nan" else False for x,y in zip(orders_before["OrderCompletedDate"],orders_before["signup"])]
        mask_after = np.array([((int(x[1][-2:]) > int(y[1])) & (int(x[0]) > int(y[0]))) if not y=="nan" else False for x,y in zip(orders_after["OrderCompletedDate"],orders_after["signup"])])

        orders_after  = orders_after[mask_after]
        orders_before = orders_before[ mask_before]

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

        # Get log order totals after joining club
        orders_after_grouped = orders_after.groupby(["Customer Number","ordertype"]).sum().reset_index()[["Customer Number","ordertype","Quantity","Log Price Total"]]
        POStotal = orders_after_grouped[orders_after_grouped["ordertype"]=='POS'][['Customer Number','Log Price Total']]
        POStotal.columns = ['Customer Number','POS Log Price Total after']
        clubtotal = orders_after_grouped[orders_after_grouped["ordertype"]=='ClubOrder'][['Customer Number','Log Price Total']]
        clubtotal.columns = ['Customer Number','Club Log Price Total after']
        websitetotal = orders_after_grouped[orders_after_grouped["ordertype"]=='Website'][['Customer Number','Log Price Total']]
        websitetotal.columns = ['Customer Number','Website Log Price Total after']

        orders_after_grouped = orders_after_grouped.merge(POStotal,how="left",left_on="Customer Number",right_on="Customer Number")
        orders_after_grouped = orders_after_grouped.merge(clubtotal,how="left",left_on="Customer Number",right_on="Customer Number")
        orders_after_grouped = orders_after_grouped.merge(websitetotal,how="left",left_on="Customer Number",right_on="Customer Number")
        # orders_after_grouped = orders_agg.merge(first_order_dates,how="left",left_on="Customer Number",right_on="Customer Number")

        orders_agg= orders_agg.merge(orders_after_grouped,how="left",left_on="Customer Number",right_on="Customer Number")
        orders_agg= orders_agg.merge(orders_before_grouped,how="left",left_on="Customer Number",right_on="Customer Number")

        orders_agg = orders_agg[['Customer Number',  'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after'  ,'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before']]

        clubs = clubs.merge(orders_agg,how="left",left_on="Customer Number",right_on="Customer Number")

        # Spending per year
        clubs['Log Spending Per Year'] = [x / y if y > 0 else 0 for x,y in zip(clubs['Log Price Total'],clubs['clubLength'])]
        # clubs = clubs[~clubs['Log Spending Per Year'].isna()]


        # Get transactions
        path = '../customer_list.csv'
        customers = pd.read_csv(path)[['Customer No.','Number Of Transactions']]
        clubs = clubs.merge(customers,how="left",left_on="Customer Number",right_on="Customer No.")

        # Finalize columns
        cols = ['Customer Number', 'Bill Zip',  'isPickup',  'clubLength',  'Shipments' , 'Age' , 'Quarter Case',  'Half Case',  'Full Case',  'Target','Quantity','Log Spending Per Year',  'Log Price Total',  'OrderCompletedDate',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total', 'POS Log Price Total after',  'Club Log Price Total after',  'Website Log Price Total after',  'POS Log Price Total Before',  'Club Log Price Total Before',  'Website Log Price Total Before', 'Number Of Transactions']
        clubs = clubs[cols]
        # clubs.columns = ['Customer Number', 'Bill Zip','isPickup','Earliest Order Date', 'signup','clubLength' , 'Shipments',  'Age',  'Quarter Case',  'Half Case',  'Full Case',  'Target',  'Quantity',  'Log Price Total','Order Counts',  'POS Log Price Total',  'Club Log Price Total',  'Website Log Price Total',  'Log Spending Per Year',  'Customer No.',  'Number Of Transactions']
        clubs = clubs.fillna(0)
        return clubs
    '''

    #  Group by etc

    # Count POS, Online when different dates
    data['POS']


    # Filter out old birthdays
    '''

    def get_test_train_set(self,df):
        TEST_SPLIT_RATIO = 0.1
        a = np.array(df.index.tolist())
        random.shuffle(a)

        split_index = int(len(a) * TEST_SPLIT_RATIO)
        df_test = df.loc[a][:split_index]
        df_train = df.loc[a][split_index:]

        with open('../reg_test_set.csv', mode='w') as test:
            clubwriter = csv.writer(test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            clubwriter.writerow(df_test.columns)
            for row in df_test.values:
                clubwriter.writerow(row)
        
        with open('../reg_train_set.csv', mode='w') as train:
            clubwriter = csv.writer(train, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            clubwriter.writerow(df_train.columns)
            for row in df_train.values:
                clubwriter.writerow(row)