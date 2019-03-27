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
    clubs["ltv"] = [float( "".join((ltv[1:].split(","))) ) for ltv in clubs["Lifetime Value"]]
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
    clubs["cancel"] = np.array([str(cancel).split("/")[0::2] for cancel in clubs["Cancel Date"]])
    clubs["signup"] = [list(str(signup).split("/")[0::2]) for signup in clubs["Signup Date"]]
    clubs["clubLength"] = [(int(end[1]) - int(start[1])) +  ((( (int(end[0])+12) - int(start[0]) ) % 12  ) / 12 ) if (len(end)==2) else (19 - int(start[1])) +  ((( (3+12) - int(start[0]) ) % 12  ) / 12 ) for end,start in zip(clubs["cancel"],clubs["signup"]) ]

    # Merge Customers onto Clubs

    clubs = clubs.merge(customers[['Customer No.','Number Of Transactions','Lifetime Value','Last Order Date','Last Order Amount']],how="left",left_on="Customer Number",right_on="Customer No.")

    # Aggregate totals by Customer ID

    path = '../order_history.csv'
    data = pd.read_csv(path)

    data["Total"] = data["Quantity"] * data["Price"]
    customer_orders = data.groupby("Customer Number").sum()[["Quantity","Total"]]
    customer_orders = customer_orders[customer_orders["Total"]!=0]

    clubs = clubs.merge(customer_orders,how="left",left_on="Customer Number",right_on="Customer Number") # Note  there will be NaNs left

    # Return final dataframe features for fitting
    '''
    Columns before filtering

    current_columns = ['Club Status', 'Bill Birth Date', 'Ship Birth Date', 'Ship City',
       'Ship State Code', 'Ship Zip', 'Pickup Location', 'Cancel Date',
       'Cancel Reason', 'Last Processed Date', 'Last Order Date_x', 'Club',
       'Customer Number', 'Signup Date',
       "Shipments (note that these don't necessarily mean orders)",
       'Lifetime Value_x', 'ltv', 'age', 'isPickup', 'cancel', 'signup',
       'clubLength', 'Customer No.', 'Number Of Transactions',
       'Lifetime Value_y', 'Last Order Date_y', 'Last Order Amount',
       'Quantity', 'Total']
    '''
    clubs = clubs[['Club Status', 'Ship City',
       'Ship State Code', 'Ship Zip',
       'Cancel Reason', 'Last Processed Date', 'Last Order Date_x', 'Club',
       'Customer Number',
       "Shipments (note that these don't necessarily mean orders)",
       'Lifetime Value_x', 'ltv', 'age', 'isPickup',
       'clubLength', 'Customer No.', 'Number Of Transactions',
       'Lifetime Value_y', 'Last Order Date_y', 'Last Order Amount',
       'Quantity', 'Total']]

    # final_columns = 


    return clubs
