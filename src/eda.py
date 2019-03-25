import pandas as pd 
import numpy as np

'''
import file, throw into dataframe
'''

def get_dataframe(filename):
    filename = "data/Order-Data-Sample.csv"
    return pd.read_csv(filename)

def print_stats(data):
    print(data.info())
    print(data.describe())
    for col in data.columns:
        if type(data[col].values[0])==str:
            print(col,data[col].unique())

def eliminate_nans(data):
    '''
    This function takes in a dataframe with nan values... and replaces str nan values with '0'
    '''
    for col in data.columns:
        if data[col].isna().sum() > 0 and type(data[col].values[0])==str:
            inds = data[col].isna()
            data[inds] = '0'

def get_intimacy_score(data):
    '''
    This function takes in a dataframe and creates a feature for total bottles poured that day and applies feature to all customers for that day.
    '''
    conditions = (data["ordertype"]=="Promo") & (data["ProductType"]=="Wine")
    cogs = data[conditions]
    total_bottles_poured = cogs.groupby("CalendarDate").sum()["Quantity"]
    return total_bottles_poured
    # need to merge back onto main df
    # need to normalize for non-busy days (during the week, etc)

def get_dow(data):
    data["CalendarDate"] = np.array([x[:10] for x in data["OrderCompletedDate"]])
    data = data.drop(columns=["OrderCompletedDate","OrderCompletedDate.1"])
    return data

def get_revenue_by_customer(data):
    data["Total"] = data["Quantity"] * data["Price"]
    revenue_by_user = data.groupby("BillCustomerNumber").sum()["Total"]
    return revenue_by_user
    
def get_age(data,today_date):
    today_year = int(today_date[:4])
    data['Age'] = [(today_year - int(x[:4])) for x in data["BillBirthDate"]]

def eliminate_noise_cols(data):
    data = data.drop(columns=["orderstatus","BillFirstName","BillLastName","BillCompany","BillAddress","BillPhone","BillEmail","ShipBirthDate","ShipFirstName","ShipLastName","BrandKey","ProductSKU","OrderNotes","ShipZipCode","BillZipCode","ShipCompany","BillAddress2","ShipEmail","ShipPhone","ShipAddress2","ShipAddress"])
    return data

def get_time_periods(data):
    ordered = data.sort_values(by=['BillCustomerNumber', 'CalendarDate'])
    # data["DaysBetweenVisits1"] = 
    return ordered

def get_skipped_shipment(data):
    # if no shipment went out to given customer on given date after joining
    pass







