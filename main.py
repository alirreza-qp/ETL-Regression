from ETL.extractions import *
from ETL.transforms import *
from ETL.loads import *


# # # ---------------------------------------------------------------             --------------------------------------------------------------- # # #
# # # ---------------------------------------------------------------  SECTION 1  --------------------------------------------------------------- # # #
# # # ---------------------------------------------------------------             --------------------------------------------------------------- # # #
def display_data(data):
    print(data.to_string())
    data.info()
    print(80*"*")

data=extract_from_csv("./data/loans.csv")
# display_data(data)
# ---------------------------------------------------------------
print(data.isnull().sum())   # ---> There aren't null
# حذف رکورد خالی
# data=drop_record_all_nans(data)
# display_data(data)
# ------------------------------------------
# پر کردن فیلد های خالی
# data=fillna(data)
# display_data(data)
# ------------------------------------------
# شناسایی داده های پرت برای متغیر های عددی
# check_outliear_column_by_ploty(data,['loan_amount'])        # loan_amount -> without outliear data
# check_outliear_column_by_ploty(data,['repaid'])             # repaid -> without outliear data
# check_outliear_column_by_ploty(data,['rate'])               # rate -> found 3 outliear data

# حذف داده های پرت  --------------------> remove outliear data from [rate]
data=remove_rate_outliers(data,0,10)
# display_data(data)

# check_outliear_column_by_ploty(data,['rate'])
# ------------------------------------------

# # گسسته سازی
data=k_bins_discretizer(data,['loan_amount','rate'])
display_data(data)


# # One-Hot Encoder
data=one_hot_encoder(data,['repaid'])
# display_data(data)


# Label Encoding
data=label_encoding(data,['loan_type'])
# display_data(data)


# (period) ایجاد متغیر جدید 
data=add_new_column(data,'period')
# display_data(data)


# حذف ستون های ناکار آمد از دیتافریم
data=drop_columns(data,['client_id','loan_id','loan_start'])
# display_data(data2)


# جمع سالانه مقادیر
# براي پيش بيني بهتر مدل داده ها را بر اساس تاريخ پايان وام به صورت سالانه جمع ميكنيم
# كه از 440 سطر به 17 سطر تبديل مي شوند
data=yearly_data(data)
print(data)


# نرمال سازی
data=min_max_scaler(data,['loan_type','loan_amount','rate','repaid_0','repaid_1','period'])
display_data(data)

# print(data.shape)        ---->     17
print("-"*100) # ---------------------------------------------------------------
# Split dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(train_data.shape)
print(test_data.shape)
print("-"*100) # ---------------------------------------------------------------
# # # ---------------------------------------------------------------             --------------------------------------------------------------- # # #
# # # ---------------------------------------------------------------  SECTION 2  --------------------------------------------------------------- # # #
# # # ---------------------------------------------------------------             --------------------------------------------------------------- # # #
# load_csv(data,'./data/target.csv')
from Regression.SingleVariable_LR import *

# SingleVariable_LR(train_data,test_data,'repaid_0')   #  R-squared : 0.65
# SingleVariable_LR(train_data,test_data,'repaid_0')   #  R-squared : 0.65
# SingleVariable_LR(train_data,test_data,'rate')       #  R-squared : 0.74 
# SingleVariable_LR(train_data,test_data,'loan_type')  #  R-squared : 0.77 
SingleVariable_LR(train_data,test_data,'period')       #  R-squared : 0.89    ---> period is the best feature for this regression

print("-"*100) # ---------------------------------------------------------------

from Regression.MultiVariable_LR import *

# MultiVariable_LR(train_data,test_data,['rate','period'])         #  R-squared : 0.9623    
# MultiVariable_LR(train_data,test_data,['loan_type','rate'])      #  R-squared : 0.9364
MultiVariable_LR(train_data,test_data,['loan_type','period'])      #  R-squared : 0.9640     ---> loan_type and period is the best features for this regression

print("-"*100) # ---------------------------------------------------------------

from Regression.Polynomial_R import *

Polynomial_R(train_data,test_data,['loan_type','loan_amount','rate','repaid_0','repaid_1','period'])    #  score : 0.99    
