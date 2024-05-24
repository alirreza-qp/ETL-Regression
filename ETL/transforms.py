import pandas as pd


# حذف رکورد کاملا خالی
def drop_record_all_nans(df):
    return df.dropna(how="all",axis=0)


# پر کردن فیلد های خالی
def fillna(data):
    data.fillna(value={
        'loan_type':data.loan_type.mode()[0],
        'loan_amount':data.loan_amount.mean(),
        'repaid':data.repaid.mode()[0],
        # 'loan_id':data.loan_id.mean(),
        'loan_start':data.loan_start.mode()[0],
        'loan_end':data.loan_end.mode()[0],
        'rate':data.rate.mean()
        
    },inplace=True)
    return data



# check outliear data with plotly
import plotly.express as px
def check_outliear_column_by_ploty(data,columns):
    fig=px.box(data,y=columns)
    fig.show()


# حذف داده های پرت
def remove_rate_outliers(data,min_r,max_r):
    df=pd.DataFrame(data)
    data=df[(df['rate']>=min_r) & (df['rate']<=max_r)]
    return data


# گسسته سازی
from sklearn.preprocessing import KBinsDiscretizer
def k_bins_discretizer(data,columns):
    dis=KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='uniform')
    for col in columns:
        data[col]=dis.fit_transform(data[[col]])
    return data



# One-Hot Encoder
def one_hot_encoder(data,columns):
    return pd.get_dummies(data,columns=columns)



# Label Encoding
from sklearn.preprocessing import LabelEncoder
def label_encoding(data,columns):
    le=LabelEncoder()
    for col in columns:
        data[col]=le.fit_transform(data[col])
    return data



# (period) ایجاد متغیر جدید 
# متغیره دوره شامل اختلاف شروع و پایان وام به ماه می باشد
# به طور مثال دوره وام 20 ماهه
def add_new_column(data,colName):
    newData=pd.to_datetime(data.loan_end) - pd.to_datetime(data.loan_start)
    data[colName] = newData.dt.days/30
    return data



# حذف ستون های مورد نظر
def drop_columns(data,columns):
    for col in columns:
        data.drop(col,axis=1,inplace=True)
    return data


# جمع سالانه مقدار وام ها
def yearly_data(data):
    data['loan_end']=pd.to_datetime(data['loan_end'])
    data.set_index('loan_end',inplace=True)
    y_data=data.resample('Y').sum()
    return y_data


# نرمال سازی
from sklearn.preprocessing import MinMaxScaler
def min_max_scaler(data,columns):
    scaler=MinMaxScaler()
    data=scaler.fit_transform(data)
    data=pd.DataFrame(data)
    data.columns=columns
    return data
