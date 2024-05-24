from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

def Polynomial_R(train_df,test_df,features):
    # features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
    X = train_df[features]
    y = train_df['loan_amount']
    # ----------------------------------------
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_test = test_df[features]
    y_test= test_df[['loan_amount']]
    model = LinearRegression()
    pf = PolynomialFeatures(degree=3)
    X_train_poly = pf.fit_transform(X)
    X_test_poly = pf.fit_transform(X_test)
    model.fit(X_train_poly, y)
    score=model.score(X_test_poly, y_test)

    # Show result
    print(f"model score : {score}")

