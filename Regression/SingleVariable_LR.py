from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

def SingleVariable_LR(train_data,test_data,feature_name):
    # Get data from dataframe for independent variable and dependent valiable
    X = train_data[[feature_name]]
    y=train_data[['loan_amount']]
    
    # ----------------------------------------------------
    # Create regression model
    model = LinearRegression()
    # ----------------------------------------------------
    # Training the model
    model.fit(X,y)
    print(f"model coef : {model.coef_}")
    print(f"model intercept : {model.intercept_}")
    # ----------------------------------------------------
    # Predict values for real values
    y_pred=model.predict(X)
    # # ----------------------------------------------------
    # Calculating the Mean Squared Error (MSE)
    mse = mean_squared_error(y,y_pred)
    # # ----------------------------------------------------
    # Calculating the R-squared score
    r2 = r2_score(y,y_pred)

    # Show result 
    print(f"Mean Squared Error : {mse}")
    print(f"R-squared : {r2}")
    # # ----------------------------------------------------
    # Predict values for new values
    new_X=test_data[[feature_name]]
    predicted_y = model.predict(new_X)
    print(f"X and y : \n{new_X} ==> {predicted_y}")
    # ----------------------------------------------------

    # Plot real data
    plt.scatter(X,y,color='Blue')

    # Estimated line drawing
    plt.plot(X,y_pred,color='red')

    # Plot new data
    plt.scatter(new_X,predicted_y,color='green')

    # Add labels to plot
    plt.xlabel('X values')
    plt.ylabel('y value')
    plt.title(f'Single variable linear regression based on ({feature_name})')
    plt.legend(['Real data' , 'estimated line' , 'New data'])
    plt.grid(True)

    # Show plot
    plt.show()