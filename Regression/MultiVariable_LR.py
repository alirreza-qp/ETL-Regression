import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def MultiVariable_LR(train_df,test_df,features_list):
    X = train_df[features_list]
    y = train_df[['loan_amount']]
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # ------------------------------------------------
    X_test = test_df[features_list]
    y_test= test_df[['loan_amount']]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    # Create regression model
    model = LinearRegression()

    # Training the model
    model.fit(X, y)

    # Using the model to predict the test data
    y_pred = model.predict(X_test)

    # Calculating the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculating the R-squared score
    r2 = r2_score(y_test, y_pred)

    # Show result
    print(f"Mean Squared Error : {mse}")
    print(f"R-squared : {r2}")
    print(f"model intercept : {model.intercept_}")
    print(f"model conf : {model.coef_}")


    # Print prediction for new data
    print("Predictions for new data:")
    for i in range(len(X_test)):
        print(f"Features: {X_test[i]}, Prediction: {y_pred[i]}")


    # Draw a regression plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot real data
    ax.scatter(X[:,1], X[:,2], y, color='blue')

    # Estimated area drawing ---> IndexError
    # ax.plot_trisurf(X[:,1].flatten(), X[:,2].flatten(), y_pred.flatten(), color='#ff0', alpha=0.5)

    # Plot new data
    ax.scatter(X_test[:,1], X_test[:,2], y_pred, color='green')

    # Add labels to plot
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.title(f'Multi variable linear regression based on ({features_list})')
    # Show plot
    plt.show()