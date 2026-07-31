"""
Simple Linear Regression from Scratch using Ordinary Least Squares (OLS)

This implementation demonstrates how simple linear regression works
under the hood without using sklearn's LinearRegression model.

The model learns:

```
y = mx + b
```

where:
m = slope/coefficient
b = intercept

The parameters are calculated using the Ordinary Least Squares method:

```
m = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)

b = ȳ - m * x̄
```

"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self):
        # Slope/coefficient of the regression line
        self.m = None

        # Intercept of the regression line
        self.b = None

    def fit(self, X_train, y_train):
        """
        Calculate the slope and intercept using the
        Ordinary Least Squares (OLS) method.

        Parameters
        ----------
        X_train : pandas.Series
            Training feature values.

        y_train : pandas.Series
            Training target values.
        """

        # Calculate the mean of X and y
        x_mean = X_train.mean()
        y_mean = y_train.mean()

        numerator = 0
        denominator = 0

        # Calculate:
        # numerator = Σ((xᵢ - x̄)(yᵢ - ȳ))
        # denominator = Σ((xᵢ - x̄)²)
        
        # The .iloc[] accessor is used because train_test_split preserves the original pandas index.

        for i in range(X_train.shape[0]):
            x = X_train.iloc[i]
            y = y_train.iloc[i]

            numerator += (x - x_mean) * (y - y_mean)
            denominator += (x - x_mean) ** 2

        self.m = numerator / denominator
        self.b = y_mean - (self.m * x_mean)

    def predict(self, X_test):
        """
        Generate predictions using the learned parameters.

        Prediction formula:

            y = mx + b

        Parameters
        ----------
        X_test : float or pandas.Series
            Input feature value(s).

        Returns
        -------
        float or pandas.Series
            Predicted value(s).
        """
        print(f"Value of m: {self.m}")
        print(f"Value of b: {self.b}")
        
        return (self.m * X_test) + self.b


# Load Dataset
data_path = os.path.normpath(os.path.join(os.getcwd(), "data", "placement.csv"))
data = pd.read_csv(data_path)


# Select Features and Target
# First column is used as the independent variable (X)
X = data.iloc[:, 0]

# Last column is used as the dependent variable (y)
y = data.iloc[:, -1]

# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the Model
model = LinearRegression()

model.fit(X_train, y_train)

# Make a Prediction
# Select the any sample from the test set.
# .iloc[0] is used to access the first value by position.
# This is important because train_test_split preserves the original DataFrame/Series indices.

predicted_value = model.predict(X_test.iloc[5])
print(f"Predicted value: {predicted_value}")
