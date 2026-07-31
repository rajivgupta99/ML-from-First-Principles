## Using Ordinary Least Squares (OLS)

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for all numerical columns.

    Correlation values range from -1 to +1:

        +1  -> Strong positive correlation
         0  -> No linear correlation
        -1  -> Strong negative correlation
    """

    correlation_matrix = data.corr()

    fig = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature Correlation Heatmap",
        labels={ "color": "Correlation"})

    fig.update_layout(
                        title={
                            "text": "Feature Correlation Heatmap",
                            "x": 0.5,
                            "xanchor": "center"
                        },
                        width=800, height=700)

    fig.show()


def plot_scatter_matrix(data, columns=None):
    """
    Plot pairwise relationships between numerical columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.

    columns : list, optional
        Columns to include in the scatter matrix.
        If None, all columns are used.
    """

    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()

    fig = px.scatter_matrix(data, dimensions=columns, title="Pairwise Relationships Between Features")

    fig.update_layout(width=1000, height=1000)
    fig.show()


def plot_3d_scatter(data, x_column, y_column, z_column, color_column=None):
    """
    Create a 3D scatter plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.

    x_column : str
        Column for the X-axis.

    y_column : str
        Column for the Y-axis.

    z_column : str
        Column for the Z-axis.

    color_column : str, optional
        Column used to color the data points.
    """

    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column, title=f"3D Scatter Plot: {x_column}, {y_column}, {z_column}")

    fig.update_layout(width=900, height=700)
    fig.show()
    

def plot_residuals(y_test, y_pred):
    """
    Plot residuals against predicted values.

    Residual:
        residual = actual_value - predicted_value

    A good regression model should generally have residuals
    randomly distributed around zero.
    """

    # Calculate residuals
    residuals = y_test - y_pred

    # Create a DataFrame containing predictions and residuals
    residual_data = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})

    # Create the residual plot
    fig = px.scatter(residual_data, x="Predicted", y="Residual",
                        title="Residual Plot",
                        labels={
                            "Predicted": "Predicted Values",
                            "Residual": "Residuals"
                        }
                    )

    # Add a horizontal reference line at y = 0
    # Ideally, residuals should be randomly distributed around this line.
    fig.add_hline(y=0, line_dash="dash")
    fig.show()


def visualize_hyperplane(model, X_test, y_test, feature_names):
    """
    Visualize a 3D slice of a multiple linear regression hyperplane
    and print the final regression equation.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        Trained regression model.

    X_test : pandas.DataFrame
        Test feature data.

    y_test : pandas.Series
        Actual target values.

    feature_names : list
        List of feature names.
    """
    """ 
    Steps: 
        1. Create x_range and y_range
        2. Create 30 × 30 coordinate grid
        3. 900 combinations of feature_x and feature_y
        4. Fix all other features at their mean
        5. Create 900 valid input rows
        6. Predict y for all 900 rows
        7. Reshape 900 predictions → 30 × 30
        8. Plot x_grid, y_grid, z_grid
    """
    
    # 1. Print the Final Regression Equation
    # Convert NumPy array to DataFrame if necessary
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    equation = f"y = {model.intercept_:.2f}"

    for feature, coefficient in zip(feature_names, model.coef_):
        sign = "+" if coefficient >= 0 else "-"

        equation += (f" {sign}{abs(coefficient):.2f} × {feature}")

    print("\nFinal Regression Equation: {equation}")

    # 2. Select Three Features for Visualization
    feature_x = feature_names[0]
    feature_y = feature_names[1]
    feature_z = feature_names[2]

    # 3. Create a Grid for the Hyperplane
    x_range = np.linspace(X_test[feature_x].min(), X_test[feature_x].max(), 30)
    y_range = np.linspace(X_test[feature_y].min(), X_test[feature_y].max(), 30)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # 4. Calculate Predictions for the Hyperplane

    # Create a DataFrame with all features.
    # The first two features vary across the grid.
    # The remaining features are fixed at their mean values.

    grid_data = pd.DataFrame(
        np.tile(
            X_test.mean().values,
            (x_grid.size, 1)
        ),
        columns=feature_names
    )

    grid_data[feature_x] = x_grid.ravel()
    grid_data[feature_y] = y_grid.ravel()

    # Predict target values for every point on the grid
    z_grid = model.predict(grid_data)
    z_grid = z_grid.reshape(x_grid.shape)
    
    # 5. Create the 3D Hyperplane Visualization
    fig = go.Figure()

    # Actual data points
    fig.add_trace(
        go.Scatter3d(
            x=X_test[feature_x],
            y=X_test[feature_y],
            z=y_test,
            mode="markers",
            name="Actual Data"
        )
    )

    # Regression hyperplane
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            name="Regression Hyperplane",
            opacity=0.7
        )
    )


    fig.update_layout(
        title="Multiple Linear Regression Hyperplane",
        scene=dict(
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            zaxis_title="Target"
        ),
        width=1000,
        height=800
    )

    fig.show()
    

N_FEATURES = 10
N_TARGETS = 1
X, y = make_regression(n_samples=10000, n_features=N_FEATURES, n_informative=7, n_targets=N_TARGETS, noise=10)

feature_names = [f"f{i + 1}" for i in range(N_FEATURES)]
data = pd.DataFrame(X, columns=feature_names)
data["target"] = y
print(data.head())

# Correlation Heatmap
plot_correlation_heatmap(data)

# Scatter Matrix
plot_scatter_matrix(data, columns=feature_names + ["target"])

# 3D Scatter Plot
plot_3d_scatter(data=data, x_column="f1",  y_column="f2", z_column="f3", color_column="target")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plot_residuals(y_test=y_test, y_pred=y_pred)

print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")


### Visulize the hyperplane and print the final equation
visualize_hyperplane(model=model, X_test=X_test, y_test=y_test,feature_names=feature_names)

## What happens when target variable is more then 1 ??
