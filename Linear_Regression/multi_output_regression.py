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
    

def plot_residuals(y_test, y_pred, target_names):
    """
    Plot residuals for multiple target variables.
    """

    for i, target_name in enumerate(target_names):
        residuals = (y_test[:, i] - y_pred[:, i])

        residual_data = pd.DataFrame({
                                        "Predicted": y_pred[:, i],
                                        "Residual": residuals
                                    })

        fig = px.scatter(
                            residual_data,
                            x="Predicted",
                            y="Residual",
                            title=f"Residual Plot - {target_name}",
                            labels={
                                "Predicted": "Predicted Values",
                                "Residual": "Residuals"
                            }
                        )

        fig.add_hline(y=0, line_dash="dash")

        fig.show()


def visualize_hyperplane(model, X_test, y_test, feature_names, target_names):
    """
    Visualize a 3D slice of a multiple linear regression hyperplane
    for one or more target variables.

    Two features are varied across a grid while all remaining
    features are fixed at their mean values.

    A separate regression surface is created for each target.
    """

    # Convert X_test to DataFrame if necessary
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Convert y_test to NumPy array
    y_test = np.asarray(y_test)

    # For a single target, convert shape from (n_samples,) to (n_samples, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Print final regression equations
    print_regression_equations(model, feature_names, target_names)

    # Select two features for visualization
    feature_x = feature_names[0]
    feature_y = feature_names[1]

    # Create a grid for the two selected features
    x_range = np.linspace(X_test[feature_x].min(), X_test[feature_x].max(), 30)
    y_range = np.linspace(X_test[feature_y].min(), X_test[feature_y].max(), 30)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Create grid data with all features fixed at their mean values
    grid_data = pd.DataFrame(np.tile(X_test.mean().values, (x_grid.size, 1)), columns=feature_names)

    # Allow the selected features to vary across the grid
    grid_data[feature_x] = x_grid.ravel()
    grid_data[feature_y] = y_grid.ravel()

    # Predict all target values for every point on the grid
    predictions = model.predict(grid_data)

    # Ensure predictions are 2D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # Create a separate regression surface for each target
    for target_index, target_name in enumerate(target_names):

        # Extract predictions for the current target
        z_grid = predictions[:, target_index].reshape(x_grid.shape)

        # Create the figure
        fig = go.Figure()

        # Plot actual data points
        fig.add_trace(
            go.Scatter3d(
                x=X_test[feature_x],
                y=X_test[feature_y],
                z=y_test[:, target_index],
                mode="markers",
                name=f"Actual {target_name}"
            )
        )

        # Plot regression surface
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                name=f"Regression Surface - {target_name}",
                opacity=0.7
            )
        )

        # Configure plot layout
        fig.update_layout(
            title=f"Multiple Linear Regression Surface - {target_name}",
            scene=dict(
                xaxis_title=feature_x,
                yaxis_title=feature_y,
                zaxis_title=target_name
            ),
            width=1000,
            height=800
        )

        fig.show()
        

def print_regression_equations(model, feature_names, target_names):
    for target_index, target_name in enumerate(target_names):

        equation = (
            f"{target_name} = "
            f"{model.intercept_[target_index]:.2f}"
        )

        for feature_index, feature_name in enumerate(feature_names):
            coefficient = (model.coef_[target_index, feature_index])

            sign = ("+" if coefficient >= 0 else "-")

            equation += (f"{sign}{abs(coefficient):.2f} × {feature_name}")

        print(equation)
        

## Each target has its own independent regression coefficients.
N_FEATURES = 10
N_TARGETS = 2

X, y = make_regression(n_samples=10000, n_features=N_FEATURES, n_informative=7, n_targets=N_TARGETS, noise=10)

feature_names = [f"f{i + 1}" for i in range(N_FEATURES)]
target_names = [f"target_{i + 1}" for i in range(N_TARGETS)]

data = pd.DataFrame(X, columns=feature_names)
for i, target_name in enumerate(target_names):
    data[target_name] = y[:, i]

print(data.head())

# Correlation Heatmap
plot_correlation_heatmap(data)

# Scatter Matrix
plot_scatter_matrix(data, columns=feature_names + target_names)

# 3D Scatter Plot
plot_3d_scatter(data=data, x_column="f1",  y_column="f2", z_column="target_1", color_column="target_2")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plot_residuals(y_test=y_test, y_pred=y_pred, target_names=target_names)

for i, target_name in enumerate(target_names):
    mae = mean_absolute_error(y_test[:, i],  y_pred[:, i])
    mse = mean_squared_error(y_test[:, i],  y_pred[:, i])
    r2 = r2_score(y_test[:, i],  y_pred[:, i])

    print(f"\n{target_name}")

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

### Visulize the hyperplane and print the final equation
visualize_hyperplane(model=model, X_test=X_test, y_test=y_test,feature_names=feature_names, target_names=target_names)

## What happens when target variable is more then 1 ??
