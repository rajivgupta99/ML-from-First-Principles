import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.pyplot as plt


class GradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.weights = None
        self.bias = 0.0
        
        self.weight_history = []
        self.bias_history = []
        self.loss_history = []
        self.gradient_history = []
        
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.epochs):
            # Predictions
            y_pred = X_train @ self.weights + self.bias
            
            # Loss in case of LR - Based on loss function selection, the gradients formula will changes but method to update weights and bias remains same
            loss = np.mean((y_train - y_pred) ** 2)
        
            # Gradients
            dw = (-2 / n_samples) * (X_train.T @ (y_train - y_pred))
            db = (-2 / n_samples) * np.sum(y_train - y_pred)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Gradient magnitude
            gradient = np.sqrt(np.sum(dw ** 2) + db ** 2) # np.linalg.norm(np.append(dw, db))
            
            # Store history
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            self.loss_history.append(loss)
            self.gradient_history.append(gradient)
        
    def predict(self, X):
        return X @ self.weights + self.bias


def plot_weights(model, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    weight_history = np.array(model.weight_history)

    for i in range(weight_history.shape[1]):
        ax.plot(weight_history[:, i], label=f"$w_{i+1}$")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.set_title("Weights vs Epoch")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MaxNLocator(8))

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    if created_fig:
        plt.show()


def plot_bias(model, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    bias_history = np.array(model.bias_history)

    ax.plot(bias_history, linewidth=2, label="Bias")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    ax.set_title("Bias vs Epoch")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MaxNLocator(8))

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    if created_fig:
        plt.show()
    

def plot_loss(model, ax=None):
    """
    The loss curve tells you how well the model is fitting the data.
    What this graph tells you
        Steep decrease initially → Gradient descent is making large improvements.
        Gradual flattening → The model is approaching the optimum.
        Nearly horizontal line → Convergence; gradients are close to zero.
        Oscillations → Learning rate may be too high.
        Increasing loss → Gradient descent is diverging (learning rate is much too high).
    """
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    loss_history = np.array(model.loss_history)

    ax.plot(loss_history, linewidth=2, color="tab:red", label="Training Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Loss Curve")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(10))

    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    ax.legend()

    if created_fig:
        plt.show()
    
    
def plot_gradients(model, ax=None):
    # The gradient magnitude tells you how the optimizer is behaving.
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    gradient_history = np.array(model.gradient_history)

    ax.plot(
        gradient_history,
        linewidth=2,
        color="tab:green",
        label="Gradient Magnitude"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$||\nabla J||$")
    ax.set_title("Gradient Magnitude vs Epoch")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(10))

    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    ax.legend()

    if created_fig:
        plt.show()


def plot_predictions(model, X_test, y_test, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_fig = True

    y_pred = model.predict(X_test)

    ax.scatter(y_test, y_pred, alpha=0.6)

    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())

    ax.plot([mn, mx], [mn, mx], color="red", linewidth=2)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")

    ax.grid(True)

    if created_fig:
        plt.show()
        

def plot_residuals(model, X_test, y_test, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_fig = True

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.5)

    ax.axhline(0, color="red", linestyle="--")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")

    ax.grid(True)

    if created_fig:
        plt.show()


def plot_lr_comparison(X_train, y_train, learning_rates=(0.0001, 0.001, 0.01, 0.1), epochs=200, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    for lr in learning_rates:
        model = GradientDescentRegressor(learning_rate=lr, epochs=epochs)
        model.fit(X_train, y_train)

        ax.plot(
            model.loss_history,
            linewidth=2,
            label=f"LR = {lr}"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Learning Rate Comparison")

    ax.grid(True)
    ax.legend()

    if created_fig:
        plt.show()
    

## Training Dashboard
def training_dashboard(model):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    fig.suptitle("Gradient Descent Training Dashboard", fontsize=14, fontweight="bold")

    plot_loss(model, ax=axes[0, 0])
    plot_gradients(model, ax=axes[0, 1])

    plot_weights(model, ax=axes[1, 0])
    plot_bias(model, ax=axes[1, 1])

    plt.show()


## Model Evaluation Dashboard
def evaluation_dashboard(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    fig.suptitle( "Model Evaluation Dashboard", fontsize=18, fontweight="bold")

    plot_predictions(model, X_test, y_test, ax=axes[0])
    plot_residuals(model, X_test, y_test, ax=axes[1])

    plt.show()


## Optimizer Dashboard
def optimizer_dashboard(X_train, y_train, learning_rates=(0.0001, 0.001, 0.01, 0.1), epochs=200):
    """
    ### How to Read a Learning Rate Comparison Graph

    * Initial slope
        * A steeper drop means the model is learning faster.
        * A flatter curve means learning is slow.

    * Convergence speed
        * The curve that reaches a flat region in fewer epochs converges faster.
        * Fewer epochs = more efficient training.

    * Final loss
        * Lower final loss indicates a better fit.
        * If multiple learning rates reach the same loss, prefer the one that gets there faster.

    * Stability
        * A smooth, steadily decreasing curve indicates stable learning.
        * Zig-zag or oscillating curves suggest the learning rate is too high.

    * Divergence
        * If the loss increases or explodes instead of decreasing, gradient descent is diverging.
        * This usually means the learning rate is much too large.

    * Plateau
        * A nearly horizontal curve indicates the model has converged.
        * Additional epochs will provide little or no improvement.

    * Compare curves
        * Curves that stay consistently below others are learning more efficiently.
        * Lower curves generally indicate better optimization progress.

    * Choose the best learning rate
        * Decreases loss rapidly.
        * Converges in the fewest epochs.
        * Reaches the lowest loss.
        * Remains smooth and stable without oscillations or divergence.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle("Optimizer Dashboard", fontsize=18, fontweight="bold")

    plot_lr_comparison(X_train, y_train, learning_rates=learning_rates, epochs=epochs, ax=ax)
    
    plt.show()
      

if __name__ == "__main__":
    N_FEATURES = 7
    N_TARGETS = 1
    X, y = make_regression(n_samples=100000, n_features=N_FEATURES, n_informative=7, n_targets=N_TARGETS, noise=50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
    
    feature_names = [f"f{i + 1}" for i in range(N_FEATURES)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    
    model = GradientDescentRegressor(0.01, 250)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Predicted y: {y_pred}")    
    
    print(f"r2 score: {r2_score(y_test, y_pred)}")
    
    training_dashboard(model)
    evaluation_dashboard(model, X_test, y_test)
    optimizer_dashboard(X_train, y_train, learning_rates=(0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3), epochs=250)   