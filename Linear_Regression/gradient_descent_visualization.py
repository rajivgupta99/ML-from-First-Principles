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


def compute_loss_surface(X_train, y_train, w_range, b_range):
    """
    Computes the MSE loss surface for different values of weight and bias.
    Works only for single-feature linear regression.
    """

    X = X_train.ravel()

    W, B = np.meshgrid(w_range, b_range)

    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):

            y_pred = W[i, j] * X + B[i, j]

            Z[i, j] = np.mean((y_train - y_pred) ** 2)

    return W, B, Z


## Contour Plot + Gradient Descent Path
## 3D Loss Surface + Gradient Descent Trajectory 
def optimization_dashboard(model, X_train, y_train):

    # Parameter history
    w_history = np.array(model.weight_history).flatten()
    b_history = np.array(model.bias_history)

    # Dynamic plotting range
    padding_w = 0.2 * (w_history.max() - w_history.min())
    padding_b = 0.2 * (b_history.max() - b_history.min())

    w_range = np.linspace(
        w_history.min() - padding_w,
        w_history.max() + padding_w,
        200
    )

    b_range = np.linspace(
        b_history.min() - padding_b,
        b_history.max() + padding_b,
        200
    )

    # Compute loss surface
    W, B, Z = compute_loss_surface(
        X_train,
        y_train,
        w_range,
        b_range
    )

    # Loss along optimization path
    X = X_train.ravel()

    losses = []

    for w, b in zip(w_history, b_history):
        y_pred = w * X + b
        losses.append(np.mean((y_train - y_pred) ** 2))

    # ================= Figure =================

    fig = plt.figure(figsize=(18, 7))

    fig.suptitle(
        "Gradient Descent Optimization",
        fontsize=18,
        fontweight="bold"
    )

    # ==========================================================
    # Contour Plot + Gradient Descent Path
    # ==========================================================

    ax1 = fig.add_subplot(1, 2, 1)

    contour = ax1.contour(
        W,
        B,
        Z,
        levels=30,
        cmap="viridis"
    )

    ax1.plot(
        w_history,
        b_history,
        "-o",
        color="red",
        linewidth=2,
        markersize=3,
        label="Gradient Descent Path"
    )

    ax1.scatter(
        w_history[0],
        b_history[0],
        color="lime",
        edgecolor="black",
        s=120,
        label="Start"
    )

    ax1.scatter(
        w_history[-1],
        b_history[-1],
        color="red",
        edgecolor="black",
        s=120,
        label="End"
    )

    ax1.set_title("Loss Contour with Optimization Path")
    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Bias")

    ax1.grid(alpha=0.3)
    ax1.legend()

    fig.colorbar(
        contour,
        ax=ax1,
        shrink=0.8,
        pad=0.02,
        label="MSE Loss"
    )

    # ==========================================================
    # 3D Loss Surface
    # ==========================================================

    ax2 = fig.add_subplot(
        1,
        2,
        2,
        projection="3d"
    )

    ax2.plot_surface(
        W,
        B,
        Z,
        cmap="viridis",
        alpha=0.8,
        edgecolor="none"
    )

    ax2.plot(
        w_history,
        b_history,
        losses,
        color="red",
        linewidth=3,
        marker="o",
        markersize=3
    )

    ax2.scatter(
        w_history[0],
        b_history[0],
        losses[0],
        color="lime",
        s=80
    )

    ax2.scatter(
        w_history[-1],
        b_history[-1],
        losses[-1],
        color="red",
        s=80
    )

    ax2.view_init(elev=30, azim=-60)

    ax2.set_title("3D Loss Surface")
    ax2.set_xlabel("Weight")
    ax2.set_ylabel("Bias")
    ax2.set_zlabel("MSE Loss")

    plt.tight_layout()

    plt.show()
      

if __name__ == "__main__":
    N_FEATURES = 1
    N_TARGETS = 1
    X, y = make_regression(n_samples=1000, n_features=N_FEATURES, n_informative=1, n_targets=N_TARGETS, noise=50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
    
    feature_names = [f"f{i + 1}" for i in range(N_FEATURES)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y
    
    model = GradientDescentRegressor(0.01, 250)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Predicted y: {y_pred}")    
    
    print(f"r2 score: {r2_score(y_test, y_pred)}")
    
    optimization_dashboard(model, X_train, y_train)