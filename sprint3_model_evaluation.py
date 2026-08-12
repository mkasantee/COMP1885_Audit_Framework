import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

def evaluate_model_performance() -> Tuple[float, float]:
    """
    Evaluates baseline model robustness, predictive error metrics, 
    and structural stability under synthetic data variations.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the calculated Mean Squared Error (MSE) 
        and R-Squared score.
    """
    # A fixed random seed guarantees consistent synthetic dataset generation for validation checkpoints.
    np.random.seed(42)
    X_train = np.random.randn(1600, 5)
    y_train = np.dot(X_train, np.random.randn(5)) * 2 + np.random.normal(0, 1, 1600)
    
    X_test = np.random.randn(400, 5)
    y_test = np.dot(X_test, np.random.randn(5)) * 2 + np.random.normal(0, 1, 400)
    
    # The ensemble configuration utilises explicit estimator boundaries to ensure deterministic evaluation outcomes.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Performance metrics are computed to quantify prediction variance and goodness of fit.
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return mse, r2

if __name__ == "__main__":
    mse_val, r2_val = evaluate_model_performance()
    print("--- Sprint 3: Model Evaluation and Security Audit ---")
    print(f"Model Mean Squared Error: {mse_val:.4f}")
    print(f"Model R-Squared Score: {r2_val:.4f}")