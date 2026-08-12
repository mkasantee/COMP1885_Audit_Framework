import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

def run_fairness_audit() -> Tuple[float, float, float]:
    """
    Computes statistical parity disparity and evaluates predictive bias 
    between vulnerable and standard municipal urban sectors.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the mean prediction for standard sectors, 
        the mean prediction for vulnerable sectors, and the computed disparity.
    """
    # Seed initialisation guarantees deterministic reproducibility across audit iterations.
    np.random.seed(42)
    n_samples = 2000
    
    X = np.random.randn(n_samples, 5)
    
    # Vulnerable sector flags are simulated to represent protected or socioeconomically sensitive urban areas.
    vulnerable_sector = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    y = np.dot(X, np.random.randn(5)) * 2 + vulnerable_sector * 1.5 + np.random.normal(0, 1, n_samples)
    y = np.clip(y, 0, 20)
    
    # An ensemble regressor is instantiated to evaluate algorithmic scoring trends across protected groups.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Predictions are segmented by vulnerability flags to measure statistical parity disparities.
    standard_preds = predictions[vulnerable_sector == 0]
    vulnerable_preds = predictions[vulnerable_sector == 1]
    
    std_mean = float(np.mean(standard_preds))
    vuln_mean = float(np.mean(vulnerable_preds))
    parity_disparity = float(abs(std_mean - vuln_mean))
    
    return std_mean, vuln_mean, parity_disparity

if __name__ == "__main__":
    std_mean, vuln_mean, parity_disparity = run_fairness_audit()
    print("--- Sprint 4: Fairness and Demographic Parity Audit Results ---")
    print(f"Standard Sector Mean Prediction: {std_mean:.4f}")
    print(f"Vulnerable Sector Mean Prediction: {vuln_mean:.4f}")
    print(f"Statistical Parity Disparity: {parity_disparity:.4f}")