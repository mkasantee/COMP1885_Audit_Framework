import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def engineer_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Engineers interaction terms and partitions the dataset 
    into training and testing subsets for spatial modelling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Partitioned feature matrices and target arrays.
    """
    # Seed initialisation guarantees deterministic reproducibility across feature transformations.
    np.random.seed(42)
    n_samples = 2000
    
    raw_data = {
        "sector_id": range(1, n_samples + 1),
        "deprivation_index": np.random.uniform(0, 10, n_samples),
        "historical_incidents": np.random.poisson(3, n_samples),
        "patrol_frequency": np.random.uniform(1, 5, n_samples),
        "vulnerability_flag": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        "burglary_count": np.random.poisson(2, n_samples)
    }
    
    df = pd.DataFrame(raw_data)
    
    # Interaction terms are engineered to capture non-linear dynamics between crime history and patrol allocations.
    df["risk_density_ratio"] = df["historical_incidents"] / (df["patrol_frequency"] + 1)
    
    features = df[["deprivation_index", "historical_incidents", "patrol_frequency", "vulnerability_flag", "risk_density_ratio"]]
    target = df["burglary_count"]
    
    return train_test_split(features, target, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = engineer_features()
    print("Sprint 2: Feature engineering completed and data partitioned successfully.")
    print(f"Training feature shape: {X_train.shape}")
    print(f"Testing feature shape: {X_test.shape}")