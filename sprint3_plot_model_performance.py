import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_sprint3_plot() -> None:
    """
    Initialises the dataset ingestion pipeline, executes the model training sequence, 
    and generates a visual artefact comparing test predictions against actual target values.
    """
    # Initialise the random seed to guarantee reproducibility across system executions.
    np.random.seed(42)
    n_samples = 2000
    
    # Construct the raw spatial data dictionary matching the enterprise schema.
    raw_data = {
        "sector_id": range(1, n_samples + 1),
        "deprivation_index": np.random.uniform(0, 10, n_samples),
        "historical_incidents": np.random.poisson(3, n_samples),
        "patrol_frequency": np.random.uniform(1, 5, n_samples),
        "vulnerability_flag": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        "burglary_count": np.random.poisson(2, n_samples)
    }
    
    dataframe = pd.DataFrame(raw_data)
    
    # Engineer the risk density ratio feature for predictive modelling input.
    dataframe["risk_density_ratio"] = dataframe["historical_incidents"] / (dataframe["patrol_frequency"] + 1)
    
    feature_cols = ["deprivation_index", "historical_incidents", "patrol_frequency", "vulnerability_flag", "risk_density_ratio"]
    X = dataframe[feature_cols]
    y = dataframe["burglary_count"]
    
    # Partition the dataset into training and testing subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the random forest regressor model on the training partition.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions across the testing partition.
    y_pred = model.predict(X_test)
    
    # Configure visualisation parameters to align with enterprise reporting standards.
    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(y_test, y_pred, alpha=0.4, color="#2ca02c", s=20)
    
    # Plot the ideal reference line representing perfect model prediction capability.
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
    
    plt.title("Sprint 3: Model Evaluation (Actual vs. Predicted Test Split)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Actual Burglary Count", fontsize=10, labelpad=8)
    plt.ylabel("Model Predicted Count", fontsize=10, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    
    output_filename = "sprint3_model_performance.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Sprint 3 visual asset successfully generated and saved as '{output_filename}'.")

if __name__ == "__main__":
    generate_sprint3_plot()