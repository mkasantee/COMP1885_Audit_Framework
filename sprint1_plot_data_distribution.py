import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sprint1_plot() -> None:
    """
    Generates and saves a publication-quality distribution plot 
    illustrating the raw deprivation index from Sprint 1 data with differentiated bar colors.
    """
    # Seed initialization matches Sprint 1 to ensure identical data distribution
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
    
    dataframe = pd.DataFrame(raw_data)
    
    # Figure styling configured to meet academic report standards
    plt.figure(figsize=(8, 5), dpi=300)
    n, bins, patches = plt.hist(dataframe["deprivation_index"], bins=30, edgecolor="black", alpha=0.85)
    
    # Apply distinct colors to each bar using a colormap
    cmap = plt.colormaps['viridis']
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(i / len(patches)))
    
    plt.title("Sprint 1: Urban Sector Deprivation Index Distribution", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Socioeconomic Deprivation Index", fontsize=10, labelpad=8)
    plt.ylabel("Frequency (Sector Count)", fontsize=10, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    output_filename = "sprint1_deprivation_distribution.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Sprint 1 visual asset successfully generated and saved as '{output_filename}'.")

if __name__ == "__main__":
    generate_sprint1_plot()