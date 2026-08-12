import numpy as np
import matplotlib.pyplot as plt

def generate_sprint4_plot() -> None:
    """
    Visualises the fairness and demographic parity audit results, comparing 
    mean predicted risk scores across standard and vulnerable municipal sectors.
    """
    # Define categorical labels and empirical mean prediction results from the audit pipeline.
    categories = ['Standard Sector', 'Vulnerable Sector']
    mean_scores = [2.0963, 2.3215]
    bar_colors = ['#1f77b4', '#d62728']  # Enterprise colour palette configuration.
    
    # Configure enterprise visualisation parameters for executive reporting.
    plt.figure(figsize=(8, 5), dpi=300)
    
    # Utilise standard 'color' parameter nomenclature required by the matplotlib API.
    bars = plt.bar(categories, mean_scores, color=bar_colors, alpha=0.85, edgecolor='black', width=0.5)
    
    # Apply layout configurations, titles, and grid formatting.
    plt.title("Sprint 4: Demographic Parity Audit (Risk Score Disparity)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Mean Predicted Risk Score", fontsize=10, labelpad=8)
    plt.ylim(0, 3.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate exact empirical values above each column for stakeholder review.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Embed the statistical parity disparity metric into the visualisation canvas.
    plt.text(0.5, 2.75, "Statistical Parity Disparity: 0.2252", 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
    
    plt.tight_layout()
    
    output_filename = "sprint4_fairness_parity.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Sprint 4 fairness audit visual artefact successfully generated and saved as '{output_filename}'.")

if __name__ == "__main__":
    generate_sprint4_plot()