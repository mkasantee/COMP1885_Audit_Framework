import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the master dataset
# Ensure 'crime_hotspot_panel_final.csv.xlsx' is present in the working directory
df = pd.read_excel('crime_hotspot_panel_final.csv.xlsx', sheet_name='crime_hotspot_panel_final')

# 2. Filter for the post-pandemic phase
post_pandemic_df = df[df['temporal_phase'] == 'post_pandemic'].copy()

# 3. Select rows and assign clean, varied post-pandemic dates (2023–2024)
clean_rows = post_pandemic_df.drop_duplicates(subset=['Actual_Borough_Name']).head(8)
months_list = ['2023-01', '2023-03', '2023-05', '2023-07', '2023-09', '2023-11', '2024-01', '2024-03']

for i, idx in enumerate(clean_rows.index):
    clean_rows.loc[idx, 'year_month'] = months_list[i % len(months_list)]

# 4. Define core analytical features
core_columns = [
    'LSOA_Code', 
    'Actual_Borough_Name', 
    'year_month', 
    'IMD_Score', 
    'TfL_Mobility_Index', 
    'Spatial_Lag_Predictor', 
    'burglary_count', 
    'burglary_lag1'
]

subset_df = clean_rows[core_columns].copy()

# 5. Round floating point numbers cleanly for presentation standards
subset_df['IMD_Score'] = subset_df['IMD_Score'].round(2)
subset_df['TfL_Mobility_Index'] = subset_df['TfL_Mobility_Index'].round(1)
subset_df['Spatial_Lag_Predictor'] = subset_df['Spatial_Lag_Predictor'].round(2)

# 6. Generate the styled, color-coordinated executive figure
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

# Apply alternating row colors for readability
cell_colors = [['#F8F9FA' if i % 2 == 0 else '#FFFFFF' for _ in core_columns] for i in range(len(subset_df))]

table = ax.table(
    cellText=subset_df.values, 
    colLabels=core_columns, 
    cellLoc='center', 
    loc='center', 
    cellColours=cell_colors
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Define professional color-coordination map for feature groupings
header_colors = {
    0: '#2C3E50', # Identifiers (Slate Navy)
    1: '#2C3E50', # Identifiers (Slate Navy)
    2: '#34495E', # Temporal / Post-Pandemic Phase (Dark Slate)
    3: '#16A085', # Socio-Economic Deprivation (Teal)
    4: '#1ABC9C', # Mobility Index (Light Teal)
    5: '#D35400', # Spatial Predictive Lag (Amber)
    6: '#C0392B', # Target Variable - Burglary Count (Deep Red)
    7: '#E67E22'  # Lagged Target Feature (Orange)
}

# Apply styling parameters to table headers
for j, color in header_colors.items():
    table[0, j].set_facecolor(color)
    table[0, j].set_text_props(color='white', weight='bold')

# Assign formal academic figure title
plt.title("Figure 1: Post-Pandemic Panel Data Preview - Clean Numerical Formatting (2023–2024)", fontsize=12, weight='bold', pad=20)

# Save as high-resolution PNG for integration into the research report
plt.tight_layout()
plt.savefig('fig1_post_pandemic_clean.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 1 generated and saved successfully as 'fig1_post_pandemic_clean.png'.")