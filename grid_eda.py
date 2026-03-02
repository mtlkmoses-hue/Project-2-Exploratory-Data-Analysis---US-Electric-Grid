import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load and Clean Data
df = pd.read_csv('US_Electric_Grid_new.csv')
for col in ['Demand Loss (MW)', 'Number of Customers Affected']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# Set the visual style
sns.set_theme(style="whitegrid")

# --- GENERATING 10 VISUALS ---

# Visual 1: Count of Events per NERC Region (Bar Chart)
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='NERC Region', order=df['NERC Region'].value_counts().index, palette='viridis')
plt.title('1. Frequency of Grid Events by NERC Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visual_1_region_freq.png')

# Visual 2: Total Demand Loss by Month (Line Chart)
plt.figure(figsize=(10,6))
df.groupby('Event Month')['Demand Loss (MW)'].sum().reindex(['January', 'February', 'March', 'April', 'May']).plot(kind='line', marker='o', color='red')
plt.title('2. Total Demand Loss (MW) Trends by Month')
plt.savefig('visual_2_monthly_loss.png')

# Visual 3: Missing Data Heatmap (Data Quality)
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('3. Missing Data Overview (Yellow = Missing)')
plt.savefig('visual_3_data_quality.png')

# Visual 4: Boxplot of Demand Loss (Identifying Outliers)
plt.figure(figsize=(10,6))
sns.boxplot(x=df['Demand Loss (MW)'], color='orange')
plt.title('4. Distribution and Outliers of Demand Loss')
plt.savefig('visual_4_outliers.png')

# Visual 5: Customers Affected vs Demand Loss (Scatter Plot)
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Number of Customers Affected', y='Demand Loss (MW)', hue='NERC Region', alpha=0.7)
plt.title('5. Correlation: Customers Affected vs. MW Loss')
plt.savefig('visual_5_correlation.png')

# Visual 6: Top 10 Most Affected Areas (Horizontal Bar)
plt.figure(figsize=(10,8))
df['Area Affected'].value_counts().head(10).plot(kind='barh', color='teal')
plt.title('6. Top 10 Areas with Most Reported Events')
plt.gca().invert_yaxis()
plt.savefig('visual_6_top_areas.png')

# Visual 7: Average Customers Affected per Region (Bar Chart)
plt.figure(figsize=(12,6))
df.groupby('NERC Region')['Number of Customers Affected'].mean().sort_values().plot(kind='bar', color='purple')
plt.title('7. Average Number of Customers Impacted per Region')
plt.savefig('visual_7_avg_impact.png')

# Visual 8: Event Counts by Month (Pie Chart)
plt.figure(figsize=(8,8))
df['Event Month'].value_counts().plot(kind='pie', autopct='%1.1f%%', cmap='Pastel1')
plt.title('8. Percentage Distribution of Events by Month')
plt.ylabel('')
plt.savefig('visual_8_month_dist.png')

# Visual 9: Distribution of Demand Loss (Histogram)
plt.figure(figsize=(10,6))
sns.histplot(df['Demand Loss (MW)'].dropna(), bins=20, kde=True, color='blue')
plt.title('9. Density of Event Sizes (MW)')
plt.savefig('visual_9_histogram.png')

# Visual 10: Regression Plot (Trend Line)
plt.figure(figsize=(10,6))
sns.regplot(data=df, x='Number of Customers Affected', y='Demand Loss (MW)', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('10. Linear Trend: Customer Base vs Grid Load Loss')
plt.savefig('visual_10_regression.png')

print("All 10 Visuals generated and saved as .png files!")
