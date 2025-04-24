import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# START
# Loading the dataset
data = pd.read_excel("Priyanshi_CA2\crop dataset.xlsx")
df = data.sample(n=1000, random_state=5)

# Reset the index after sampling
df.reset_index(drop=True, inplace=True)

# ------------------------------------- Raw Data -------------------------------------
print("------ HEAD ------")
print(df.head())

print("\n------ DESCRIBE ------")
print(df.describe())

print("\n------ INFO ------")
print(df.info())
#no. of columns
print("\n------ COLUMNS ------")
print(df.columns.tolist())
#no of rows
print("\n------ SHAPE ------")
print("Number of Rows and Columns:", df.shape)

print("\n------ NULL VALUES ------")
print(df.isnull().sum())

print("\n------ DUPLICATE VALUES ------")
print("Total Duplicates:", df.duplicated().sum())

print("\n------ DATA TYPES ------")
print(df.dtypes)

# Explore unique values in key categorical columns
print("\n------ UNIQUE VALUES ------")

if 'Season' in df.columns:
    print("Unique Seasons:", df['Season'].unique())

if 'Crop' in df.columns:
    print("Unique Crops:", df['Crop'].unique())

if 'State_Name' in df.columns:
    print("Unique States:", df['State_Name'].unique())

if 'District_Name' in df.columns:
    print("Unique Districts:", df['District_Name'].unique())

if 'Year' in df.columns:
    print("Unique Years:", df['Year'].unique())

# --------------------------------Data Cleaning and preprocessing----------------------------------------

# Rename columns for simplicity
df.rename(columns={
    'State_Name': 'State',
    'District_Name': 'District',
    'Crop_Year': 'Year',  # use this only if your column is named 'Crop_Year'
    'Season': 'Season',
}, inplace=True)

# Check new column names
print("Renamed columns:", df.columns.tolist())

# Strip leading and trailing spaces from all text columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 1. Drop rows where all values are NaN
df.dropna(how='all', inplace=True)

# 2. Drop rows where 'Production' is 0
df = df[df['Production'] != 0]

# 3. Drop rows where 'Production' is NaN
df.dropna(subset=['Production'], inplace=True)

# 4. Reset index
df.reset_index(drop=True, inplace=True)

print("\n MISSING VALUES SUMMARY")
print(df.isnull().sum())

print("\n TOTAL ROWS:", len(df))
print(" COLUMNS WITH MISSING VALUES:")
print(df.columns[df.isnull().any()])

# --------------------------------------EDA(Exploratory Data Analysis)---------------------------------------

# 1. Basic Info & Structure
print(" First 5 Rows:")
print(df.head())

print("\n DataFrame Info:")
print(df.info())

print("\n Summary Statistics (Numerical):")
print(df.describe())

print("\nShape of the Dataset:")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

# 2. Missing & Duplicate Values
print("\n Missing Values Per Column:")
print(df.isnull().sum())

print("\n Total Duplicated Rows:")
print(df.duplicated().sum())

# 3. Data Types & Column Names
print("\n Data Types of Columns:")
print(df.dtypes)

print("\n Column Names:")
print(df.columns.tolist())

# 4. Categorical Columns – Unique Values
print("\n Unique Values in Key Categorical Columns:")

if 'Year' in df.columns:
    print("Years:", sorted(df['Year'].unique()))

if 'Season' in df.columns:
    print("Seasons:", df['Season'].unique())

if 'Crop' in df.columns:
    print("Crops:", df['Crop'].nunique(), "—", df['Crop'].unique()[:10])  # first 10 crops

if 'State' in df.columns:
    print("States:", df['State'].nunique())

if 'District' in df.columns:
    print("Districts:", df['District'].nunique())

# 5. Distribution of Area and Production
df = df.dropna(subset=['Area'])           # Remove missing values
df = df[df['Area'] < df['Area'].quantile(0.99)]  # Drop top 1% outliers

# Remove 0 and NaNs before plotting
cleaned_area = df['Area'].dropna()
cleaned_area = cleaned_area[cleaned_area > 0]

# Set figure size
plt.figure(figsize=(12, 6))

# Use better binning and zoom in
sns.histplot(cleaned_area, bins=np.arange(0, 50000, 1000), color='skyblue', kde=True)

# Optional: Mean and Median lines
plt.axvline(cleaned_area.mean(), color='red', linestyle='--', label='Mean')
plt.axvline(cleaned_area.median(), color='green', linestyle='--', label='Median')
plt.legend()

# Axis labels and title
plt.title('Distribution of Area (Cleaned & Zoomed)', fontsize=14, fontweight='bold')
plt.xlabel('Area', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0, 50000)

plt.tight_layout()
plt.show()
#plt show
# Log transformation of Area
df_cleaned = df[df['Area'] > 0].copy()
df_cleaned['log_area'] = np.log1p(df_cleaned['Area'])  # log(Area + 1) to avoid log(0)

# Plotting
plt.figure(figsize=(12, 6))
sns.histplot(df_cleaned['log_area'], bins=50, kde=True, color='lightcoral', edgecolor='black')

# Add mean and median lines
log_mean = df_cleaned['log_area'].mean()
log_median = df_cleaned['log_area'].median()

plt.axvline(log_mean, color='red', linestyle='--', linewidth=1.5, label='Mean')
plt.axvline(log_median, color='green', linestyle='--', linewidth=1.5, label='Median')

# Titles and labels
plt.title('Log-Transformed Distribution of Area', fontsize=14, fontweight='bold')
plt.xlabel('log(Area + 1)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# -----------------------Objective 1-----------------------------------
# Year-wise Crop Production Trend Analysis:
# To understand how production has changed over the years for different crops.

# Group the data by Year and Crop, summing up Production
yearly_crop_prod = df.groupby(['Year', 'Crop'])['Production'].sum().reset_index()

# Optionally: check top 5 crops based on total production
top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(5).index.tolist()

# Filter the data for just those top crops
filtered_data = yearly_crop_prod[yearly_crop_prod['Crop'].isin(top_crops)]

# Set up the plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot each crop trend
for crop in top_crops:
    crop_data = filtered_data[filtered_data['Crop'] == crop]
    plt.plot(crop_data['Year'], crop_data['Production'], marker='o', label=crop)

# Customize plot
plt.yscale('log')

plt.title(" Year-wise Production Trend of Top 5 Crops", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.legend(title='Crop')
# Set x-ticks at 3-year intervals
plt.xticks(ticks=range(filtered_data['Year'].min(), filtered_data['Year'].max() + 1, 3))

plt.tight_layout()
plt.show()

# -------------------------Objective 2-------------------------------------------------
# Crop Yield Calculation and Comparison:
# To measure the efficiency of crop production in terms of land used.

# Calculate yield first
df['Yield'] = df['Production'] / df['Area']

# Get top 10 crops by total production
top_crops = df.groupby('Crop')['Production'].sum().nlargest(10).index

# Filter the dataset for those top crops
filtered_df = df[df['Crop'].isin(top_crops)]

# Calculate average yield per top crop
avg_yield = filtered_df.groupby('Crop')['Yield'].mean().reset_index()
avg_yield = avg_yield.sort_values(by='Yield', ascending=False)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=avg_yield, x='Crop', y='Yield', palette='crest')
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.title('Average Yield of Top 10 Crops by Production')
plt.ylabel('Yield (tons per hectare)')
plt.xlabel('Crop')
plt.tight_layout()
plt.show()

# --------------------------------------------------------Objective 3---------------------------------------------------------------
# Outlier Detection in Production and Yield:
# To identify and analyze unusually high or low values in production and yield.

top_crops = df.groupby('Crop')['Production'].sum().nlargest(10).index
df_top = df[df['Crop'].isin(top_crops)]

df_top['Yield'] = df_top['Production'] / df_top['Area']

def detect_outliers_iqr(column):
    Q1 = df_top[column].quantile(0.25)
    Q3 = df_top[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_top[(df_top[column] < (Q1 - 1.5 * IQR)) | (df_top[column] > (Q3 + 1.5 * IQR))]
    return outliers

outliers_prod = detect_outliers_iqr('Production')
outliers_yield = detect_outliers_iqr('Yield')
# fig
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='Crop', y='Production')
plt.yscale('log')  # Log scale for better visibility
plt.title(' Outlier Detection: Production (Log Scale)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_top, x='Year', y='Production', hue='Crop')
plt.yscale('log')  # To handle large value range
plt.title('Year-wise Production with Outliers (Log Scale)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='Crop', y='Yield')
plt.yscale('log')  # Log scale for better visibility
plt.title('Outlier Detection: Yield (Log Scale)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#fig
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_top, x='Year', y='Yield', hue='Crop')
plt.yscale('log')  # Also use log for yield if range is large
plt.title('Year-wise Yield with Outliers (Log Scale)')
plt.tight_layout()
plt.show()

# --------------------------------------------------objective 4----------------------------------------------------------------
# Correlation Analysis Between Area and Production:
# To study the relationship between land area used and total crop production.

correlation = df[['Area', 'Production']].corr()
print(correlation)

# Remove rows with zero or negative values (log scale doesn't support them)
df_filtered = df[(df['Area'] > 0) & (df['Production'] > 0)]

# Calculate Pearson correlation
corr, _ = pearsonr(df_filtered['Area'], df_filtered['Production'])

# Create the plot
plt.figure(figsize=(10, 6))
sns.regplot(
    x='Area', 
    y='Production', 
    data=df_filtered,
    scatter_kws={'alpha':0.3},
    line_kws={'color':'blue'},
    logx=True
)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Area (Hectares)')
plt.ylabel('Production (Tonnes)')
plt.title(f'Correlation Between Area and Production (Log Scale)\nPearson Correlation: {corr:.2f}')
plt.tight_layout()
plt.show()

# -----------------------------------------Objective 5-------------------------------------------------------
# Season-wise Production Distribution:
# To analyze how production varies by agricultural season.

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Season', y='Production', palette='Set2')

plt.yscale('log')  # Use log scale to handle extreme values
plt.title('Season-wise Production Distribution')
plt.xlabel('Season')
plt.ylabel('Production (Tonnes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------Objective 6---------------------------------------------
# Compare Average Production per District Over Years:
# To analyze each district across all years to spot high- and low-performing areas.

# Step 1: Group by District and calculate average production
district_avg = df.groupby('District')['Production'].mean().sort_values(ascending=False)

# Step 2: Select top 20 districts
top20_districts = district_avg.head(20)

# Step 3: Plot with log scale
plt.figure(figsize=(10, 8))
top20_districts.plot(kind='barh', color='lightcoral')

plt.xlabel('Average Production (Tonnes) [Log Scale]', fontsize=12)
plt.ylabel('District')
plt.title('Top 20 Districts by Average Production (Log Scale, Across All Years)', fontsize=14)
plt.xscale('log')  # Log scale
plt.gca().invert_yaxis()  # Highest at the top
plt.tight_layout()
plt.show()

# Heatmap of Top 10 Crop Production Across States
# Step 1: Aggregate total production by crop
top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10).index

# Step 2: Filter data for top 10 crops
df_top = df[df['Crop'].isin(top_crops)]

# Step 3: Group by Crop and State to get total production
pivot_df = df_top.groupby(['Crop', 'State'])['Production'].sum().reset_index()

# Step 4: Create pivot table
heatmap_data = pivot_df.pivot(index='Crop', columns='State', values='Production')

# Step 5: Replace zeros with NaN to avoid log issues , then apply log10
heatmap_data = heatmap_data.replace(0, np.nan)
heatmap_log = np.log10(heatmap_data)

# Step 6: Plot heatmap
plt.figure(figsize=(26,10))
sns.heatmap(
    heatmap_log,
    cmap="YlGnBu",
    linewidths=0.4,
    linecolor='gray',
    annot=True,           # Show values inside the cells
    fmt=".1f",            # Format for annotation text
    annot_kws={"size": 8} # Font size for annotations
)

plt.title('Heatmap of Top 10 Crop Production Across States (Log Scale)', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Crop', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Pie chart of top 5 crops by production share
# Step 1: Group by Crop and sum production
crop_prod = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(5)

# Step 2: Plot pie chart
plt.figure(figsize=(8, 8))
colors = plt.cm.Set3.colors  # Optional: use a nice color palette
plt.pie(
    crop_prod,
    labels=crop_prod.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)
plt.title('Top 5 Crops by Total Production Share')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
plt.tight_layout()
plt.show()
# end
#git
