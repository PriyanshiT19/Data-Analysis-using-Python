# 🌾 Crop Production Analysis with Python

This project performs an in-depth exploratory data analysis (EDA) on crop production across India. Using a cleaned dataset and powerful Python libraries, we explore trends, detect outliers, analyze correlations, and visualize agricultural performance across different regions and seasons.

---

## 📁 Project Structure

The analysis is structured around key objectives and follows a clean pipeline:

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Objective-wise Visual Insights**

---

## 📌 Key Objectives

### 1. 📈 Year-wise Crop Production Trend
- Analyze how production varies over time for top crops.
- Visualized using line plots with log-scaled y-axis.

### 2. 🌱 Crop Yield Analysis
- Yield is calculated as `Production / Area`.
- Compared across top 10 crops to assess efficiency.
- Log-scaled bar plot used for clarity.

### 3. ⚠️ Outlier Detection
- Outliers in `Production` and `Yield` identified using the IQR method.
- Boxplots and scatter plots with log scale used for visualization.

### 4. 🔁 Correlation Analysis
- Examines the relationship between `Area` and `Production`.
- Uses Pearson correlation and log-log regression plot.

### 5. 🕒 Season-wise Production Distribution
- Compares production across agricultural seasons.
- Boxplot used to reveal spread, median, and outliers.

### 6. 🗺️ District-wise Production
- Identifies top 20 districts by average production over time.
- Horizontal bar plot with log scaling used.

### 7. 🌍 Heatmap of Crop-State Production
- Heatmap showing how the top 10 crops are produced across states.
- Values are log-transformed and annotated for precision.

### 8. 🥧 Crop Share Pie Chart
- Visualizes the top 5 crops by total production share using a pie chart.

---

## 📊 Tools & Libraries Used

- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib` – Core plotting
- `seaborn` – Enhanced statistical plotting
- `scipy.stats` – Pearson correlation
- `openpyxl` – Excel file support (if applicable)

---

## 🧹 Data Preprocessing Highlights

- Removed rows with missing or zero production.
- Standardized column names and stripped whitespace.
- Handled outliers with IQR method.
- Applied log scales where appropriate for better visual interpretation.

---

## 🖼️ Sample Visualizations

- Line plots of crop trends
- Log-scaled boxplots for outliers
- Heatmaps showing state-wise crop dominance
- Pie charts summarizing production share


