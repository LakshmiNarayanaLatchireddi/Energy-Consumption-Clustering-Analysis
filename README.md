# Energy Consumption Analysis using PCA, Clustering, and Statistical Modeling

## Project Overview
This project analyzes household electricity consumption using the **Residential Energy Consumption Survey (RECS 2020)** dataset. The objective is to identify distinct household energy consumption patterns and validate them using **unsupervised learning, statistical testing, regression analysis, and machine learning classification**.

All analyses described here are fully implemented and reproducible using the provided notebook.

## Objectives
- Analyze residential electricity consumption behavior
- Reduce high-dimensional household features using Principal Component Analysis (PCA)
- Identify meaningful household energy consumption clusters
- Compare multiple clustering algorithms
- Statistically validate differences between clusters
- Model electricity consumption using regression techniques
- Evaluate machine learning models for predicting cluster membership

## Dataset
- **Source**: U.S. Energy Information Administration (EIA)
- **Dataset**: Residential Energy Consumption Survey (RECS) 2020
- **Primary Variable**: Annual electricity consumption (`KWH`)
- **Transformed Variable**: `log_KWH` (used for regression and statistical analysis)

⚠️ The raw dataset is not included in this repository due to size and licensing constraints.

Dataset link:  
https://www.eia.gov/consumption/residential/data/2020/

## Methodology

### 1. Data Preparation and Exploration
- Selection of numeric household, demographic, and climate-related variables
- Removal of invalid or negative values
- Missing value handling using median imputation
- Feature scaling using standardization
- Exploratory analysis of electricity consumption distributions

### 2. Dimensionality Reduction
- Principal Component Analysis (PCA) applied to standardized features
- Six principal components retained based on explained variance
- PCA features used for both clustering and regression analysis

### 3. Clustering Analysis
Clustering was performed in PCA space to reduce noise and multicollinearity.

**Clustering methods implemented:**
- K-Means (k-means++ initialization)
- K-Medoids
- Agglomerative Clustering (Ward linkage)
- Gaussian Mixture Model (GMM)

**Cluster validation metrics:**
- Silhouette Score
- Calinski–Harabasz Index
- Davies–Bouldin Index

The comparison showed that K-Means achieved the best overall performance, while K-Medoids was used to evaluate robustness to outliers.

### 4. Statistical Validation of Clusters
- One-way ANOVA conducted to test differences in mean `log_KWH` across clusters
- Tukey’s HSD used for pairwise comparison of cluster means
- Results confirmed statistically significant differences between household energy consumption clusters

### 5. Regression Analysis

#### Ordinary Least Squares (OLS)
- OLS regression performed using `log_KWH` as the dependent variable
- Household and climate predictors included
- Multicollinearity assessed using Variance Inflation Factor (VIF)
- Residual diagnostics performed to assess model assumptions

#### PCA-Based Regression (PCR)
- Regression performed using principal components as predictors
- Model evaluated using:
  - R²
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

PCR provided stable predictive performance with reduced dimensionality.


### 6. Machine Learning Classification
- Cluster labels treated as categorical targets
- Models trained to predict cluster membership

**Models implemented:**
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)

**Evaluation metrics:**
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Matthews Correlation Coefficient (MCC)

The best-performing model was selected based on Macro F1-score, followed by a confusion matrix for interpretation.
