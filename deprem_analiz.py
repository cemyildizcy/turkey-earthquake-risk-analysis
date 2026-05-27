# -*- coding: utf-8 -*-
"""
=============================================================================
    TURKEY EARTHQUAKE RISK ANALYSIS & PREDICTIVE MODELING
    
    Description: End-to-end Machine Learning pipeline for analyzing 30 years 
                 of earthquake data in Turkey. Includes exploratory data 
                 analysis (EDA), K-Means clustering for regional seismicity, 
                 and classification models (Random Forest, SVM, LR) to predict 
                 earthquake risk severity based on spatiotemporal features.
    
    Data Source: USGS Earthquake API (1994-2025, Magnitude >= 3.0)
    Author: Cem Yıldız
    Date: 2026
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    silhouette_score, 
    ConfusionMatrixDisplay
)
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

def extract_region(place_str):
    """Extracts region/city name from USGS place string."""
    if pd.isna(place_str):
        return "Unknown"
    if " of " in str(place_str):
        region = str(place_str).split(" of ")[-1]
        return region.replace(", Turkey", "").strip()
    return str(place_str)

def categorize_risk(magnitude):
    """Categorizes earthquake magnitude into risk levels."""
    if magnitude < 4.0: return 'Low'
    elif magnitude < 5.0: return 'Medium'
    else: return 'High'

def main():
    print("=" * 60)
    print("   TURKEY EARTHQUAKE RISK ANALYSIS PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. DATA LOADING & PREPROCESSING
    # -------------------------------------------------------------------------
    print("\n[1/4] Data Loading & Preprocessing...")
    df = pd.read_csv("data/turkiye_depremler.csv")
    
    # Select and rename columns
    cols_to_keep = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'magType']
    df = df[cols_to_keep]
    df.columns = ['date', 'latitude', 'longitude', 'depth', 'magnitude', 'place', 'mag_type']
    
    # Datetime conversions and feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['depth'] >= 0) & (df['magnitude'] >= 3.0)].dropna(subset=['latitude', 'longitude', 'depth', 'magnitude'])
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    df['region'] = df['place'].apply(extract_region)
    df['risk_level'] = df['magnitude'].apply(categorize_risk)
    
    print(f"Data processed: {len(df)} records.")

    # -------------------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("\n[2/4] Generating Visualizations...")
    
    # Correlation Matrix
    numeric_cols = ['latitude', 'longitude', 'depth', 'magnitude', 'year', 'month', 'hour']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('gorseller/04_korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Geographical Scatter Plot
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                          c=df['magnitude'], cmap='hot_r', alpha=0.5, 
                          s=df['magnitude']**2)
    plt.colorbar(scatter, label='Magnitude (Richter)')
    plt.title('Earthquake Distribution in Turkey (1994-2025)')
    plt.xlim(25.5, 45.5); plt.ylim(35.5, 42.5)
    plt.grid(True, alpha=0.3)
    plt.savefig('gorseller/05_deprem_haritasi.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 3. UNSUPERVISED LEARNING: K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    print("\n[3/4] Performing K-Means Clustering...")
    cluster_features = ['latitude', 'longitude', 'depth', 'magnitude']
    X_cluster = df[cluster_features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Optimal K=4 based on Elbow Method & Silhouette analysis
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f"Clustering complete. Silhouette Score: {sil_score:.4f}")

    # -------------------------------------------------------------------------
    # 4. SUPERVISED LEARNING: RISK CLASSIFICATION
    # -------------------------------------------------------------------------
    print("\n[4/4] Training Classification Models...")
    
    X = df[['latitude', 'longitude', 'depth', 'year', 'month']]
    y = df['risk_level']
    
    # Train/Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    cls_scaler = StandardScaler()
    X_train_scaled = cls_scaler.fit_transform(X_train)
    X_test_scaled = cls_scaler.transform(X_test)
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    # Model 3: Support Vector Machine (RBF)
    svm = SVC(kernel='rbf', random_state=42, C=1.0)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    
    # Evaluation Summary
    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM (RBF)'],
        'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)],
        'F1 Score (weighted)': [f1_score(y_test, y_pred_lr, average='weighted'), f1_score(y_test, y_pred_rf, average='weighted'), f1_score(y_test, y_pred_svm, average='weighted')]
    })
    print("\nModel Performance Comparison:")
    print(results.round(4).to_string(index=False))
    
    # Feature Importance (Random Forest)
    importances = rf.feature_importances_
    features = ['Latitude', 'Longitude', 'Depth', 'Year', 'Month']
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances[indices], color='#3498db', edgecolor='black')
    plt.yticks(range(len(features)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.savefig('gorseller/11_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPipeline execution finished successfully.")

if __name__ == "__main__":
    main()