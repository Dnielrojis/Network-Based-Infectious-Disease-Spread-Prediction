# Network-Based Infectious Disease Spread Prediction
### 3MTT/DeepTech Capstone Project - Data Science Track

## 📌 Project Overview

### Pipeline
![Project Pipeline Visualization](https://github.com/Dnielrojis/Network-Based-Infectious-Disease-Spread-Prediction/blob/main/Images/Disease%20spread%20flowchart.jpg)

This project focuses on predicting which individuals in a simulated population are likely to be infected during a disease outbreak. Unlike standard classification tasks, the risk of infection here is determined by a **Core-Periphery network structure**, where social connections and proximity to "Patient Zero" are the primary drivers of spread.

### The Problem
Traditional demographic data (Age, Constitution) often fails to capture the structural risks of an outbreak. The challenge is to identify high-risk individuals within 130 different simulated populations, each containing 5,000 individuals with unique behavioral traits.

### The Solution
We developed a hybrid machine learning pipeline that transforms raw connection data into **Mathematical Graph Objects**. By calculating network-specific metrics like "Distance to Index Patient" and "Degree Centrality," we provided our model with the structural context needed to predict infection paths.

---

## 🛠️ Technical Stack & Tools
- **Language:** Python 3.x
- **Graph Theory:** `NetworkX` (Used to build and analyze population networks)
- **Data Manipulation:** `Pandas`, `NumPy`
- **Parsing:** `ast.literal_eval` (Safe evaluation of stringified lists)
- **Machine Learning:** `LightGBM` (Gradient Boosting Machine)
- **Validation:** `GroupKFold` (Cross-validation respecting population boundaries)

---

## 🏗️ Project Architecture

### 1. Data Transformation (Graph Construction)
- Raw connection strings were parsed into Python lists.
- For each of the 130 populations, an undirected graph was initialized where **Nodes** represent people and **Edges** represent social connections.

### 2. Feature Engineering (Graph Metrics)
To improve model performance, we engineered the following features:
- **dist_to_index:** The shortest path distance from a node to the Index Patient.
- **degree_centrality:** A measure of how many direct connections an individual has (identifying potential super-spreaders).
- **clustering_coeff:** Measures the degree to which nodes in a graph tend to cluster together.

### 3. Model Training Pipeline
- **Validation Strategy:** We used **GroupKFold** to ensure that individuals from the same simulation were never split between training and validation, preventing data leakage.
- **Algorithm:** LightGBM was chosen for its high efficiency and ability to handle categorical behavioral data.
- **Optimization:** We implemented **Early Stopping** to prevent overfitting and removed non-predictive IDs to improve generalization.

---

## 📊 Results & Findings
- **Metric:** ROC AUC (Area Under the Curve).
- **Current Performance:** Achieved an **Overall Cross-Validation AUC of 0.5989**.
- **Key Insight:** Distance to the Index Patient emerged as a significant predictor of infection risk. The core-periphery structure of the network means that individuals in the dense "core" are at substantially higher risk than those on the periphery.

---

## 🚀 How to Run
1. Ensure you have the datasets in the `../Datasets/` directory.
2. Install dependencies:
   ```bash
   pip install pandas networkx lightgbm scikit-learn matplotlib
