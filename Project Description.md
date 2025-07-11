# AI-Driven Analysis of Screen Time and Social Interaction to Detect Social Isolation

**“Detect Early. Connect Deeply. Prevent Isolation.”**

---

## Overview

This project proposes an **AI-driven solution** to identify **physical, digital, and emotional isolation** by analyzing behavioral data such as screen time, communication frequency, and social media engagement. By applying **Random Forest**, **Deep Neural Networks (DNNs)**, and other ML algorithms, the system detects early signs of social withdrawal.

The aim is to support **early intervention** through predictive analytics, helping individuals and mental health professionals understand and respond to patterns of isolation in digitally active populations.

---

## Key Features

* Tracks and analyzes **screen time**, **communication logs**, and **social activity**
* Predicts social isolation using trained **Random Forest classifiers**
* Displays evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**
* Generates **feature importance graphs**
* Includes **data preprocessing**, **feature engineering**, and **hyperparameter tuning**
* Ready for deployment with real-time input and REST API integration

---

## Problem Solved

* People often suffer from isolation without clear symptoms
* Manual surveys/self-reports can be biased or infrequent
* This system provides a **data-driven model** to detect social isolation patterns automatically and proactively

> *Example: A person with high screen time and low social interaction frequency may be flagged for early emotional isolation.*

---

## Tech Stack

| Component            | Technology                                                         |
| -------------------- | ------------------------------------------------------------------ |
| Programming Language | Python                                                             |
| ML Algorithms        | Random Forest, DNN, SVM (optional)                                 |
| Libraries            | scikit-learn, pandas, NumPy, matplotlib, seaborn, TensorFlow/Keras |
| Input Format         | Screen time logs, interaction frequency (CSV format)               |
| Interface            | Jupyter Notebook / Flask API (optional)                            |
| Model Deployment     | Pickle/Joblib for model serialization                              |

---

## Software Design Principles

* **SRP** – Clean modular files for model training, preprocessing, and visualization
* **DRY** – Reusable functions for data cleaning and evaluation
* **Explainability** – Feature importance graph to interpret model behavior
* **Extensibility** – Easily pluggable ML models (RF, SVM, DNN, etc.)

---

## How It Works

1. **Data Collection**

   * Weekly logs of screen time, message counts, and social activity

2. **Preprocessing**

   * Missing value handling, normalization, outlier detection

3. **Feature Engineering**

   * Extracts features like `screen_time`, `communication_frequency`, `social_interaction_score`, etc.

4. **Model Training**

   * Trained on labeled data using Random Forest
   * Hyperparameters tuned (e.g., `n_estimators`, `max_depth`)

5. **Model Evaluation**

   * Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC

6. **Feature Importance**

   * Visualized using bar plots to explain decision-making

7. **Deployment**

   * Model saved using Pickle or Joblib
   * Flask API accepts input and returns prediction label: *Isolated*, *At Risk*, *Engaged*

---

## Screenshots & Diagrams

* Confusion Matrix
* Feature Importance Plot
* ROC Curve
* System Architecture Diagram *(if available)*

---

## Run Locally

### For Jupyter Notebook

```bash
git clone https://github.com/yourusername/AI-Social-Isolation-Detection.git
cd AI-Social-Isolation-Detection
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/isolation_detection.ipynb` and run all cells.

### For Flask API (optional)

```bash
cd app/
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000) to access API
