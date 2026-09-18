# Credit Card Fraud Detection — Unsupervised Learning

An unsupervised machine learning project for detecting suspicious credit card transactions **without using the fraud label during training**.

The goal was to see whether anomaly detection and clustering methods could identify unusual transaction behavior, while keeping the actual `fraud` label only for evaluation.

## What I Used

* **Isolation Forest** — anomaly detection
* **Local Outlier Factor (LOF)** — local anomaly detection
* **K-Means** — transaction segmentation
* **DBSCAN** — density-based clustering
* **PCA** — dimensionality reduction
* **SHAP** — model explainability

## Dataset

* **1,000,000 transactions**
* **8 original features**
* **87,403 fraud cases (8.74%)**
* No missing values

Important features include:

`distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, and `online_order`.

The `fraud` column was **not used for training**. It was kept separately and used only after prediction to evaluate the models.

## Approach

The workflow was:

**EDA → Feature Engineering → Preprocessing → Anomaly Detection → Clustering → PCA → Explainability**

I applied `log1p` transformations to heavily right-skewed transaction features and standardized the numerical features before applying distance-based methods.

For anomaly detection, I compared Isolation Forest and LOF on a held-out evaluation split.

## Results

| Model            |    ROC-AUC | Avg Precision |
| ---------------- | ---------: | ------------: |
| Isolation Forest | **0.7787** |    **0.2209** |
| LOF              |     0.5775 |        0.1357 |

Isolation Forest performed better than LOF on this dataset.

At the selected contamination level of `0.02`, Isolation Forest flagged about **2% of the evaluation transactions**.

I also tested different contamination values to understand the trade-off between fraud recall and the number of alerts generated.

## Clustering

K-Means selected **4 clusters** based on silhouette score.

* Silhouette Score: **0.3000**
* Davies-Bouldin Index: **1.4207**

DBSCAN was also tested to identify dense transaction groups and noise points.

## Explainability

SHAP was used with Isolation Forest to inspect which features contributed to individual anomaly scores.

This helps move from simply saying:

> "This transaction is unusual."

to understanding **which transaction characteristics contributed to the model's decision**.

## Key Takeaway

This project helped me understand how unsupervised learning can be applied to fraud detection when the model does not directly learn from historical fraud labels.

The main lesson was that anomaly detection is not simply about finding "fraud." It is about finding unusual behavior and then evaluating whether those patterns are actually useful for the business problem.

## Tools

Python · Pandas · NumPy · Scikit-learn · SHAP · Category Encoders · Matplotlib · Seaborn · Joblib · Jupyter Notebook

## Project File

`fraud_detection_unsupervised_protfolio-checkpoint.ipynb`
