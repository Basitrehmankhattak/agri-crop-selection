#  Predictive Modeling for Agricultural Crop Selection

##  Business Problem & Overview
In precision agriculture, selecting the optimal crop based on soil metrics is critical for maximizing yield. However, conducting complete soil testing across multiple metrics ($N, P, K, pH$) can be cost-prohibitive for smallholder farmers. 

This project solves this real-world constraint by identifying the **single most predictive soil metric** for crop recommendation using a multi-class Logistic Regression pipeline.

---

##  Tech Stack & Environment
* **Language:** Python 3.10+
* **Environment:** WSL2 / VS Code
* **Data Manipulation:** `pandas`
* **Machine Learning:** `scikit-learn` (`LogisticRegression`, `train_test_split`, `f1_score`)
* **Version Control:** Git & GitHub

---

## Engineering Methodology

1. **Data Ingestion & Preprocessing:**
   - Ingested soil measurements (`soil_measures.csv`) covering Nitrogen ($N$), Phosphorous ($P$), Potassium ($K$), and $pH$.
   - Split dataset into **80% Training** and **20% Testing** sets using stratified sampling (`stratify=y`) to maintain target class distributions across multi-class crops.

2. **Feature Evaluation Pipeline:**
   - Constructed an automated iteration loop to train individual multi-class Logistic Regression models (`multi_class='multinomial'`) on each soil feature independently.
   - Evaluated standalone predictive performance on hidden test data using the **Weighted $F_1$-Score** to account for multi-class classification balance.

3. **Optimization & Stability:**
   - Managed solver iterations (`max_iter=2000`) to guarantee model convergence across varying feature scales.

---

##  Results & Business Impact

| Soil Feature | Model Type | Evaluation Metric | Best Performer |
| :--- | :--- | :--- | :---: |
| **Nitrogen ($N$)** | Multinomial Logistic Regression | Weighted $F_1$-Score | |
| **Phosphorous ($P$)** | Multinomial Logistic Regression | Weighted $F_1$-Score | |
| **Potassium ($K$)** | Multinomial Logistic Regression | Weighted $F_1$-Score | ⭐ **Winner** |
| **$pH$** | Multinomial Logistic Regression | Weighted $F_1$-Score | |

> **Key Finding:** Potassium (`K`) yielded the highest individual weighted $F_1$-score, making it the most cost-effective single soil measurement for recommending crops under strict testing budget constraints.

---

##  How to Run Locally

```bash
# 1. Clone repository
git clone [https://github.com/Basitrehmankhattak/agri-crop-selection.git](https://github.com/Basitrehmankhattak/agri-crop-selection.git)
cd agri-crop-selection

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install pandas scikit-learn notebook

# 4. Launch Jupyter Notebook
jupyter notebook