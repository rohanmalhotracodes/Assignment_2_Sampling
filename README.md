# Assignment 2 — Sampling

**Name:** Rohan Malhotra  
**Roll No:** 102303437

---

## Repository Structure
- `Assignment_2_Sampling.ipynb` — **Colab notebook**
- `Creditcard_data.csv` — dataset
- `src` — python modules (sampling + models + runner)
- `results` — outputs (table + graphs)
- `DISCUSSION.md` — short findings
- `requirements.txt` — dependencies

---

## Methodology (Detailed)

### 1) Problem statement
We need to compare **five sampling techniques** on an **imbalanced credit-card fraud dataset**, train **five ML models** on each sampled dataset (n=500), and report the **accuracy matrix** + graphs.

### 2) Dataset
`Creditcard_data.csv` has multiple features and a target column `Class`:
- `Class = 0` → non-fraud
- `Class = 1` → fraud

### 3) Imbalance handling — SMOTE
The original dataset is highly imbalanced. We apply **SMOTE** to generate synthetic minority (fraud) examples and obtain a balanced dataset (~50:50).

SMOTE parameters:
- `k_neighbors = 3`
- `random_state = 42`

### 4) Train/Test split
After SMOTE, we do an 80/20 **stratified** split:
- Train split is used for sampling (n=500 each)
- Test split is fixed and used for evaluating all models

### 5) Sampling techniques (n=500 from TRAIN only)
1. **Sampling1 — Simple Random Sampling (SRS)**: uniform random selection without replacement.  
2. **Sampling2 — Stratified Sampling**: selects equal rows from each class (train is balanced after SMOTE).  
3. **Sampling3 — Systematic Sampling**: shuffle, then pick every *k*-th row.  
4. **Sampling4 — Cluster Sampling (KMeans-based)**: cluster standardized train features and select clusters until 500 rows.  
5. **Sampling5 — Bootstrap Sampling**: sample with replacement.  

### 6) ML models (5)
- **M1:** Logistic Regression  
- **M2:** Decision Tree  
- **M3:** Random Forest  
- **M4:** Gradient Boosting  
- **M5:** SVC (RBF)  

### 7) Metric
We evaluate using **Accuracy** on the same fixed test set.

---

## Results

### Accuracy Matrix (Models × Sampling Techniques)
|                 |   Sampling1_SRS |   Sampling2_Stratified |   Sampling3_Systematic |   Sampling4_Cluster |   Sampling5_Bootstrap |
|:----------------|----------------:|-----------------------:|-----------------------:|--------------------:|----------------------:|
| M1_LogReg       |          0.915  |                 0.9183 |                 0.9379 |              0.8856 |                0.9118 |
| M2_DecisionTree |          0.9641 |                 0.9542 |                 0.9641 |              0.9673 |                0.9706 |
| M3_RandomForest |          0.9967 |                 0.9967 |                 0.9967 |              0.9183 |                0.9967 |
| M4_GradBoost    |          0.9935 |                 0.9935 |                 0.9967 |              0.9314 |                0.9902 |
| M5_SVC          |          0.9771 |                 0.9739 |                 0.9739 |              0.9118 |                0.9706 |

---

## Result Graphs

### Heatmap — Accuracy matrix
<img width="1280" height="960" alt="accuracy_heatmap" src="https://github.com/user-attachments/assets/e628a9b7-7a70-4f44-b937-304836af4bd4" />


### Best accuracy per model (across sampling)
<img width="1280" height="960" alt="best_by_model" src="https://github.com/user-attachments/assets/c4697ba2-e950-4023-a8a0-5bc3e4acfd6f" />


### Best accuracy per sampling technique (across models)
<img width="1280" height="960" alt="best_by_sampling" src="https://github.com/user-attachments/assets/7ed539cf-632a-4d8d-b1d9-111b462b4895" />


---

## How to Run

### A) Colab (recommended)
Open `notebooks/Assignment_2_Sampling.ipynb` in Colab and upload `Creditcard_data.csv` when prompted.

### B) Local run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
python run.py --data ../data/Creditcard_data.csv --sample_size 500 --seed 42 --outdir ../results
```
