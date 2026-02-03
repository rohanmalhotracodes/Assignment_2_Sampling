# Sampling Assignment (GitHub Submission)

**Name:** Rohan Malhotra  
**Roll No:** 102303437  

This repository contains the complete solution for the **Sampling Assignment** (imbalanced credit card dataset): balancing the dataset, creating 5 samples using 5 sampling techniques, training 5 ML models, and comparing accuracies.

---

## Repository Structure

- `data/`  
  - `Creditcard_data.csv` (dataset)
- `src/`  
  - `run.py` (main runner)
  - `sampling.py` (5 sampling techniques)
  - `models.py` (5 ML models)
- `results/`  
  - `accuracy_matrix.csv` (generated)
  - `accuracy_matrix.md` (generated)
- `report/`  
  - `DISCUSSION.md` (discussion + best technique/model summary)
- `Sampling_Assignment.pdf` (assignment statement for reference)

---

## How to Run

### 1) Create environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the experiment
```bash
cd src
python run.py --data ../data/Creditcard_data.csv --sample_size 500 --seed 42 --outdir ../results
```

This will generate:
- `results/accuracy_matrix.csv`
- `results/accuracy_matrix.md`

---

## Results (Accuracy Matrix)

| Model           |   Sampling1_SRS |   Sampling2_Stratified |   Sampling3_Systematic |   Sampling4_Cluster |   Sampling5_Bootstrap |
|:----------------|----------------:|-----------------------:|-----------------------:|--------------------:|----------------------:|
| M1_LogReg       |          0.915  |                 0.9183 |                 0.9379 |              0.8856 |                0.9118 |
| M2_DecisionTree |          0.9641 |                 0.9542 |                 0.9641 |              0.9673 |                0.9706 |
| M3_RandomForest |          0.9967 |                 0.9967 |                 0.9967 |              0.9183 |                0.9967 |
| M4_GradBoost    |          0.9935 |                 0.9935 |                 0.9967 |              0.9314 |                0.9902 |
| M5_SVC          |          0.9771 |                 0.9739 |                 0.9739 |              0.9118 |                0.9706 |

### Best sampling technique per model
|                 | Best Sampling        |   Best Accuracy |
|:----------------|:---------------------|----------------:|
| M1_LogReg       | Sampling3_Systematic |          0.9379 |
| M2_DecisionTree | Sampling5_Bootstrap  |          0.9706 |
| M3_RandomForest | Sampling1_SRS        |          0.9967 |
| M4_GradBoost    | Sampling3_Systematic |          0.9967 |
| M5_SVC          | Sampling1_SRS        |          0.9771 |

### Best model per sampling technique
|                      | Best Model      |   Best Accuracy |
|:---------------------|:----------------|----------------:|
| Sampling1_SRS        | M3_RandomForest |          0.9967 |
| Sampling2_Stratified | M3_RandomForest |          0.9967 |
| Sampling3_Systematic | M3_RandomForest |          0.9967 |
| Sampling4_Cluster    | M2_DecisionTree |          0.9673 |
| Sampling5_Bootstrap  | M3_RandomForest |          0.9967 |

---

## Notes / Assumptions
- Dataset balancing is done using **SMOTE** (k_neighbors=3) to obtain a 50:50 class distribution before sampling.
- Models are trained on sampled subsets of the balanced training split and evaluated on a **fixed** held-out test split.
- All random seeds are fixed (`seed=42`) for reproducibility.

For a short write-up, see `report/DISCUSSION.md`.
