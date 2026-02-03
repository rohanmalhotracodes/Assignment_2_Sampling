# Discussion & Findings

## What we did
1. Loaded the highly imbalanced credit card dataset (`Class` = 0/1).
2. Balanced the dataset using **SMOTE** (on the full dataset) to obtain a 50:50 class distribution.
3. Split the balanced dataset into train/test (80/20, stratified).
4. Created **five samples** of size **n=500** from the training set using five sampling techniques:
   - Sampling1: Simple Random Sampling (SRS)
   - Sampling2: Stratified Sampling
   - Sampling3: Systematic Sampling
   - Sampling4: Cluster Sampling (KMeans-based)
   - Sampling5: Bootstrap Sampling (with replacement)
5. Trained **five ML models** on each sample and evaluated **Accuracy** on the same held-out test set.

## Accuracy matrix (test set)
| Model           |   Sampling1_SRS |   Sampling2_Stratified |   Sampling3_Systematic |   Sampling4_Cluster |   Sampling5_Bootstrap |
|:----------------|----------------:|-----------------------:|-----------------------:|--------------------:|----------------------:|
| M1_LogReg       |          0.915  |                 0.9183 |                 0.9379 |              0.8856 |                0.9118 |
| M2_DecisionTree |          0.9641 |                 0.9542 |                 0.9641 |              0.9673 |                0.9706 |
| M3_RandomForest |          0.9967 |                 0.9967 |                 0.9967 |              0.9183 |                0.9967 |
| M4_GradBoost    |          0.9935 |                 0.9935 |                 0.9967 |              0.9314 |                0.9902 |
| M5_SVC          |          0.9771 |                 0.9739 |                 0.9739 |              0.9118 |                0.9706 |

## Best sampling technique for each model
|                 | Best Sampling        |   Best Accuracy |
|:----------------|:---------------------|----------------:|
| M1_LogReg       | Sampling3_Systematic |          0.9379 |
| M2_DecisionTree | Sampling5_Bootstrap  |          0.9706 |
| M3_RandomForest | Sampling1_SRS        |          0.9967 |
| M4_GradBoost    | Sampling3_Systematic |          0.9967 |
| M5_SVC          | Sampling1_SRS        |          0.9771 |

## Best model for each sampling technique
|                      | Best Model      |   Best Accuracy |
|:---------------------|:----------------|----------------:|
| Sampling1_SRS        | M3_RandomForest |          0.9967 |
| Sampling2_Stratified | M3_RandomForest |          0.9967 |
| Sampling3_Systematic | M3_RandomForest |          0.9967 |
| Sampling4_Cluster    | M2_DecisionTree |          0.9673 |
| Sampling5_Bootstrap  | M3_RandomForest |          0.9967 |

## Key takeaways
- **RandomForest (M3)** performed best overall on most sampling strategies (SRS / Stratified / Systematic / Bootstrap).
- **Cluster sampling** produced noticeably lower accuracy for ensemble models here, likely because cluster selection can bias the sample distribution.
- **Systematic sampling** slightly improved Logistic Regression and Gradient Boosting compared to plain random sampling in this run.

> Note: Results can vary slightly with random seeds, hyperparameters, and chosen sample size. The provided code fixes seeds for reproducibility.
