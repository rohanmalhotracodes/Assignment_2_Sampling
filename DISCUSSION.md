# Discussion & Findings

## Accuracy matrix (test set)
|                 |   Sampling1_SRS |   Sampling2_Stratified |   Sampling3_Systematic |   Sampling4_Cluster |   Sampling5_Bootstrap |
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

## Notes
- Dataset is balanced using SMOTE (50:50) before sampling.
- Each sampling technique draws n=500 points from the balanced training set.
- All models are evaluated on the same held-out test set.
