# R² Score Report - Model 2 (Tanh + ReLU)

## Training Results Directory
`results_20251111_181434`

---

## Model Information
- **Model Type**: MLP with Symmetric Activation (Tanh) + ReLU
- **Timestamp**: 20251111_181434
- **Architecture**: Linear(16→64) → Tanh → Linear(64→64) → ReLU → Dropout(0.2) → Linear(64→1)
- **Parameters**: 5,313

---

## Dataset
- **Total Samples**: 50,000
- **Training Samples**: 40,000 (80%)
- **Validation Samples**: 10,000 (20%)
- **Input Features**: 16 (Pauli vectors)
- **Target**: Stabilizer Norm (SN) labels

---

## R² Scores (Coefficient of Determination)

### Training Set
- **R² Score**: **0.9650** (96.50%)
- **MSE**: 0.003206

### Validation Set
- **R² Score**: **0.9639** (96.39%)
- **MSE**: 0.003279

### Comparison
- **R² Difference**: 0.0011 (0.11%)
- **MSE Difference**: -0.000073

---

## Interpretation

### Overall Performance
✅ **EXCELLENT**: The model explains **>96% of variance** in the data

This is an excellent result, indicating that the model has learned the underlying relationship between Pauli vectors and stabilizer norms very well.

### Generalization Quality
✅ **EXCELLENT GENERALIZATION**: Train and validation R² scores are extremely close

- Training R²: 96.50%
- Validation R²: 96.39%
- Difference: Only 0.11%

This small difference indicates that the model generalizes very well to unseen data and is not overfitting.

---

## What R² Means

The R² score (coefficient of determination) measures how well the model's predictions match the actual values:

- **R² = 1.0**: Perfect predictions
- **R² = 0.9639**: The model explains 96.39% of the variance in the validation data
- **R² = 0.0**: Model performs no better than predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

In this case, **R² = 0.9639** means that:
1. 96.39% of the variation in stabilizer norm can be predicted from the Pauli vectors
2. Only 3.61% of the variation remains unexplained

---

## Comparison with MSE

The MSE (Mean Squared Error) from training log was:
- Best validation MSE: 0.003282 (from training_log.json)
- Computed validation MSE: 0.003279 (from R² script)

These values match closely, confirming the computation is correct.

---

## Key Findings

1. **High Predictive Power**: With R² > 96%, the model is highly effective at predicting stabilizer norms from Pauli vectors

2. **No Overfitting**: The minimal gap between training (96.50%) and validation (96.39%) R² indicates the model generalizes well

3. **Consistent Performance**: Both MSE and R² metrics confirm excellent model performance

4. **Tanh Activation Works Well**: The symmetric Tanh activation in the first layer effectively captures the structure in the quantum data

---

## Recommendations

✅ **Model is production-ready** for predicting stabilizer norms from Pauli vectors

The high R² score and excellent generalization make this model suitable for:
- Quantum state analysis
- Stabilizer norm prediction tasks
- Research applications requiring accurate predictions

---

## Files Generated

- `r_squared_results.json`: Detailed numerical results
- `R_SQUARED_REPORT.md`: This human-readable report (you are here)
- `compute_r_squared.py`: Script to recompute R² scores

To recompute R² for this or other results:
```bash
python compute_r_squared.py results_20251111_181434
```
