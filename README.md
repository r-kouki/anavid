# Classification Model - Random Forest Classifier

A practical Python script for solving a binary classification problem using Random Forest. This project predicts the 'Class' column (0 or 1) from student employability data.

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv assignment
```

### 2. Activate Virtual Environment

**Linux/Mac:**
```bash
source assignment/bin/activate
```

**Windows:**
```bash
assignment\Scripts\activate
```

### 3. Install Required Libraries

```bash
pip install pandas scikit-learn matplotlib seaborn openpyxl
```

## Running the Script

```bash
python classification_model.py
```

## Data

- **File:** `data.xlsx`
- **Size:** 260 rows, 39 columns
- **Target:** Binary classification (Class 0 or 1)
- **Features:** 37 features including demographic info, skill assessments, and employability metrics

## Model Pipeline

### 1. Data Loading
- Reads Excel file directly using pandas and openpyxl

### 2. Data Cleaning
- Drops 'ID' column (non-predictive identifier)
- Fills missing values with column means using `SimpleImputer`

### 3. Cross-Validation
- Uses **5-fold StratifiedKFold** for stable evaluation on small dataset
- Ensures each fold maintains class distribution
- Reports F1 scores for each fold and mean F1 score

### 4. Final Test
- **80/20 train-test split** with stratification
- Trains final Random Forest model (100 trees)
- Generates detailed classification report

### 5. Visualization
- **Confusion Matrix:** Shows model prediction accuracy
- **Feature Importances:** Bar chart of top 10 most important features

## Results

### Cross-Validation Performance
- **Mean F1 Score:** 0.9683 (±0.065)
- Consistent performance across all 5 folds

### Test Set Performance
- **Overall Accuracy:** 92%
- **Class 0:** F1-score 0.93 (Precision: 0.90, Recall: 0.96)
- **Class 1:** F1-score 0.92 (Precision: 0.96, Recall: 0.88)

## Visualizations

The script generates `model_results.png` with two plots:

![Model Results](model_results.png)

### Confusion Matrix
Shows the distribution of correct and incorrect predictions for each class.

### Top 10 Feature Importances
Identifies which features the Random Forest model relies on most for making predictions.

## Technical Notes

- **Algorithm:** Random Forest with 100 estimators
- **Random State:** 42 (for reproducibility)
- **Imputation Strategy:** Mean for numerical features
- **Validation Strategy:** Stratified K-Fold (k=5) for small dataset reliability

## Dependencies

- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms and metrics
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `openpyxl` - Excel file reading
- `numpy` - Numerical computing

## Output Files

- `model_results.png` - Confusion matrix and feature importance plots
