"""
================================================================================
CLASSIFICATION MODEL - Random Forest Classifier
================================================================================

ENVIRONMENT SETUP:
------------------
Setting up a virtual environment to keep dependencies isolated from your 
system Python. Run these commands in your terminal:

1. Create the virtual environment:
   python -m venv assignment

2. Activate it:
   - On Linux/Mac:   source assignment/bin/activate
   - On Windows:     assignment\Scripts\activate

3. Install the required libraries:
   pip install pandas scikit-learn matplotlib seaborn openpyxl

4. Run the script:
   python classification_model.py

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Reading the Excel file directly (Sheet1 is default)
print("Loading data...")
df = pd.read_excel("data.xlsx", engine='openpyxl')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Dropping the 'ID' column - it's just an identifier, not predictive
# Keeping ID would cause the model to memorize rows instead of learning patterns
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
    print("Dropped 'ID' column (not predictive)")

# Separating features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}")
print()

# Handling missing values with mean imputation
# Using SimpleImputer because it's clean and sklearn-native
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

missing_filled = df.isnull().sum().sum()
print(f"Missing values filled with column means: {missing_filled}")
print()

# Using StratifiedKFold because dataset is small (only 260 rows)
print("=" * 50)
print("CROSS-VALIDATION RESULTS")
print("=" * 50)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')

print(f"F1 scores per fold: {np.round(cv_scores, 4)}")
print(f"Mean F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print()

# Training on 80/20 split to get detailed metrics
print("=" * 50)
print("FINAL TEST RESULTS (80/20 Split)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintaining class balance in both sets
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Train the final model
clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
clf_final.fit(X_train, y_train)

# Predictions on test set
y_pred = clf_final.predict(X_test)

# Classification report shows precision, recall, f1-score for each class
print("Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred))

# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion Matrix ---
# Shows where the model is making mistakes
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# --- Feature Importances (Top 10) ---
# Shows which features the model relies on most
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf_final.feature_importances_
}).sort_values('importance', ascending=False)

top_10 = feature_importance.head(10)

sns.barplot(x='importance', y='feature', data=top_10, ax=axes[1], hue='feature', palette='viridis', legend=False)
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('Feature')
axes[1].set_title('Top 10 Feature Importances')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150)
plt.show()

print()
print("=" * 50)
print("DONE! Plots saved to 'model_results.png'")
print("=" * 50)
