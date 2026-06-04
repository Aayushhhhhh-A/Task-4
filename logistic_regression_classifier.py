# Task 4: Classification with Logistic Regression
# Dataset: Breast Cancer Wisconsin (built into scikit-learn)
# Tools: Scikit-learn, Pandas, Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 1 = malignant, 0 = benign  (sklearn labels are reversed)

print("=" * 60)
print("BREAST CANCER WISCONSIN DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Classes: {data.target_names}")
print(f"Class distribution:\n{df['target'].value_counts()}")
print()

# ─────────────────────────────────────────
# 2. TRAIN/TEST SPLIT & FEATURE SCALING
# ─────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print()

# ─────────────────────────────────────────
# 3. TRAIN LOGISTIC REGRESSION MODEL
# ─────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# ─────────────────────────────────────────
# 4. EVALUATION METRICS
# ─────────────────────────────────────────
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("=" * 60)
print("MODEL EVALUATION (Default Threshold = 0.5)")
print("=" * 60)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ─────────────────────────────────────────
# 5. THRESHOLD TUNING
# ─────────────────────────────────────────
print("=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy'}")
print("-" * 60)
for t in thresholds:
    y_pred_t = (y_pred_prob >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    a = accuracy_score(y_test, y_pred_t)
    marker = " <-- default" if t == 0.5 else ""
    print(f"{t:<12.1f} {p:<12.4f} {r:<12.4f} {f:<12.4f} {a:.4f}{marker}")

# ─────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Task 4 — Logistic Regression on Breast Cancer Dataset",
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# --- Plot 1: Sigmoid Function ---
ax1 = fig.add_subplot(gs[0, 0])
z = np.linspace(-8, 8, 300)
sigmoid = 1 / (1 + np.exp(-z))
ax1.plot(z, sigmoid, color='royalblue', linewidth=2.5)
ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='Threshold = 0.5')
ax1.axvline(0,   color='gray', linestyle=':', linewidth=1)
ax1.fill_between(z, sigmoid, 0.5, where=(sigmoid > 0.5), alpha=0.15, color='green', label='Predict Malignant')
ax1.fill_between(z, sigmoid, 0.5, where=(sigmoid < 0.5), alpha=0.15, color='orange', label='Predict Benign')
ax1.set_title('Sigmoid Function  σ(z) = 1/(1+e⁻ᶻ)', fontsize=11)
ax1.set_xlabel('z  (linear combination of features)')
ax1.set_ylabel('Probability')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Confusion Matrix ---
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('Confusion Matrix', fontsize=11)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# --- Plot 3: ROC Curve ---
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
ax3.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.4f}')
ax3.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
ax3.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_title('ROC Curve', fontsize=11)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate (Recall)')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Plot 4: Precision-Recall vs Threshold ---
ax4 = fig.add_subplot(gs[1, 0])
thresh_range = np.linspace(0.01, 0.99, 200)
precisions, recalls, f1s = [], [], []
for t in thresh_range:
    yp = (y_pred_prob >= t).astype(int)
    precisions.append(precision_score(y_test, yp, zero_division=0))
    recalls.append(recall_score(y_test, yp, zero_division=0))
    f1s.append(f1_score(y_test, yp, zero_division=0))
ax4.plot(thresh_range, precisions, label='Precision', color='blue', lw=2)
ax4.plot(thresh_range, recalls,   label='Recall',    color='green', lw=2)
ax4.plot(thresh_range, f1s,       label='F1 Score',  color='red',   lw=2)
ax4.axvline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Threshold=0.5')
ax4.set_title('Precision / Recall / F1 vs Threshold', fontsize=11)
ax4.set_xlabel('Decision Threshold')
ax4.set_ylabel('Score')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# --- Plot 5: Prediction Probability Distribution ---
ax5 = fig.add_subplot(gs[1, 1])
malignant_probs = y_pred_prob[y_test == 1]
benign_probs    = y_pred_prob[y_test == 0]
ax5.hist(malignant_probs, bins=25, alpha=0.6, color='red',   label='Malignant (actual)', edgecolor='black', linewidth=0.5)
ax5.hist(benign_probs,    bins=25, alpha=0.6, color='green', label='Benign (actual)',    edgecolor='black', linewidth=0.5)
ax5.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
ax5.set_title('Predicted Probability Distribution', fontsize=11)
ax5.set_xlabel('Predicted Probability (Malignant)')
ax5.set_ylabel('Count')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- Plot 6: Top 10 Feature Importances ---
ax6 = fig.add_subplot(gs[1, 2])
coef = pd.Series(np.abs(model.coef_[0]), index=data.feature_names)
top10 = coef.nlargest(10).sort_values()
colors = ['#2196F3' if c > 0 else '#F44336' for c in top10]
top10.plot(kind='barh', ax=ax6, color='steelblue', edgecolor='black', linewidth=0.5)
ax6.set_title('Top 10 Feature Importances\n(|Coefficient| magnitude)', fontsize=11)
ax6.set_xlabel('|Coefficient|')
ax6.grid(True, axis='x', alpha=0.3)

plt.savefig('/mnt/user-data/outputs/logistic_regression_results.png',
            dpi=150, bbox_inches='tight')
print("\nPlot saved as: logistic_regression_results.png")
plt.show()
print("\nDone!")
