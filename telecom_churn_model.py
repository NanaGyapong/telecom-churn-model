"""
Telecom Customer Churn Prediction Model
========================================
End-to-end pipeline: data generation → EDA → preprocessing → 
model training → evaluation → feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC TELECOM DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 5000

def generate_telecom_data(n):
    tenure          = np.random.exponential(scale=30, size=n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n).clip(20, 120)
    total_charges   = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)

    contract        = np.random.choice(["Month-to-month", "One year", "Two year"],
                                        n, p=[0.55, 0.25, 0.20])
    internet        = np.random.choice(["DSL", "Fiber optic", "No"],
                                        n, p=[0.35, 0.45, 0.20])
    payment         = np.random.choice(
                        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                        n, p=[0.35, 0.22, 0.22, 0.21])

    num_complaints  = np.random.poisson(0.5, n)
    num_calls       = np.random.poisson(3, n)
    data_usage_gb   = np.random.exponential(10, n).clip(0, 80)

    # Churn probability based on realistic drivers
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        - 0.20 * (tenure > 24)
        + 0.15 * (num_complaints > 1)
        + 0.10 * (monthly_charges > 80)
        - 0.10 * (contract == "Two year")
        + 0.08 * (payment == "Electronic check")
        + np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.95)

    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges.round(2),
        "total_charges":    total_charges.round(2),
        "contract":         contract,
        "internet_service": internet,
        "payment_method":   payment,
        "num_complaints":   num_complaints,
        "num_support_calls":num_calls,
        "data_usage_gb":    data_usage_gb.round(2),
        "senior_citizen":   np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "partner":          np.random.choice(["Yes", "No"], n),
        "dependents":       np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tech_support":     np.random.choice(["Yes", "No", "No internet"], n, p=[0.29, 0.49, 0.22]),
        "online_security":  np.random.choice(["Yes", "No", "No internet"], n, p=[0.28, 0.50, 0.22]),
        "churn":            churn,
    })
    return df

df = generate_telecom_data(N)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.1%}\n")
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
cat_cols = ["contract", "internet_service", "payment_method",
            "partner", "dependents", "tech_support", "online_security"]
num_cols = ["tenure", "monthly_charges", "total_charges",
            "num_complaints", "num_support_calls", "data_usage_gb", "senior_citizen"]

df_encoded = df.copy()
le = LabelEncoder()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("churn", axis=1)
y = df_encoded["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train_sm)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "y_proba":y_proba,
        "auc":    roc_auc_score(y_test, y_proba),
        "ap":     average_precision_score(y_test, y_proba),
    }
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  ROC-AUC: {results[name]['auc']:.4f}  |  Avg Precision: {results[name]['ap']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n★ Best model: {best_name} (AUC = {best['auc']:.4f})")

# ─────────────────────────────────────────────
# 4. VISUALISATIONS  (single figure, 6 panels)
# ─────────────────────────────────────────────
palette = {"bg": "#0f172a", "card": "#1e293b", "accent": "#6366f1",
           "green": "#22c55e", "red": "#ef4444", "text": "#f1f5f9",
           "muted": "#94a3b8", "border": "#334155"}

plt.rcParams.update({
    "figure.facecolor": palette["bg"],
    "axes.facecolor":   palette["card"],
    "axes.edgecolor":   palette["border"],
    "axes.labelcolor":  palette["text"],
    "xtick.color":      palette["muted"],
    "ytick.color":      palette["muted"],
    "text.color":       palette["text"],
    "grid.color":       palette["border"],
    "grid.linewidth":   0.5,
})

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(palette["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: Churn Distribution ──────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["churn"].value_counts()
bars   = ax1.bar(["Stayed", "Churned"], counts.values,
                  color=[palette["green"], palette["red"]], width=0.5, zorder=2)
ax1.set_title("Churn Distribution", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Customers")
ax1.grid(axis="y", zorder=1)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}\n({val/N:.1%})", ha="center", va="bottom",
             fontsize=10, color=palette["text"])

# ── Panel 2: Churn by Contract ────────────────
ax2 = fig.add_subplot(gs[0, 1])
churn_by_contract = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
colors = [palette["red"] if v > 0.25 else palette["accent"] for v in churn_by_contract.values]
ax2.barh(churn_by_contract.index, churn_by_contract.values * 100,
          color=colors, zorder=2)
ax2.set_title("Churn Rate by Contract", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Churn Rate (%)")
ax2.grid(axis="x", zorder=1)
for i, (idx, val) in enumerate(churn_by_contract.items()):
    ax2.text(val * 100 + 0.5, i, f"{val:.1%}", va="center",
             fontsize=10, color=palette["text"])

# ── Panel 3: Monthly Charges Distribution ─────
ax3 = fig.add_subplot(gs[0, 2])
for label, color, ls in [(0, palette["green"], "-"), (1, palette["red"], "--")]:
    subset = df[df["churn"] == label]["monthly_charges"]
    ax3.hist(subset, bins=30, alpha=0.55, color=color, label=["Stayed", "Churned"][label],
             edgecolor="none", density=True)
ax3.set_title("Monthly Charges Distribution", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Monthly Charges ($)")
ax3.set_ylabel("Density")
ax3.legend(facecolor=palette["card"])

# ── Panel 4: ROC Curves ───────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
colors_roc = [palette["accent"], palette["green"], "#f59e0b"]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax4.plot(fpr, tpr, color=col, lw=2,
             label=f"{name} (AUC = {res['auc']:.3f})")
ax4.plot([0,1],[0,1], color=palette["muted"], lw=1, linestyle="--", label="Random")
ax4.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend(facecolor=palette["card"], fontsize=9)
ax4.grid(True)

# ── Panel 5: Confusion Matrix (best model) ────
ax5 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax5,
            xticklabels=["Stayed","Churned"],
            yticklabels=["Stayed","Churned"],
            linewidths=0.5, linecolor=palette["border"],
            annot_kws={"size": 13, "weight": "bold"})
ax5.set_title(f"Confusion Matrix\n{best_name}", fontsize=12, fontweight="bold", pad=10)
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")

# ── Panel 6: Feature Importance ───────────────
ax6 = fig.add_subplot(gs[2, :])
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)
bar_colors  = [palette["red"] if v > importances.quantile(0.75) else palette["accent"]
               for v in importances.values]
ax6.barh(importances.index, importances.values, color=bar_colors, zorder=2)
ax6.set_title("Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=10)
ax6.set_xlabel("Importance Score")
ax6.grid(axis="x", zorder=1)
for i, (feat, val) in enumerate(importances.items()):
    ax6.text(val + 0.001, i, f"{val:.3f}", va="center",
             fontsize=9, color=palette["text"])

fig.suptitle("Telecom Customer Churn Model — Full Analysis",
             fontsize=16, fontweight="bold", y=0.98, color=palette["text"])

plt.savefig("/mnt/user-data/outputs/telecom_churn_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=palette["bg"])
print("\nPlot saved.")

# ─────────────────────────────────────────────
# 5. SAVE SCORED TEST SET
# ─────────────────────────────────────────────
test_df = X_test.copy()
test_df["actual_churn"]     = y_test.values
test_df["churn_probability"] = best["y_proba"].round(4)
test_df["predicted_churn"]  = best["y_pred"]
test_df["risk_segment"] = pd.cut(test_df["churn_probability"],
                                  bins=[0, 0.3, 0.6, 1.0],
                                  labels=["Low", "Medium", "High"])
test_df.to_csv("/mnt/user-data/outputs/churn_scored_customers.csv", index=False)
print("Scored customer file saved.")

print(f"""
╔══════════════════════════════════════════╗
║         MODEL SUMMARY                   ║
║  Best Model : {best_name:<26}║
║  ROC-AUC    : {best['auc']:.4f}                    ║
║  Avg Prec.  : {best['ap']:.4f}                    ║
╚══════════════════════════════════════════╝
""")
