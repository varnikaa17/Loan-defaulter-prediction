import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ============================================================
# BANK LOAN DEFAULTER PREDICTION
# Dataset is synthetic and designed for an ML project/demo.
# ============================================================

DATA_FILE = "data.csv"

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 65)
print("BANK LOAN DEFAULTER PREDICTION")
print("=" * 65)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDefault distribution:")
print(df["Default"].value_counts())
print("\nDefault percentage:")
print((df["Default"].value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# Features / target
# ------------------------------------------------------------

X = df.drop(columns=["Default", "Customer_ID", "Name"])
y = df["Default"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ],
    remainder="passthrough"
)

# ------------------------------------------------------------
# Honest train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

models = {
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=600,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=600,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.90,
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []

print("\n" + "=" * 65)
print("5-FOLD CROSS VALIDATION")
print("=" * 65)

for name, classifier in models.items():

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1
    )

    rows.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Sensitivity": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC_AUC": scores["test_roc_auc"].mean()
    })

results = pd.DataFrame(rows).sort_values(
    "ROC_AUC",
    ascending=False
)

print(results.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.3f}".format,
        "Precision": "{:.3f}".format,
        "Sensitivity": "{:.3f}".format,
        "F1": "{:.3f}".format,
        "ROC_AUC": "{:.3f}".format
    }
))

# ------------------------------------------------------------
# Select model by cross-validated ROC-AUC
# ------------------------------------------------------------

best_name = results.iloc[0]["Model"]
best_classifier = models[best_name]

print("\nSelected model:", best_name)

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_classifier)
])

best_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Test prediction
# ------------------------------------------------------------

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
sensitivity = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 65)
print("FINAL UNSEEN TEST PERFORMANCE")
print("=" * 65)
print(f"Model                : {best_name}")
print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Sensitivity / Recall  : {sensitivity * 100:.2f}%")
print(f"F1 Score              : {f1 * 100:.2f}%")
print(f"ROC-AUC               : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-Defaulter", "Defaulter"],
    zero_division=0
))

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Defaulter", "Defaulter"],
    yticklabels=["Non-Defaulter", "Defaulter"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

# ------------------------------------------------------------
# ROC curve
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"{best_name} (AUC = {auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random Classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan Defaulter Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.close()

# ------------------------------------------------------------
# Model comparison graph
# ------------------------------------------------------------

results_plot = results.set_index("Model")[
    ["Accuracy", "Precision", "Sensitivity", "F1", "ROC_AUC"]
]

results_plot.plot(kind="bar", figsize=(11, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)
plt.close()

# ------------------------------------------------------------
# Test predictions
# ------------------------------------------------------------

test_output = df.loc[
    X_test.index,
    [
        "Customer_ID",
        "Name",
        "Age",
        "Monthly_Salary_INR",
        "Loan_Amount_INR",
        "Credit_Score",
        "Missed_Payments_12M",
        "Previous_Default",
        "Default"
    ]
].copy()

test_output["Actual"] = test_output["Default"].map({
    0: "Non-Defaulter",
    1: "Defaulter"
})

test_output["Prediction"] = y_pred

test_output["Prediction"] = test_output["Prediction"].map({
    0: "Non-Defaulter",
    1: "Defaulter"
})

test_output["Default_Probability_%"] = (y_prob * 100).round(2)

test_output.drop(columns=["Default"], inplace=True)

test_output.to_csv("test_predictions.csv", index=False)

# ------------------------------------------------------------
# Predictions for all 1000 records
# ------------------------------------------------------------

all_X = df.drop(columns=["Default", "Customer_ID", "Name"])

all_prob = best_model.predict_proba(all_X)[:, 1]
all_pred = (all_prob >= 0.50).astype(int)

all_output = df[
    [
        "Customer_ID",
        "Name",
        "Age",
        "Monthly_Salary_INR",
        "Loan_Amount_INR",
        "Credit_Score",
        "Missed_Payments_12M",
        "Previous_Default",
        "Default"
    ]
].copy()

all_output["Actual"] = all_output["Default"].map({
    0: "Non-Defaulter",
    1: "Defaulter"
})

all_output["Prediction"] = pd.Series(
    all_pred,
    index=all_output.index
).map({
    0: "Non-Defaulter",
    1: "Defaulter"
})

all_output["Default_Probability_%"] = (all_prob * 100).round(2)

all_output.drop(columns=["Default"], inplace=True)

all_output.to_csv("all_1000_predictions.csv", index=False)

print("\n" + "=" * 65)
print("FILES CREATED")
print("=" * 65)
print("test_predictions.csv")
print("all_1000_predictions.csv")
print("confusion_matrix.png")
print("roc_curve.png")
print("model_comparison.png")

print("\nFirst 20 test predictions:")
print(test_output.head(20).to_string(index=False))

print("\nProject completed.")
