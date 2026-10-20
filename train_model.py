
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FOLDER = Path(__file__).parent
DATA_FILE = FOLDER / "scam_scenarios_v2.csv"
MODEL_FILE = FOLDER / "baseline_model.joblib"
REPORT_FILE = FOLDER / "baseline_metrics.json"
LANGUAGE_REPORT_FILE = FOLDER / "language_metrics.csv"

LABELS = ["legitimate", "neutral", "suspicious"]

# 1. Load and validate the dataset
df = pd.read_csv(DATA_FILE, dtype=str)

required_columns = ["message", "expected_label", "language"]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_columns).copy()
df["message"] = df["message"].str.strip()
df["expected_label"] = df["expected_label"].str.strip().str.lower()
df["language"] = df["language"].str.strip().str.lower()

df = df[df["message"] != ""]

if not set(df["expected_label"]).issubset(set(LABELS)):
    raise ValueError("Unexpected labels found in expected_label.")

print("Dataset size:", len(df))
print("\nLabel distribution:")
print(df["expected_label"].value_counts())

print("\nLanguage distribution:")
print(df["language"].value_counts())

# 2. Split before training.
# This is a preliminary split; related scenarios may cross the split.
X = df["message"]
y = df["expected_label"]

X_train, X_test, y_train, y_test, lang_train, lang_test = (
    train_test_split(
        X,
        y,
        df["language"],
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

# 3. Build a baseline pipeline.
# Character n-grams can help with spelling and script variations.
model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=100_000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

# 4. Train on the training split only
model.fit(X_train, y_train)

# 5. Evaluate on held-out test messages
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 45)
print("PRELIMINARY TEST RESULTS")
print("=" * 45)
print("Training messages:", len(X_train))
print("Test messages:", len(X_test))
print(f"Accuracy: {accuracy:.3f}")

print("\nOverall classification report:")
overall_report = classification_report(
    y_test,
    predictions,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)
print(classification_report(
    y_test,
    predictions,
    labels=LABELS,
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=LABELS))

# 6. Evaluate each language within the test set
language_rows = []

print("\n" + "=" * 45)
print("RESULTS BY LANGUAGE")
print("=" * 45)

test_results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": predictions,
    "language": lang_test,
})

for language in sorted(test_results["language"].unique()):
    subset = test_results[test_results["language"] == language]

    print(f"\nLanguage: {language} | Test samples: {len(subset)}")
    print(classification_report(
        subset["actual"],
        subset["predicted"],
        labels=LABELS,
        zero_division=0,
    ))

    report = classification_report(
        subset["actual"],
        subset["predicted"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    language_rows.append({
        "language": language,
        "test_samples": len(subset),
        "accuracy": accuracy_score(subset["actual"], subset["predicted"]),
        "macro_f1": report["macro avg"]["f1-score"],
        "suspicious_recall": report["suspicious"]["recall"],
        "suspicious_precision": report["suspicious"]["precision"],
    })

pd.DataFrame(language_rows).to_csv(
    LANGUAGE_REPORT_FILE, index=False, encoding="utf-8-sig"
)

# Save a machine-readable record of the preliminary evaluation
with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump({
        "evaluation_type": "preliminary random stratified split",
        "warning": (
            "Related scenarios may occur in both splits. "
            "Do not treat these metrics as final generalization results."
        ),
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "overall_report": overall_report,
        "label_order_for_confusion_matrix": LABELS,
    }, file, indent=2)

# 7. Refit on all available messages for later integration.
# The metrics above still come from the held-out split.
model.fit(df["message"], df["expected_label"])
joblib.dump(model, MODEL_FILE)

print("\nSaved model:", MODEL_FILE.name)
print("Saved evaluation:", REPORT_FILE.name)
print("Saved language metrics:", LANGUAGE_REPORT_FILE.name)
print("\nBaseline complete. Dataset file was not modified.")
