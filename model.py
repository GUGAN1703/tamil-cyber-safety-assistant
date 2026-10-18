
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. Load the dataset
df = pd.read_csv("scam_scenarios_30.csv")

X = df["message"].astype(str)
y = df["expected_label"].str.strip().str.lower()

print("Dataset size:", len(df))
print("\nLabel counts:")
print(y.value_counts())


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Build the ML pipeline
model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(ngram_range=(1, 2))
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
])

# Inspect the training examples containing meeting-related words
meeting_examples = df[
    df["message"].str.contains(
        "meeting|schedule|appointment",
        case=False,
        regex=True
    )
]

print("\nMeeting-related examples in our dataset:")
print(meeting_examples[["message", "expected_label"]].to_string(index=False))


# 4. Train on the training data
model.fit(X_train, y_train)


# 5. Evaluate on unseen test messages
predictions = model.predict(X_test)

print("\nTest accuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["legitimate", "neutral", "suspicious"],
        zero_division=0
    )
)


# 6. Retrain the final demo model on all 30 examples
model.fit(X, y)


# 7. Create a function to analyze a new message
def analyze_message(message):
    predicted_label = model.predict([message])[0]

    scores = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_

    model_scores = dict(zip(classes, scores))

    return {
        "label": predicted_label,
        "model_scores": {
            label: round(float(score), 3)
            for label, score in model_scores.items()
        }
    }


# 8. Test with a new message
if __name__ == "__main__":
    message = input("\nEnter an English message to analyze: ")

    result = analyze_message(message)

    print("\nAnalysis result:")
    print("Predicted label:", result["label"])
    print("Model scores:", result["model_scores"])



# Show every test message and its actual/predicted label
comparison = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": predictions
})

print("\nTest predictions:")
print(comparison.to_string(index=False))

print("\nMisclassified messages:")
print(
    comparison[comparison["actual"] != comparison["predicted"]]
    .to_string(index=False)
)
