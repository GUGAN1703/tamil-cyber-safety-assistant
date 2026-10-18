
from pathlib import Path
import pandas as pd

# Load the CSV from the same folder as this script
DATASET_PATH = Path(__file__).with_name("scam_scenarios_v2.csv")

REQUIRED_COLUMNS = [
    "scenario_id",
    "message",
    "language",
    "expected_label",
    "scenario_type",
    "warning_signs",
    "reason",
    "recommended_action",
    "confidence_notes",
    "source_type",
    "scenario_group_id",
]

VALID_LABELS = {"suspicious", "legitimate", "neutral"}
VALID_LANGUAGES = {"ta", "en", "tanglish"}

df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

print("=" * 55)
print("DATASET VALIDATION REPORT")
print("=" * 55)

# 1. Basic information
print("\n1. BASIC INFORMATION")
print("Dataset:", DATASET_PATH.name)
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Check required columns
print("\n2. REQUIRED COLUMNS")
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

if missing_columns:
    print("FAIL - Missing columns:", missing_columns)
else:
    print("PASS - All required columns exist")

# Stop checks that require missing columns
if missing_columns:
    raise SystemExit("Fix the missing columns before continuing.")

# 3. Check empty values
print("\n3. EMPTY VALUES")
empty_counts = (
    df[REQUIRED_COLUMNS]
    .apply(lambda column: column.str.strip().eq("").sum())
)

empty_counts = empty_counts[empty_counts > 0]

if empty_counts.empty:
    print("PASS - No empty required fields")
else:
    print("WARNING - Empty fields found:")
    print(empty_counts.to_string())

# 4. Check duplicate IDs
print("\n4. DUPLICATE SCENARIO IDs")
duplicate_ids = df[
    df["scenario_id"].duplicated(keep=False)
]

if duplicate_ids.empty:
    print("PASS - Scenario IDs are unique")
else:
    print("WARNING - Duplicate IDs:")
    print(duplicate_ids[["scenario_id", "message"]].to_string(index=False))

# 5. Check duplicate messages
print("\n5. DUPLICATE MESSAGES")
normalised_messages = (
    df["message"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

duplicate_messages = df[
    normalised_messages.duplicated(keep=False)
].copy()

if duplicate_messages.empty:
    print("PASS - No exact duplicates after basic normalisation")
else:
    print("WARNING - Possible duplicate messages:")
    print(
        duplicate_messages[
            ["scenario_id", "language", "expected_label", "message"]
        ].to_string(index=False)
    )

# 6. Label distribution
print("\n6. LABEL DISTRIBUTION")
print(df["expected_label"].value_counts().to_string())

invalid_labels = df[
    ~df["expected_label"].str.strip().str.lower().isin(VALID_LABELS)
]

if invalid_labels.empty:
    print("PASS - All labels are recognised")
else:
    print("WARNING - Invalid labels:")
    print(invalid_labels[["scenario_id", "expected_label"]].to_string(index=False))

# 7. Language distribution
print("\n7. LANGUAGE DISTRIBUTION")
print(df["language"].value_counts().to_string())

invalid_languages = df[
    ~df["language"].str.strip().str.lower().isin(VALID_LANGUAGES)
]

if invalid_languages.empty:
    print("PASS - All language codes are recognised")
else:
    print("WARNING - Invalid language codes:")
    print(invalid_languages[["scenario_id", "language"]].to_string(index=False))

print("\nLabel distribution by language:")
print(
    pd.crosstab(df["language"], df["expected_label"]).to_string()
)

# 8. Scenario type distribution
print("\n8. SCENARIO TYPES")
print(df["scenario_type"].value_counts().to_string())

# 9. Scenario group validation
print("\n9. SCENARIO GROUPS")

group_sizes = df.groupby("scenario_group_id").size()
print("Unique groups:", df["scenario_group_id"].nunique())
print("Groups with multiple rows:", int((group_sizes > 1).sum()))

group_language_counts = df.groupby("scenario_group_id")["language"].nunique()
multilingual_groups = group_language_counts[group_language_counts > 1]

print("Groups containing multiple languages:", len(multilingual_groups))

if len(group_sizes) == len(df):
    print(
        "WARNING - Every row has its own group ID. "
        "If rows are translations or variants of the same scenario, "
        "assign them a shared group ID."
    )

# Check whether a group contains conflicting labels
labels_per_group = df.groupby("scenario_group_id")[
    "expected_label"
].nunique()

conflicting_groups = labels_per_group[labels_per_group > 1]

if conflicting_groups.empty:
    print("PASS - No groups contain conflicting labels")
else:
    print("WARNING - Groups with conflicting labels:")
    print(conflicting_groups.to_string())

# 10. Final reminder
print("\n10. HUMAN REVIEW")
print("Review scenario_type, warning_signs, and reason against each message.")
print("This script checks structure, not whether a label is factually correct.")

print("\nValidation checks finished.")
