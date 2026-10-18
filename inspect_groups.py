
import pandas as pd
from pathlib import Path

path = Path(__file__).with_name("scam_scenarios_v2.csv")
df = pd.read_csv(path, dtype=str, keep_default_na=False)

# Sort similar scenario categories together for manual review
df = df.sort_values(["scenario_type", "language", "scenario_id"])

columns = [
    "scenario_id",
    "scenario_type",
    "language",
    "expected_label",
    "message",
    "scenario_group_id",
]

output = Path(__file__).with_name("scenario_review.csv")
df[columns].to_csv(output, index=False, encoding="utf-8-sig")

print("Created:", output.name)
print("Rows to review:", len(df))
print("\nScenario types:")
print(df["scenario_type"].value_counts().to_string())
print("\nOpen scenario_review.csv and compare related messages.")
