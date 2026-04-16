from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "outputs" / "semantic_mapping_candidates.csv")

cls = "glacier"

subset = df[df["thesis_class"] == cls]
subset = subset[subset["source_table"] == "top5"]
subset = subset.sort_values("count", ascending=False)

print(subset.head(20).to_string(index=False))