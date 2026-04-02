import pandas as pd

df = pd.read_csv("../../outputs/semantic_mapping_candidates.csv")

cls = "sea"

subset = df[df["thesis_class"] == cls]
subset = subset[subset["source_table"] == "top5"]
subset = subset.sort_values("count", ascending=False)

print(subset.head(20).to_string(index=False))