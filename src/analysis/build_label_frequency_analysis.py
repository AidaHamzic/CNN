import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "outputs" / "combined_raw_predictions.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "label_frequency_analysis.csv"

THESIS_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def build_label_class_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        thesis_class = row["thesis_class"]
        try:
            predictions = json.loads(row["top5_predictions_json"])
        except (TypeError, json.JSONDecodeError):
            continue

        for pred in predictions:
            label = pred.get("label")
            if label:
                counts[label][thesis_class] += 1

    return counts


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} prediction rows from {INPUT_CSV.name}")

    label_class_counts = build_label_class_counts(df)
    print(f"Found {len(label_class_counts)} unique ImageNet labels in top-5 predictions")

    rows = []
    for label, class_counts in label_class_counts.items():
        total = sum(class_counts.values())
        dominant_class = max(class_counts, key=class_counts.get)
        dominant_count = class_counts[dominant_class]
        dominant_pct = (dominant_count / total * 100) if total > 0 else 0.0

        row = {
            "label": label,
            "total_appearances": total,
            "dominant_class": dominant_class,
            "dominant_pct": round(dominant_pct, 2),
        }

        for cls in THESIS_CLASSES:
            cls_count = class_counts.get(cls, 0)
            row[cls] = cls_count
            row[f"{cls}_pct"] = round(cls_count / total * 100, 2) if total > 0 else 0.0

        rows.append(row)

    out = (
        pd.DataFrame(rows)
        .sort_values(["dominant_class", "dominant_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Total labels analysed: {len(out)}")


    print("\n--- Threshold qualification preview ---")
    print(f"Labels with dominant_pct >= 65%: {(out['dominant_pct'] >= 65).sum()}")
    print(f"Labels with total_appearances >= 20: {(out['total_appearances'] >= 20).sum()}")
    qualified = out[(out["dominant_pct"] >= 65) & (out["total_appearances"] >= 20)]
    print(f"Labels meeting BOTH criteria (will be in semantic map): {len(qualified)}")

    print("\n--- Qualifying labels by class ---")
    for cls in THESIS_CLASSES:
        subset = qualified[qualified["dominant_class"] == cls][
            ["label", "dominant_pct", "total_appearances"]
        ]
        print(f"\n{cls.upper()} ({len(subset)} labels):")
        if not subset.empty:
            print(subset.to_string(index=False))
        else:
            print("  (none)")


if __name__ == "__main__":
    main()