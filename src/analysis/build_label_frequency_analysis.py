import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.analysis.semantic_mapping import SEMANTIC_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "outputs" / "combined_raw_predictions.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "label_frequency_analysis.csv"

THESIS_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
SPECIFICITY_THRESHOLD = 0.65


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


def build_label_to_class_map(semantic_map: dict) -> dict[str, str]:

    result = {}
    for cls, labels in semantic_map.items():
        for label in labels:
            result[label] = cls
    return result


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")

    label_class_counts = build_label_class_counts(df)
    label_to_class = build_label_to_class_map(SEMANTIC_MAP)

    rows = []

    for label, class_counts in label_class_counts.items():
        total = sum(class_counts.values())
        dominant_class = max(class_counts, key=class_counts.get)
        dominant_count = class_counts[dominant_class]
        dominant_pct = dominant_count / total if total > 0 else 0.0


        per_class = {cls: class_counts.get(cls, 0) for cls in THESIS_CLASSES}
        per_class_pct = {
            f"{cls}_pct": (class_counts.get(cls, 0) / total * 100) if total > 0 else 0.0
            for cls in THESIS_CLASSES
        }


        mapped_to = label_to_class.get(label)
        included = mapped_to is not None


        if included:
            reason = (
                f"Included in '{mapped_to}' — fires {dominant_pct * 100:.1f}% "
                f"of the time in that class (>= {SPECIFICITY_THRESHOLD * 100:.0f}% threshold)"
            )
        elif dominant_pct >= SPECIFICITY_THRESHOLD:
            reason = (
                f"Eligible for '{dominant_class}' at {dominant_pct * 100:.1f}% "
                f"but not included — either not semantically coherent or "
                f"better represented by other labels already in the map"
            )
        else:
            reason = (
                f"Rejected — dominant class '{dominant_class}' only at "
                f"{dominant_pct * 100:.1f}% (< {SPECIFICITY_THRESHOLD * 100:.0f}% threshold). "
                f"Fires too broadly across multiple classes."
            )

        row = {
            "label": label,
            "total_appearances": total,
            "dominant_class": dominant_class,
            "dominant_pct": round(dominant_pct * 100, 1),
            "included_in_map": included,
            "mapped_to_class": mapped_to if included else "",
            "decision_reason": reason,
        }
        row.update({cls: per_class[cls] for cls in THESIS_CLASSES})
        row.update(per_class_pct)

        rows.append(row)

    out = (
        pd.DataFrame(rows)
        .sort_values(["dominant_class", "dominant_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Total unique labels analysed: {len(out)}")
    print(f"Labels included in semantic map: {out['included_in_map'].sum()}")
    print(f"Labels rejected (below {SPECIFICITY_THRESHOLD * 100:.0f}% threshold): "
          f"{(~out['included_in_map']).sum()}")

    print("\n--- Included labels by class ---")
    included = out[out["included_in_map"]].copy()
    for cls in THESIS_CLASSES:
        subset = included[included["mapped_to_class"] == cls][
            ["label", "dominant_pct", "total_appearances"]
        ].sort_values("dominant_pct", ascending=False)
        print(f"\n{cls.upper()} ({len(subset)} labels):")
        print(subset.to_string(index=False))

    print("\n--- Top rejected labels per class (closest to threshold) ---")
    rejected = out[~out["included_in_map"]].copy()
    for cls in THESIS_CLASSES:
        subset = rejected[rejected["dominant_class"] == cls].nlargest(5, "dominant_pct")[
            ["label", "dominant_pct", "total_appearances", "decision_reason"]
        ]
        if not subset.empty:
            print(f"\n{cls.upper()} — top rejected:")
            print(subset.to_string(index=False))


if __name__ == "__main__":
    main()