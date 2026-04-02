import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("outputs/combined_raw_predictions.csv")
OUTPUT_DIR = Path("outputs")


def parse_top5_predictions(top5_json: str) -> list[dict]:
    try:
        parsed = json.loads(top5_json)
        if not isinstance(parsed, list):
            return []
        return parsed
    except (TypeError, json.JSONDecodeError):
        return []


def build_top1_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    top1_freq = (
        df.groupby(["thesis_class", "model_name", "top1_label"])
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["thesis_class", "model_name", "count", "top1_label"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )
    return top1_freq


def build_top5_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    counter = defaultdict(Counter)

    for _, row in df.iterrows():
        thesis_class = row["thesis_class"]
        model_name = row["model_name"]
        top5_predictions = parse_top5_predictions(row["top5_predictions_json"])

        for pred in top5_predictions:
            label = pred.get("label")
            if label is not None:
                counter[(thesis_class, model_name)][label] += 1

    rows = []
    for (thesis_class, model_name), label_counter in counter.items():
        for label, count in label_counter.items():
            rows.append(
                {
                    "thesis_class": thesis_class,
                    "model_name": model_name,
                    "label": label,
                    "count": count,
                }
            )

    top5_freq = (
        pd.DataFrame(rows)
        .sort_values(
            by=["thesis_class", "model_name", "count", "label"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )
    return top5_freq


def build_top10_summary(
    df: pd.DataFrame,
    label_column: str,
    output_label_name: str,
) -> pd.DataFrame:
    top10 = (
        df.groupby(["thesis_class", "model_name"], group_keys=False)
        .head(10)
        .reset_index(drop=True)
        .copy()
    )
    top10 = top10.rename(columns={label_column: output_label_name})
    return top10


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "thesis_class",
        "model_name",
        "top1_label",
        "top5_predictions_json",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    top1_freq = build_top1_frequency_table(df)
    top5_freq = build_top5_frequency_table(df)

    if top5_freq.empty:
        raise ValueError("Top-5 frequency table is empty. Check top5_predictions_json parsing.")

    top1_top10 = build_top10_summary(
        top1_freq,
        label_column="top1_label",
        output_label_name="label",
    )
    top5_top10 = build_top10_summary(
        top5_freq,
        label_column="label",
        output_label_name="label",
    )

    top1_freq_path = OUTPUT_DIR / "top1_label_frequencies_by_class_model.csv"
    top5_freq_path = OUTPUT_DIR / "top5_label_frequencies_by_class_model.csv"
    top1_top10_path = OUTPUT_DIR / "top1_top10_summary.csv"
    top5_top10_path = OUTPUT_DIR / "top5_top10_summary.csv"

    top1_freq.to_csv(top1_freq_path, index=False)
    top5_freq.to_csv(top5_freq_path, index=False)
    top1_top10.to_csv(top1_top10_path, index=False)
    top5_top10.to_csv(top5_top10_path, index=False)

    print("DONE")
    print(f"Saved: {top1_freq_path}")
    print(f"Saved: {top5_freq_path}")
    print(f"Saved: {top1_top10_path}")
    print(f"Saved: {top5_top10_path}")

    print("\nTop-1 rows:", len(top1_freq))
    print("Top-5 rows:", len(top5_freq))


if __name__ == "__main__":
    main()