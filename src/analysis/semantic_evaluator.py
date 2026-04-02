import json
from pathlib import Path

import pandas as pd

from src.analysis.semantic_mapping import SEMANTIC_MAP

INPUT_CSV = Path("../../outputs/combined_raw_predictions.csv")
OUTPUT_CSV = Path("../../outputs/semantic_evaluated_predictions.csv")


def build_label_to_class_map(semantic_map: dict[str, set[str]]) -> dict[str, str]:
    label_to_class = {}

    for thesis_class, labels in semantic_map.items():
        for label in labels:
            if label in label_to_class:
                raise ValueError(
                    f"Duplicate label in semantic map: '{label}' appears in both "
                    f"'{label_to_class[label]}' and '{thesis_class}'"
                )
            label_to_class[label] = thesis_class

    return label_to_class


LABEL_TO_CLASS = build_label_to_class_map(SEMANTIC_MAP)


def map_top1_label_to_thesis(top1_label: str) -> str | None:
    if not isinstance(top1_label, str):
        return None
    return LABEL_TO_CLASS.get(top1_label)


def parse_top5_predictions(top5_json: str) -> list[dict]:
    try:
        parsed = json.loads(top5_json)
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict) and "label" in item]


def map_top5_labels_to_thesis(top5_predictions: list[dict]) -> list[str]:
    mapped = []

    for item in top5_predictions:
        label = item.get("label")
        if not isinstance(label, str):
            continue

        thesis_class = LABEL_TO_CLASS.get(label)
        if thesis_class is not None:
            mapped.append(thesis_class)

    return mapped


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "image_id",
        "thesis_class",
        "model_name",
        "top1_label",
        "top5_predictions_json",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    output_rows = []

    for _, row in df.iterrows():
        predicted_class = map_top1_label_to_thesis(row["top1_label"])

        top5_predictions = parse_top5_predictions(row["top5_predictions_json"])
        mapped_top5_classes = map_top5_labels_to_thesis(top5_predictions)

        row_dict = row.to_dict()
        row_dict["semantic_predicted_class"] = predicted_class
        row_dict["semantic_correct"] = (
            predicted_class == row["thesis_class"]
            if predicted_class is not None
            else False
        )
        row_dict["top5_mapped_thesis_classes_json"] = json.dumps(mapped_top5_classes)
        row_dict["top5_contains_true_class"] = row["thesis_class"] in mapped_top5_classes

        output_rows.append(row_dict)

    out_df = pd.DataFrame(output_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    none_rate = out_df["semantic_predicted_class"].isna().mean()
    print(f"Unmapped label rate: {none_rate:.4f}")

    accuracy = out_df["semantic_correct"].mean()

    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Rows: {len(out_df)}")
    print(f"Semantic top-1 accuracy: {accuracy:.4f}")
    print("\nAccuracy by model:")
    print(out_df.groupby("model_name")["semantic_correct"].mean().to_string())

    print("\nTop unmapped labels:")
    print(
        out_df.loc[out_df["semantic_predicted_class"].isna(), "top1_label"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nFirst 10 rows:")
    print(
        out_df[
            [
                "image_id",
                "model_name",
                "thesis_class",
                "top1_label",
                "semantic_predicted_class",
                "semantic_correct",
                "top5_contains_true_class",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()