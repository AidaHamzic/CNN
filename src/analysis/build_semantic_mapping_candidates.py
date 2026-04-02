import pandas as pd

TOP1_PATH = "outputs/top1_label_frequencies_by_class_model.csv"
TOP5_PATH = "outputs/top5_label_frequencies_by_class_model.csv"
OUT_PATH = "outputs/semantic_mapping_candidates.csv"


def main():
    top1 = pd.read_csv(TOP1_PATH)
    top5 = pd.read_csv(TOP5_PATH)

    top1 = top1.rename(columns={"top1_label": "label"}) if "top1_label" in top1.columns else top1
    top5 = top5.rename(columns={"top5_labels": "label"}) if "top5_labels" in top5.columns else top5

    top1["source_table"] = "top1"
    top5["source_table"] = "top5"

    combined = pd.concat([top1, top5], ignore_index=True)

    required = {"thesis_class", "model_name", "label", "count", "source_table"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = (
        combined.groupby(["thesis_class", "label", "source_table"], as_index=False)["count"]
        .sum()
        .sort_values(["thesis_class", "count", "label"], ascending=[True, False, True])
    )

    model_presence = (
        combined.groupby(["thesis_class", "label"])["model_name"]
        .nunique()
        .reset_index(name="models_seen_in")
    )

    out = summary.merge(model_presence, on=["thesis_class", "label"], how="left")

    out["decision"] = ""
    out["reason"] = ""

    out.to_csv(OUT_PATH, index=False)

    print("DONE")
    print(f"Saved to: {OUT_PATH}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()