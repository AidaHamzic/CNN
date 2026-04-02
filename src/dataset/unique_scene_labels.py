from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_CSV = PROJECT_ROOT / "data" / "interim" / "places365_val_index.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "places365_unique_scene_labels.csv"


def main():
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Index CSV not found: {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)

    required_columns = {"scene_label", "filename", "image_path", "exists"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in index CSV: {sorted(missing)}")

    unique_labels_df = (
        df[["scene_label"]]
        .drop_duplicates()
        .sort_values("scene_label")
        .reset_index(drop=True)
    )

    unique_labels_df.to_csv(OUTPUT_CSV, index=False)

    print("Unique scene label list created successfully")
    print(f"Total unique scene labels: {len(unique_labels_df)}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()