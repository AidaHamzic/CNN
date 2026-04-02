from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_CSV = PROJECT_ROOT / "data" / "interim" / "places365_val_index.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "thesis_dataset.csv"

CLASS_MAPPING = {

    "sea": [
        "beach",
        "coast",
        "ocean",
        "wave",
        "harbor"

    ],

    "forest": [
        "forest-broadleaf",
        "rainforest",
        "bamboo_forest",
        "forest_path",
        "forest_road"
    ],

    "mountain": [
        "mountain",
        "mountain_path",
        "mountain_snowy",
        "canyon",
        "volcano"



    ],

    "glacier": [
        "glacier",
        "iceberg",
        "ice_shelf",
        "ice_floe",
        "crevasse"
    ],

    "street": [
        "street",
        "crosswalk",
        "highway",
        "alley",
        "downtown"
    ],

    "buildings": [
        "building_facade",
        "skyscraper",
        "apartment_building-outdoor",
        "house",
        "office_building"

    ]
}


def build_reverse_mapping():
    reverse = {}
    for cls, labels in CLASS_MAPPING.items():
        for label in labels:
            if label in reverse:
                raise ValueError(f"Duplicate mapping: {label}")
            reverse[label] = cls
    return reverse


def main():

    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing file: {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)


    dataset_labels = set(df["scene_label"].unique())

    for cls, labels in CLASS_MAPPING.items():
        for label in labels:
            if label not in dataset_labels:
                raise ValueError(f"INVALID LABEL IN MAPPING: {label}")

    reverse_map = build_reverse_mapping()

    df["thesis_class"] = df["scene_label"].map(reverse_map)

    filtered_df = df[df["thesis_class"].notna()].copy()

    if len(filtered_df) == 0:
        raise RuntimeError("No data after filtering")

    print("\nClass distribution:")
    print(filtered_df["thesis_class"].value_counts())

    filtered_df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()