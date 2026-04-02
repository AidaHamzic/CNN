from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_TXT = PROJECT_ROOT / "data" / "raw" / "places365" / "val.txt"
VAL_IMAGES = PROJECT_ROOT / "data" / "raw" / "places365" / "val_256"

OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "places365_val_index.csv"


def parse_line(line: str):
    """
    val/field-cultivated/Places365_val_00032701.jpg
    """
    parts = line.strip().split("/")

    if len(parts) != 3:
        raise ValueError(f"Invalid line format: {line}")

    split_name = parts[0]          # val
    scene_label = parts[1]         # field-cultivated
    filename = parts[2]            # image file

    return split_name, scene_label, filename


def main():

    if not VAL_TXT.exists():
        raise FileNotFoundError("val.txt not found")

    if not VAL_IMAGES.exists():
        raise FileNotFoundError("val_256 folder not found")

    rows = []

    with open(VAL_TXT, "r") as f:
        for i, line in enumerate(f):

            line = line.strip()
            if not line:
                continue

            split_name, scene_label, filename = parse_line(line)

            image_path = VAL_IMAGES / filename

            rows.append({
                "index": i,
                "split": split_name,
                "scene_label": scene_label,
                "filename": filename,
                "image_path": str(image_path),
                "exists": image_path.exists()
            })

    df = pd.DataFrame(rows)

    # HARD VALIDATION (no silent failures)

    missing = df[df["exists"] == False]

    if len(missing) > 0:
        print("\nERROR: Missing files detected")
        print(missing.head())
        raise RuntimeError("Dataset validation failed: missing files")

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDataset index created successfully")
    print(f"Total images: {len(df)}")
    print(f"Unique scenes: {df['scene_label'].nunique()}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()