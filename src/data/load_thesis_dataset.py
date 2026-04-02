from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "index",
    "split",
    "scene_label",
    "filename",
    "image_path",
    "exists",
    "thesis_class",
]


def load_thesis_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in thesis dataset: {missing}")

    df = df.copy()

    df["exists"] = df["exists"].astype(bool)

    if not df["exists"].all():
        missing_count = int((~df["exists"]).sum())
        raise ValueError(
            f"thesis_dataset.csv contains {missing_count} rows with exists=False"
        )

    if df["image_path"].duplicated().any():
        dup_count = int(df["image_path"].duplicated().sum())
        raise ValueError(
            f"thesis_dataset.csv contains {dup_count} duplicate image_path values"
        )

    path_exists = df["image_path"].map(lambda p: Path(str(p)).exists())
    if not path_exists.all():
        missing_count = int((~path_exists).sum())
        raise FileNotFoundError(
            f"{missing_count} image_path values in thesis_dataset.csv do not exist on disk"
        )

    if "image_id" not in df.columns:
        df["image_id"] = df["filename"]

    return df.reset_index(drop=True)