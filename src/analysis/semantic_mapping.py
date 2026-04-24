from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREQUENCY_CSV = PROJECT_ROOT / "outputs" / "label_frequency_analysis.csv"


MIN_DOMINANT_PCT = 65.0  # percent
MIN_TOTAL_APPEARANCES = 20  # count

THESIS_CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]



def _build_semantic_map() -> dict[str, set[str]]:

    if not FREQUENCY_CSV.exists():
        raise FileNotFoundError(
            f"Cannot build SEMANTIC_MAP: missing {FREQUENCY_CSV}\n"
            f"Run 'python -m src.analysis.build_label_frequency_analysis' first."
        )

    df = pd.read_csv(FREQUENCY_CSV)


    qualified = df[
        (df["dominant_pct"] >= MIN_DOMINANT_PCT)
        & (df["total_appearances"] >= MIN_TOTAL_APPEARANCES)
        ]

    semantic_map: dict[str, set[str]] = defaultdict(set)
    for _, row in qualified.iterrows():
        thesis_class = row["dominant_class"]
        if thesis_class in THESIS_CLASSES:
            semantic_map[thesis_class].add(row["label"])


    return {cls: semantic_map.get(cls, set()) for cls in THESIS_CLASSES}


SEMANTIC_MAP: dict[str, set[str]] = _build_semantic_map()
