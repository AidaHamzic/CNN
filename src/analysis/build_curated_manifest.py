from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "outputs" / "semantic_evaluated_predictions.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "curated_manifest.csv"

IMAGES_PER_CATEGORY = 5

CATEGORY_NAMES = {
    1: "all_correct",
    2: "all_wrong_two_agree",
    3: "two_correct_one_wrong",
    4: "two_wrong_agree_one_correct",
}

CATEGORY_DESCRIPTIONS = {
    1: "All 3 models correct — architectural consensus in success",
    2: "All 3 models wrong, 2 agree on same wrong class — shared blind spot",
    3: "2 models correct, 1 wrong — one architecture breaks from majority",
    4: "1 model correct, 2 wrong same class — one architecture resists shared failure",
}


def assign_category(row: pd.Series) -> int | None:
    preds = [row["pred_vgg16"], row["pred_resnet18"], row["pred_mobilenetv2"]]
    correct = [row["correct_vgg16"], row["correct_resnet18"], row["correct_mobilenetv2"]]
    n_correct = sum(correct)

    if n_correct == 3:
        return 1

    if n_correct == 2:
        return 3

    if n_correct == 1:
        wrong_preds = [p for p, c in zip(preds, correct) if not c]
        if len(wrong_preds) == 2 and wrong_preds[0] == wrong_preds[1]:
            return 4
        return None

    if n_correct == 0:
        if len(set(preds)) == 2:
            return 2

    return None


def build_manifest() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "image_id", "image_path", "thesis_class", "scene_label",
        "model_name", "semantic_predicted_class", "semantic_correct",
        "top1_label", "top1_confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pred_wide = df.pivot_table(
        index="image_id", columns="model_name",
        values="semantic_predicted_class", aggfunc="first",
    ).rename(columns=lambda m: f"pred_{m}")

    correct_wide = df.pivot_table(
        index="image_id", columns="model_name",
        values="semantic_correct", aggfunc="first",
    ).rename(columns=lambda m: f"correct_{m}")

    conf_wide = df.pivot_table(
        index="image_id", columns="model_name",
        values="top1_confidence", aggfunc="first",
    ).rename(columns=lambda m: f"conf_{m}")

    label_wide = df.pivot_table(
        index="image_id", columns="model_name",
        values="top1_label", aggfunc="first",
    ).rename(columns=lambda m: f"top1_label_{m}")

    meta = (
        df[["image_id", "image_path", "thesis_class", "scene_label"]]
        .drop_duplicates("image_id")
        .set_index("image_id")
    )

    wide = meta.join([pred_wide, correct_wide, conf_wide, label_wide])
    wide = wide.dropna(subset=["pred_vgg16", "pred_resnet18", "pred_mobilenetv2"]).copy()
    wide["avg_confidence"] = wide[["conf_vgg16", "conf_resnet18", "conf_mobilenetv2"]].mean(axis=1)
    wide["category"] = wide.apply(assign_category, axis=1)
    wide = wide[wide["category"].notna()].copy()
    wide["category"] = wide["category"].astype(int)
    wide["category_name"] = wide["category"].map(CATEGORY_NAMES)
    wide["category_description"] = wide["category"].map(CATEGORY_DESCRIPTIONS)

    records = []
    summary = {}

    for thesis_class in sorted(wide["thesis_class"].unique()):
        summary[thesis_class] = {}
        for cat in sorted(CATEGORY_NAMES.keys()):
            subset = (
                wide[
                    (wide["thesis_class"] == thesis_class)
                    & (wide["category"] == cat)
                    ]
                .sort_values("avg_confidence", ascending=False)
            )
            selected = subset.head(IMAGES_PER_CATEGORY)
            summary[thesis_class][cat] = len(selected)
            if len(selected) > 0:
                records.append(selected)

    manifest = pd.concat(records).reset_index()

    col_order = [
        "image_id", "image_path", "thesis_class", "scene_label",
        "category", "category_name", "category_description",
        "pred_vgg16", "pred_resnet18", "pred_mobilenetv2",
        "correct_vgg16", "correct_resnet18", "correct_mobilenetv2",
        "top1_label_vgg16", "top1_label_resnet18", "top1_label_mobilenetv2",
        "conf_vgg16", "conf_resnet18", "conf_mobilenetv2",
        "avg_confidence",
    ]
    manifest = manifest[col_order]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"Total images selected: {len(manifest)}")
    print(f"Target: {IMAGES_PER_CATEGORY} x {len(CATEGORY_NAMES)} categories x 6 classes = "
          f"{IMAGES_PER_CATEGORY * len(CATEGORY_NAMES) * 6}")
    print()
    print(f"{'Class':<14}  Cat1  Cat2  Cat3  Cat4  Total")
    print("-" * 45)
    for thesis_class in sorted(summary.keys()):
        row_total = sum(summary[thesis_class].values())
        counts = "  ".join(f"{summary[thesis_class].get(c, 0):>4}"
                           for c in sorted(CATEGORY_NAMES.keys()))
        print(f"{thesis_class:<14}  {counts}  {row_total:>5}")

    print()
    print("Category legend:")
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        print(f"  {cat} — {desc}")

    print()
    warnings = [
        (tc, c, cnt)
        for tc, cats in summary.items()
        for c, cnt in cats.items()
        if cnt < IMAGES_PER_CATEGORY
    ]
    if warnings:
        print("WARNING — slots with fewer than 5 images:")
        for tc, c, cnt in warnings:
            print(f"  {tc} / category {c} ({CATEGORY_NAMES[c]}): only {cnt} available")
    else:
        print("All slots filled with 5 images.")


if __name__ == "__main__":
    build_manifest()