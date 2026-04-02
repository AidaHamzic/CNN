import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from src.config.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.data.load_thesis_dataset import load_thesis_dataset
from src.data.transforms import build_inference_transform
from src.models.imagenet_utils import decode_topk, top1_from_topk
from src.models.registry import MODEL_REGISTRY

DEVICE = torch.device("cpu")
MODEL_ORDER = ["vgg16", "resnet18", "mobilenetv2"]


def run_combined_raw_inference(
    dataset_csv: str | Path = "data/interim/thesis_dataset.csv",
    output_csv: str | Path = "outputs/combined_raw_predictions.csv",
) -> None:
    dataset_csv = Path(dataset_csv)
    output_csv = Path(output_csv)

    df = load_thesis_dataset(dataset_csv)
    transform = build_inference_transform()

    rows: list[dict] = []

    for model_name in MODEL_ORDER:
        print(f"\n=== Loading {model_name} ===")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY")

        model_entry = MODEL_REGISTRY[model_name]
        model = model_entry["loader"]().to(DEVICE)
        model.eval()

        layer_config = model_entry["layers"]

        required_layer_keys = ["early", "middle", "late", "gradcam_target"]
        missing_layer_keys = [k for k in required_layer_keys if k not in layer_config]
        if missing_layer_keys:
            raise ValueError(
                f"Model '{model_name}' is missing layer config keys: {missing_layer_keys}"
            )

        for i, sample in df.iterrows():
            image_path = Path(str(sample["image_path"]))

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image file not found during inference: {image_path}"
                )

            try:
                image = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                raise RuntimeError(f"Failed to open image '{image_path}': {e}") from e

            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)

            decoded_top5 = decode_topk(logits, model_name=model_name, topk=5)
            top1 = top1_from_topk(decoded_top5)


            output_row = sample.to_dict()


            output_row["model_name"] = model_name
            output_row["image_size"] = IMAGE_SIZE
            output_row["imagenet_mean"] = json.dumps(IMAGENET_MEAN)
            output_row["imagenet_std"] = json.dumps(IMAGENET_STD)

            output_row["early_layer_name"] = layer_config["early"]
            output_row["middle_layer_name"] = layer_config["middle"]
            output_row["late_layer_name"] = layer_config["late"]
            output_row["gradcam_target_layer"] = layer_config["gradcam_target"]

            output_row["top1_index"] = int(top1["index"])
            output_row["top1_label"] = top1["label"]
            output_row["top1_confidence"] = float(top1["confidence"])
            output_row["top5_predictions_json"] = json.dumps(decoded_top5)

            rows.append(output_row)

            if (i + 1) % 100 == 0:
                print(f"{model_name}: processed {i + 1}/{len(df)}")

    output_df = pd.DataFrame(rows)

    expected_rows = len(df) * len(MODEL_ORDER)
    if len(output_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} output rows, got {len(output_df)}"
        )

    counts_per_model = output_df["model_name"].value_counts().to_dict()
    for model_name in MODEL_ORDER:
        if counts_per_model.get(model_name, 0) != len(df):
            raise ValueError(
                f"Model '{model_name}' has {counts_per_model.get(model_name, 0)} rows; "
                f"expected {len(df)}"
            )

    if output_df["top1_label"].isna().any():
        raise ValueError("Some rows have missing top1_label")

    if output_df["top5_predictions_json"].isna().any():
        raise ValueError("Some rows have missing top5_predictions_json")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print("\nDONE")
    print(f"Saved to: {output_csv}")
    print(f"Total rows: {len(output_df)}")


if __name__ == "__main__":
    run_combined_raw_inference()