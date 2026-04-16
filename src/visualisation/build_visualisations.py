from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from src.config.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.data.transforms import build_inference_transform
from src.models.registry import MODEL_REGISTRY
from src.visualisation.gradcam import GradCAM
from src.visualisation.feature_maps import FeatureMapExtractor, make_feature_map_grid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "curated_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualisations"

DEVICE = torch.device("cpu")
MODEL_ORDER = ["vgg16", "resnet18", "mobilenetv2"]

N_CHANNELS = 16


def load_and_transform(image_path: Path, transform) -> tuple[Image.Image, torch.Tensor]:
    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"Failed to open image '{image_path}': {e}") from e


    original_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return original_resized, tensor


def process_image_model(
        image_path: Path,
        original_resized: Image.Image,
        tensor: torch.Tensor,
        model: torch.nn.Module,
        model_name: str,
        layer_config: dict,
        out_dir: Path,
) -> None:

    gradcam = GradCAM(model, layer_config["gradcam_target"])
    try:
        cam, class_idx, confidence = gradcam.generate(tensor)
        overlay = gradcam.overlay(original_resized, cam, alpha=0.5)
        overlay.save(out_dir / f"{model_name}_gradcam.png")
    finally:
        gradcam.remove_hooks()


    stage_layers = {
        "early": layer_config["early"],
        "middle": layer_config["middle"],
        "late": layer_config["late"],
    }
    extractor = FeatureMapExtractor(model, stage_layers)
    try:
        maps = extractor.extract(tensor)
        for stage, activations in maps.items():
            grid = make_feature_map_grid(activations, n_channels=N_CHANNELS)
            grid.save(out_dir / f"{model_name}_features_{stage}.png")
    finally:
        extractor.remove_hooks()


def run_precompute() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)
    transform = build_inference_transform()


    unique_images = manifest[["image_id", "image_path"]].drop_duplicates("image_id")
    print(f"\nImages to process: {len(unique_images)}")
    print(f"Models: {MODEL_ORDER}")
    print(f"Total forward passes: {len(unique_images) * len(MODEL_ORDER)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for model_name in MODEL_ORDER:
        print(f"\n{'=' * 50}")
        print(f"Loading {model_name}...")

        model_entry = MODEL_REGISTRY[model_name]
        model = model_entry["loader"]().to(DEVICE)
        model.eval()
        layer_config = model_entry["layers"]

        for _, row in tqdm(
                unique_images.iterrows(),
                total=len(unique_images),
                desc=model_name,
        ):
            image_id = row["image_id"]
            image_path = Path(row["image_path"])

            if not image_path.exists():
                print(f"\nWARNING: Image not found, skipping: {image_path}")
                continue


            image_stem = Path(image_id).stem
            out_dir = OUTPUT_DIR / image_stem
            out_dir.mkdir(parents=True, exist_ok=True)


            original_out = out_dir / "original.jpg"
            if not original_out.exists():
                try:
                    original_resized, tensor = load_and_transform(image_path, transform)
                    original_resized.save(original_out, quality=95)
                except RuntimeError as e:
                    print(f"\nWARNING: {e}")
                    continue
            else:
                try:
                    original_resized, tensor = load_and_transform(image_path, transform)
                except RuntimeError as e:
                    print(f"\nWARNING: {e}")
                    continue


            expected = [
                out_dir / f"{model_name}_gradcam.png",
                out_dir / f"{model_name}_features_early.png",
                out_dir / f"{model_name}_features_middle.png",
                out_dir / f"{model_name}_features_late.png",
            ]
            if all(p.exists() for p in expected):
                continue

            try:
                process_image_model(
                    image_path=image_path,
                    original_resized=original_resized,
                    tensor=tensor,
                    model=model,
                    model_name=model_name,
                    layer_config=layer_config,
                    out_dir=out_dir,
                )
            except Exception as e:
                print(f"\nERROR processing {image_id} with {model_name}: {e}")
                continue


        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"{model_name} done.")

    print(f"\n{'=' * 50}")
    print("PRECOMPUTE COMPLETE")
    print(f"Saved to: {OUTPUT_DIR}")


    gradcam_count = len(list(OUTPUT_DIR.rglob("*_gradcam.png")))
    feature_count = len(list(OUTPUT_DIR.rglob("*_features_*.png")))
    print(f"Grad-CAM overlays saved: {gradcam_count}")
    print(f"Feature map grids saved: {feature_count}")
    print(f"Expected: {len(unique_images) * len(MODEL_ORDER)} Grad-CAMs, "
          f"{len(unique_images) * len(MODEL_ORDER) * 3} feature map grids")


if __name__ == "__main__":
    run_precompute()