from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.config.constants import IMAGE_SIZE
from src.data.transforms import build_inference_transform
from src.models.registry import MODEL_REGISTRY
from src.visualisation.gradcam import GradCAM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "curated_manifest.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "visualisations"
DEVICE       = torch.device("cpu")
MODEL_ORDER  = ["vgg16", "resnet18", "mobilenetv2"]


def cam_to_heatmap(cam: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    cam_pil     = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_pil     = cam_pil.resize(target_size, resample=Image.BICUBIC)
    cam_arr     = np.array(cam_pil).astype(np.float32) / 255.0
    colored     = cm.jet(cam_arr)
    rgb         = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def run() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    manifest       = pd.read_csv(MANIFEST_CSV)
    unique_images  = manifest[["image_id", "image_path"]].drop_duplicates("image_id")
    transform      = build_inference_transform()

    print(f"\nImages : {len(unique_images)}")
    print(f"Models : {MODEL_ORDER}")
    print(f"Output : {OUTPUT_DIR}\n")

    for model_name in MODEL_ORDER:
        print(f"\n{'=' * 50}")
        print(f"Loading {model_name} …")

        model_entry = MODEL_REGISTRY[model_name]
        model       = model_entry["loader"]().to(DEVICE)
        model.eval()
        target_layer = model_entry["layers"]["gradcam_target"]

        for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc=model_name):
            image_id   = row["image_id"]
            image_path = Path(row["image_path"])
            image_stem = Path(image_id).stem
            out_dir    = OUTPUT_DIR / image_stem
            out_path   = out_dir / f"{model_name}_gradcam_heatmap.png"

            if out_path.exists():
                continue

            if not image_path.exists():
                print(f"\nWARNING: image not found, skipping: {image_path}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                img = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"\nWARNING: cannot open {image_path}: {e}")
                continue

            tensor     = transform(img).unsqueeze(0).to(DEVICE)
            out_size   = (IMAGE_SIZE, IMAGE_SIZE)

            gradcam = GradCAM(model, target_layer)
            try:
                cam, _, _ = gradcam.generate(tensor)
                heatmap   = cam_to_heatmap(cam, out_size)
                heatmap.save(out_path)
            except Exception as e:
                print(f"\nERROR {image_id} / {model_name}: {e}")
            finally:
                gradcam.remove_hooks()

        del model
        print(f"{model_name} done.")

    total = len(list(OUTPUT_DIR.rglob("*_gradcam_heatmap.png")))
    print(f"\nDone. Heatmap-only files saved: {total}")


if __name__ == "__main__":
    run()
