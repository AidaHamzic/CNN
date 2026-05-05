from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.config.constants import IMAGE_SIZE
from src.data.transforms import build_inference_transform
from src.models.registry import MODEL_REGISTRY
from src.visualisation.gradcam import GradCAM
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "curated_manifest.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "visualisations"
TEMP_DIR     = PROJECT_ROOT / "outputs" / "_norm_tmp"
DEVICE       = torch.device("cpu")
MODEL_ORDER  = ["vgg16", "resnet18", "mobilenetv2"]


def _raw_cam(gradcam: GradCAM) -> np.ndarray:

    weights  = gradcam._gradients.mean(dim=[2, 3], keepdim=True)
    cam      = (weights * gradcam._activations).sum(dim=1, keepdim=True)
    cam      = F.relu(cam)
    return cam.squeeze().cpu().numpy().astype(np.float32)


def _overlay(original: Image.Image, cam_norm: np.ndarray, alpha: float = 0.5) -> Image.Image:
    w, h        = original.size
    cam_pil     = Image.fromarray((cam_norm * 255).astype(np.uint8), mode="L")
    cam_pil     = cam_pil.resize((w, h), resample=Image.BICUBIC)
    cam_arr     = np.array(cam_pil).astype(np.float32) / 255.0
    colored     = cm.jet(cam_arr)
    heatmap     = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8), mode="RGB")
    return Image.blend(original.convert("RGB"), heatmap, alpha)


def run() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    manifest      = pd.read_csv(MANIFEST_CSV)
    unique_images = manifest[["image_id", "image_path"]].drop_duplicates("image_id")
    transform     = build_inference_transform()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nImages : {len(unique_images)}")
    print(f"Models : {MODEL_ORDER}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Temp   : {TEMP_DIR}\n")


    print("Phase 1 — collecting raw CAMs …")

    for model_name in MODEL_ORDER:
        print(f"\n{'=' * 50}")
        print(f"Loading {model_name} …")

        model_entry  = MODEL_REGISTRY[model_name]
        model        = model_entry["loader"]().to(DEVICE)
        model.eval()
        target_layer = model_entry["layers"]["gradcam_target"]

        for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc=model_name):
            image_id   = row["image_id"]
            image_path = Path(row["image_path"])
            image_stem = Path(image_id).stem


            tmp_path = TEMP_DIR / f"{image_stem}_{model_name}.npy"
            if tmp_path.exists():
                continue

            if not image_path.exists():
                print(f"\nWARNING: image not found, skipping: {image_path}")
                continue

            try:
                img    = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                print(f"\nWARNING: cannot open {image_path}: {e}")
                continue

            tensor  = transform(img).unsqueeze(0).to(DEVICE)
            gradcam = GradCAM(model, target_layer)
            try:
                gradcam.generate(tensor)
                cam_raw = _raw_cam(gradcam)
                np.save(tmp_path, cam_raw)
            except Exception as e:
                print(f"\nERROR {image_id} / {model_name}: {e}")
            finally:
                gradcam.remove_hooks()

        del model
        print(f"{model_name} done.")


    print("\n\nPhase 2 — normalising across models and saving overlays …")

    saved = 0
    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="normalising"):
        image_id   = row["image_id"]
        image_path = Path(row["image_path"])
        image_stem = Path(image_id).stem
        out_dir    = OUTPUT_DIR / image_stem


        out_paths = {m: out_dir / f"{m}_gradcam_normalized.png" for m in MODEL_ORDER}
        if all(p.exists() for p in out_paths.values()):
            continue


        raw_cams = {}
        missing  = False
        for m in MODEL_ORDER:
            tmp_path = TEMP_DIR / f"{image_stem}_{m}.npy"
            if not tmp_path.exists():
                print(f"\nWARNING: missing raw cam for {image_stem} / {m}, skipping image.")
                missing = True
                break
            raw_cams[m] = np.load(tmp_path)

        if missing:
            continue


        global_max = max(c.max() for c in raw_cams.values())
        if global_max < 1e-8:
            global_max = 1.0

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
        except (UnidentifiedImageError, OSError) as e:
            print(f"\nWARNING: cannot open {image_path}: {e}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        for m in MODEL_ORDER:
            cam_norm = np.clip(raw_cams[m] / global_max, 0.0, 1.0)
            overlay  = _overlay(img, cam_norm)
            overlay.save(out_paths[m])
            saved += 1


    print(f"\nDone. Normalised overlays saved: {saved}")
    print(f"Raw CAM files kept in: {TEMP_DIR}")


if __name__ == "__main__":
    run()
