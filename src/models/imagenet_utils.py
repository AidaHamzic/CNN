import torch
from torchvision.models import (
    VGG16_Weights,
    ResNet18_Weights,
    MobileNet_V2_Weights,
)
WEIGHTS_MAP = {
    "vgg16": VGG16_Weights.IMAGENET1K_V1,
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "mobilenetv2": MobileNet_V2_Weights.IMAGENET1K_V1,
}


def get_imagenet_classes(model_name: str):
    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown model name: {model_name}")

    classes = WEIGHTS_MAP[model_name].meta["categories"]

    if len(classes) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes, got {len(classes)}")

    return classes


def decode_topk(logits: torch.Tensor, model_name: str, topk: int = 5):
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, 1000], got {tuple(logits.shape)}")

    classes = get_imagenet_classes(model_name)

    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=topk, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        idx_int = int(idx.item())
        results.append({
            "index": idx_int,
            "label": classes[idx_int],
            "confidence": float(prob.item()),
        })

    return results


def top1_from_topk(decoded_topk):
    if not decoded_topk:
        raise ValueError("decoded_topk is empty")
    return decoded_topk[0]