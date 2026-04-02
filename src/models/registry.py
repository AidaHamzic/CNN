import torch
import torchvision.models as models


def load_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def load_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def load_mobilenetv2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    return model


MODEL_REGISTRY = {
    "vgg16": {
        "loader": load_vgg16,
        "layers": {
            "early": "features.0",
            "middle": "features.14",
            "late": "features.28",
            "gradcam_target": "features.28",
        },
    },
    "resnet18": {
        "loader": load_resnet18,
        "layers": {
            "early": "layer1",
            "middle": "layer3",
            "late": "layer4",
            "gradcam_target": "layer4",
        },
    },
    "mobilenetv2": {
        "loader": load_mobilenetv2,
        "layers": {
            "early": "features.1",
            "middle": "features.7",
            "late": "features.18",
            "gradcam_target": "features.18",
        },
    },
}