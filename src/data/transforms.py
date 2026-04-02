from torchvision import transforms
from src.config.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def build_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])