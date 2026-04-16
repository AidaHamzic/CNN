import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _get_module(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = [k for k in modules if k]
        raise ValueError(
            f"Layer '{layer_name}' not found in model.\n"
            f"Available layers (first 20): {available[:20]}"
        )
    return modules[layer_name]


def _apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    colored = cm.jet(cam)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored


class GradCAM:


    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self._activations = None
        self._gradients = None
        self._tensor_grad_handle = None
        self._fwd_handle = None
        self._register_forward_hook(_get_module(model, target_layer_name))

    def _register_forward_hook(self, layer: torch.nn.Module) -> None:
        def _forward_hook(module, input, output):
            self._activations = output.detach()
            if self._tensor_grad_handle is not None:
                self._tensor_grad_handle.remove()
            self._tensor_grad_handle = output.register_hook(
                lambda grad: setattr(self, "_gradients", grad.detach())
            )

        self._fwd_handle = layer.register_forward_hook(_forward_hook)

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._tensor_grad_handle is not None:
            self._tensor_grad_handle.remove()

    def generate(
            self,
            input_tensor: torch.Tensor,
            class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        self._activations = None
        self._gradients = None

        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            confidence = float(probs[0, class_idx].item())

            self.model.zero_grad()
            logits[0, class_idx].backward()

        if self._activations is None:
            raise RuntimeError(
                f"Forward hook did not fire for layer '{self.target_layer_name}'."
            )
        if self._gradients is None:
            raise RuntimeError(
                f"Gradient hook did not fire for layer '{self.target_layer_name}'."
            )


        weights = self._gradients.mean(dim=[2, 3], keepdim=True)


        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence

    def overlay(
            self,
            original_image: Image.Image,
            cam: np.ndarray,
            alpha: float = 0.5,
    ) -> Image.Image:
        w, h = original_image.size
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
        cam_pil = cam_pil.resize((w, h), resample=Image.BICUBIC)
        cam_resized = np.array(cam_pil).astype(np.float32) / 255.0
        heatmap_rgb = _apply_jet_colormap(cam_resized)
        heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")
        original_rgb = original_image.convert("RGB")
        blended = Image.blend(original_rgb, heatmap_pil, alpha=alpha)
        return blended