import numpy as np
import torch
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


def _normalise_channel(channel: np.ndarray) -> np.ndarray:

    cmin, cmax = channel.min(), channel.max()
    if cmax - cmin > 1e-8:
        return (channel - cmin) / (cmax - cmin)
    return np.zeros_like(channel)


def make_feature_map_grid(
        activations: np.ndarray,
        n_channels: int = 16,
        cell_size: int = 96,
        padding: int = 3,
) -> Image.Image:

    n = min(n_channels, activations.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols

    grid_w = cols * cell_size + (cols + 1) * padding
    grid_h = rows * cell_size + (rows + 1) * padding
    grid = np.ones((grid_h, grid_w), dtype=np.float32)

    for idx in range(n):
        row = idx // cols
        col = idx % cols
        y = padding + row * (cell_size + padding)
        x = padding + col * (cell_size + padding)

        channel = activations[idx]
        channel_norm = _normalise_channel(channel)


        cell_pil = Image.fromarray((channel_norm * 255).astype(np.uint8), mode="L")
        cell_pil = cell_pil.resize((cell_size, cell_size), resample=Image.BICUBIC)
        cell_arr = np.array(cell_pil).astype(np.float32) / 255.0

        grid[y: y + cell_size, x: x + cell_size] = cell_arr

    grid_uint8 = (grid * 255).astype(np.uint8)
    return Image.fromarray(grid_uint8, mode="L").convert("RGB")


class FeatureMapExtractor:


    def __init__(
            self,
            model: torch.nn.Module,
            layer_names: dict[str, str],
    ) -> None:

        self.model = model
        self.layer_names = layer_names
        self._activations: dict[str, torch.Tensor] = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for stage, layer_name in self.layer_names.items():
            layer = _get_module(self.model, layer_name)


            def _make_hook(s):
                def _hook(module, input, output):
                    self._activations[s] = output.detach()

                return _hook

            handle = layer.register_forward_hook(_make_hook(stage))
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def extract(self, input_tensor: torch.Tensor) -> dict[str, np.ndarray]:

        self._activations.clear()
        self.model.eval()

        with torch.no_grad():
            self.model(input_tensor)

        if not self._activations:
            raise RuntimeError("No activations captured. Check layer names.")

        result = {}
        for stage in self.layer_names:
            if stage not in self._activations:
                raise RuntimeError(
                    f"Hook did not fire for stage '{stage}' "
                    f"(layer: '{self.layer_names[stage]}')"
                )

            result[stage] = self._activations[stage].squeeze(0).cpu().numpy()

        return result