"""Sensor noise models. Each sensor picks one per access tier; privileged readings are exact."""
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Gaussian:
    """Additive zero-mean Gaussian noise with a standard deviation per field."""
    std: Mapping[str, float]
    nonnegative: tuple[str, ...] = field(default=())  # fields clipped at zero (e.g. touch force)

    def apply(self, data: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        out = dict(data)
        for key, std in self.std.items():
            value = np.asarray(data[key], dtype=float)
            noisy = value + rng.normal(0.0, std, value.shape)
            if key in self.nonnegative:
                noisy = np.maximum(noisy, 0.0)
            out[key] = noisy if value.ndim else float(noisy)
        return out


@dataclass(frozen=True)
class PixelNoise:
    """Gaussian noise on an 8-bit image field, clipped back to 0..255."""
    std: float
    key: str = "image"

    def apply(self, data: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        out = dict(data)
        image = np.asarray(data[self.key], dtype=float)
        out[self.key] = np.clip(image + rng.normal(0.0, self.std, image.shape), 0, 255).astype(np.uint8)
        return out
