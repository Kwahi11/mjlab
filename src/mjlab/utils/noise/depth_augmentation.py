"""Depth image augmentation for sim-to-real transfer.

Implements three augmentations applied to normalized depth images (B, 1, H, W):
  1. Additive Gaussian noise (per step)
  2. Random pixel dropout (per step, simulating sensor failures)
  3. Random spatial shift (per episode, simulating camera misalignment)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from typing_extensions import override

from mjlab.utils.noise import noise_cfg, noise_model


@dataclass(kw_only=True)
class DepthAugmentationCfg(noise_cfg.NoiseModelCfg, class_type=noise_model.NoiseModel):
  """Depth-specific augmentation combining noise, dropout, and shifts.

  Noise and dropout are applied fresh every step. The spatial shift is
  sampled once per episode per environment (at reset) and held constant,
  simulating a fixed camera misalignment per world.
  """

  # Gaussian noise standard deviation (in normalized depth units).
  noise_std: float = 0.01

  # Fraction of pixels to drop out (set to 0).
  dropout_ratio: float = 0.01

  # Maximum shift in pixels for random translation (applied independently
  # to x and y). Mimics the random crop/shift from RAD (Laskin et al. 2020).
  max_shift_pixels: int = 4

  # Dummy noise_cfg required by NoiseModelCfg base — unused since the
  # model overrides __call__ entirely.
  noise_cfg: noise_cfg.NoiseCfg | None = None  # type: ignore[assignment]


class DepthAugmentationModel(noise_model.NoiseModel):
  """Depth augmentation noise model with per-episode shift state."""

  def __init__(
    self,
    cfg: DepthAugmentationCfg,
    num_envs: int,
    device: str,
  ):
    # Skip NoiseModel.__init__ validation since we don't use noise_cfg.
    self._cfg = cfg
    self._num_envs = num_envs
    self._device = device

    # Per-env shift offsets, sampled at reset.
    self._dy = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._dx = torch.zeros(num_envs, dtype=torch.long, device=device)

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    max_shift = self._cfg.max_shift_pixels
    if max_shift <= 0:
      return
    indices = slice(None) if env_ids is None else env_ids
    n = self._num_envs if isinstance(indices, slice) else len(env_ids)  # type: ignore[arg-type]
    self._dy[indices] = torch.randint(0, 2 * max_shift + 1, (n,), device=self._device)
    self._dx[indices] = torch.randint(0, 2 * max_shift + 1, (n,), device=self._device)

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    out = data

    # 1. Additive Gaussian noise (per step).
    if self._cfg.noise_std > 0:
      out = out + self._cfg.noise_std * torch.randn_like(out)

    # 2. Random pixel dropout (per step).
    if self._cfg.dropout_ratio > 0:
      mask = torch.rand_like(out) > self._cfg.dropout_ratio
      out = out * mask

    # 3. Spatial shift using episode-persistent offsets.
    if self._cfg.max_shift_pixels > 0:
      out = self._apply_shift(out)

    return torch.clamp(out, 0.0, 1.0)

  def _apply_shift(self, images: torch.Tensor) -> torch.Tensor:
    b, c, h, w = images.shape
    max_shift = self._cfg.max_shift_pixels
    padded = torch.nn.functional.pad(images, [max_shift] * 4, mode="constant")
    # Build gather indices for vectorized crop: (B, C, H, W).
    batch_idx = torch.arange(b, device=images.device)[:, None, None, None]
    row_base = torch.arange(h, device=images.device)[None, None, :, None]
    col_base = torch.arange(w, device=images.device)[None, None, None, :]
    rows = row_base + self._dy[:b, None, None, None]  # (B, 1, H, 1)
    cols = col_base + self._dx[:b, None, None, None]  # (B, 1, 1, W)
    chan = torch.arange(c, device=images.device)[None, :, None, None]
    return padded[batch_idx, chan, rows, cols]


# Wire the model class back to the config.
DepthAugmentationCfg.class_type = DepthAugmentationModel
