"""RGB image augmentation for sim-to-real transfer.

Implements three augmentations applied to normalized RGB images (B, 3, H, W):
  1. Random spatial shift (per episode)
  2. Color jitter — brightness, contrast, saturation (per episode)
  3. Gaussian blur (per episode)

All parameters are sampled once at reset and held constant for the entire
episode, simulating fixed camera/lighting variation across deployments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from typing_extensions import override

from mjlab.utils.noise import noise_cfg, noise_model


@dataclass(kw_only=True)
class RgbAugmentationCfg(noise_cfg.NoiseModelCfg, class_type=noise_model.NoiseModel):
  """RGB-specific augmentation combining shift, color jitter, and blur.

  All augmentation parameters are sampled once per episode per environment
  (at reset) and held constant throughout the episode.
  """

  # Maximum shift in pixels for random translation (applied independently
  # to x and y). Mimics the random crop/shift from RAD (Laskin et al. 2020).
  max_shift_pixels: int = 4

  # Brightness jitter: multiplier sampled from [1 - brightness, 1 + brightness].
  brightness: float = 0.3

  # Contrast jitter: multiplier sampled from [1 - contrast, 1 + contrast].
  contrast: float = 0.3

  # Saturation jitter: multiplier sampled from [1 - saturation, 1 + saturation].
  saturation: float = 0.3

  # Per-channel color scale range, simulating auto white balance / exposure.
  # Each RGB channel gets an independent multiplier from this range.
  # Set to (1.0, 1.0) to disable.
  color_scale_range: tuple[float, float] = (0.7, 1.3)

  # Gaussian blur kernel size (must be odd). Set to 0 to disable.
  blur_kernel_size: int = 3

  # Gaussian blur sigma range. Sigma is sampled uniformly per episode.
  blur_sigma_range: tuple[float, float] = (0.1, 1.5)

  # Dummy noise_cfg required by NoiseModelCfg base — unused since the
  # model overrides __call__ entirely.
  noise_cfg: noise_cfg.NoiseCfg | None = None  # type: ignore[assignment]


class RgbAugmentationModel(noise_model.NoiseModel):
  """RGB augmentation noise model with all state sampled per episode."""

  def __init__(
    self,
    cfg: RgbAugmentationCfg,
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

    # Per-env color jitter factors, sampled at reset.
    # Shape (num_envs, 1, 1, 1) for broadcasting over (B, C, H, W).
    self._brightness_factor = torch.ones(num_envs, 1, 1, 1, device=device)
    self._contrast_factor = torch.ones(num_envs, 1, 1, 1, device=device)
    self._saturation_factor = torch.ones(num_envs, 1, 1, 1, device=device)

    # Per-env per-channel color scale, sampled at reset.
    # Shape (num_envs, 3, 1, 1) for broadcasting over (B, C, H, W).
    self._color_scale = torch.ones(num_envs, 3, 1, 1, device=device)

    # Per-env blur kernel, sampled at reset.
    k = cfg.blur_kernel_size
    if k > 0:
      assert k % 2 == 1, "blur_kernel_size must be odd"
      ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
      self._xx, self._yy = torch.meshgrid(ax, ax, indexing="ij")
      # Pre-compute distance squared for kernel: (1, 1, k*k).
      self._dist_sq = (self._xx**2 + self._yy**2).reshape(1, 1, -1)
      # Per-env kernel: (num_envs, 1, k*k).
      self._blur_kernel = torch.zeros(num_envs, 1, k * k, device=device)
    else:
      self._xx = self._yy = None
      self._dist_sq = None
      self._blur_kernel = None

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    indices = slice(None) if env_ids is None else env_ids
    n = (
      self._num_envs if isinstance(indices, slice) else len(env_ids)  # type: ignore[arg-type]
    )

    # Shift offsets.
    max_shift = self._cfg.max_shift_pixels
    if max_shift > 0:
      self._dy[indices] = torch.randint(0, 2 * max_shift + 1, (n,), device=self._device)
      self._dx[indices] = torch.randint(0, 2 * max_shift + 1, (n,), device=self._device)

    # Color jitter factors.
    if self._cfg.brightness > 0:
      b = self._cfg.brightness
      self._brightness_factor[indices] = torch.empty(
        n, 1, 1, 1, device=self._device
      ).uniform_(1 - b, 1 + b)

    if self._cfg.contrast > 0:
      c = self._cfg.contrast
      self._contrast_factor[indices] = torch.empty(
        n, 1, 1, 1, device=self._device
      ).uniform_(1 - c, 1 + c)

    if self._cfg.saturation > 0:
      s = self._cfg.saturation
      self._saturation_factor[indices] = torch.empty(
        n, 1, 1, 1, device=self._device
      ).uniform_(1 - s, 1 + s)

    # Per-channel color scale.
    lo, hi = self._cfg.color_scale_range
    if lo != hi:
      self._color_scale[indices] = torch.empty(
        n, 3, 1, 1, device=self._device
      ).uniform_(lo, hi)

    # Blur kernel.
    if self._blur_kernel is not None and self._dist_sq is not None:
      lo, hi = self._cfg.blur_sigma_range
      sigma = torch.empty(n, 1, 1, device=self._device).uniform_(lo, hi)
      kernel = torch.exp(-self._dist_sq / (2 * sigma**2))
      self._blur_kernel[indices] = kernel / kernel.sum(dim=-1, keepdim=True)

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    out = data

    # 1. Spatial shift using episode-persistent offsets.
    if self._cfg.max_shift_pixels > 0:
      out = self._apply_shift(out)

    # 2. Color jitter using episode-persistent factors.
    out = self._apply_color_jitter(out)

    # 3. Gaussian blur using episode-persistent kernel.
    if self._blur_kernel is not None:
      out = self._apply_blur(out)

    return torch.clamp(out, 0.0, 1.0)

  def _apply_shift(self, images: torch.Tensor) -> torch.Tensor:
    b, c, h, w = images.shape
    max_shift = self._cfg.max_shift_pixels
    padded = torch.nn.functional.pad(images, [max_shift] * 4, mode="replicate")
    # Build gather indices for vectorized crop: (B, C, H, W).
    batch_idx = torch.arange(b, device=images.device)[:, None, None, None]
    row_base = torch.arange(h, device=images.device)[None, None, :, None]
    col_base = torch.arange(w, device=images.device)[None, None, None, :]
    rows = row_base + self._dy[:b, None, None, None]  # (B, 1, H, 1)
    cols = col_base + self._dx[:b, None, None, None]  # (B, 1, 1, W)
    chan = torch.arange(c, device=images.device)[None, :, None, None]
    return padded[batch_idx, chan, rows, cols]

  def _apply_color_jitter(self, images: torch.Tensor) -> torch.Tensor:
    b = images.shape[0]

    # Per-channel color scale (white balance / exposure).
    lo, hi = self._cfg.color_scale_range
    if lo != hi:
      images = images * self._color_scale[:b]

    # Brightness.
    if self._cfg.brightness > 0:
      images = images * self._brightness_factor[:b]

    # Contrast: blend toward per-image mean.
    if self._cfg.contrast > 0:
      mean = images.mean(dim=(1, 2, 3), keepdim=True)
      images = mean + self._contrast_factor[:b] * (images - mean)

    # Saturation: blend toward grayscale.
    if self._cfg.saturation > 0:
      gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
      images = gray + self._saturation_factor[:b] * (images - gray)

    return images

  def _apply_blur(self, images: torch.Tensor) -> torch.Tensor:
    assert self._blur_kernel is not None
    b, c, h, w = images.shape
    k = self._cfg.blur_kernel_size

    # Use unfold for grouped convolution with pre-computed kernel.
    pad = k // 2
    padded = torch.nn.functional.pad(images, [pad] * 4, mode="replicate")
    # Unfold to patches: (B, C, H, W, k*k).
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    patches = patches.contiguous().view(b, c, h, w, k * k)
    # Apply per-env kernel: (B, 1, 1, 1, k*k).
    kernel = self._blur_kernel[:b].unsqueeze(1).unsqueeze(1)
    return (patches * kernel).sum(dim=-1)


# Wire the model class back to the config.
RgbAugmentationCfg.class_type = RgbAugmentationModel
