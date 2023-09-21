import cv2
import numpy as np


def center_crop(x, size):
  assert x.ndim in (3, 4)  # image or video
  H, W, C = x.shape[-3:]
  assert H == W  # for simplicity
  if isinstance(size, float):
    assert size <= 1
    size = int(np.round(size * H))
  assert isinstance(size, int) and H >= size > 0
  if H != size:
    H_offset = (H - min(size, H)) // 2
    W_offset = (W - min(size, W)) // 2
    assert H == size + 2 * H_offset
    assert W == size + 2 * W_offset
    x = x[..., H_offset:-H_offset, W_offset:-W_offset, :]
  return x


def pad(x, shape, fill_value=0., dtype=None):
  if isinstance(shape, int):
    shape = (shape, *x.shape[1:])
  assert x.ndim == len(shape)
  if x.shape != shape:
    dtype = dtype or x.dtype
    x_ = np.full(shape, fill_value, dtype=dtype)
    slices = tuple(slice(0, i) for i in x.shape)
    x_[slices] = x
    x = x_
  return x


def normalize(x, mean, std, channel_axis=-1):
  assert x.ndim in (3, 4)  # image or video
  assert mean.shape == std.shape and mean.ndim == 1  # 1d arrays
  assert x.shape[channel_axis] == len(mean) == len(std)  # match the number of channels
  if channel_axis < 0:
    channel_axis = x.ndim + channel_axis
  shape = [-1 if i == channel_axis else 1 for i in range(x.ndim)]
  mean = mean.reshape(shape)
  std = std.reshape(shape)
  x = (x - mean) / std
  return x


def resize(x, size):
  assert x.ndim in (3, 4)  # image or video
  assert isinstance(size, int) and size > 0
  H, W, C = x.shape[-3:]
  assert H == W  # for simplicity
  if H != size:
    if x.ndim == 3:
      x = cv2.resize(
        x, (size, size),
        interpolation=cv2.INTER_AREA
      )
    elif x.ndim == 4:
      x_ = np.empty((len(x), size, size, C), dtype=x.dtype)
      for i, x_i in enumerate(x):
        x_[i] = cv2.resize(
          x_i, (size, size),
          interpolation=cv2.INTER_AREA
        )
      x = x_
  return x
