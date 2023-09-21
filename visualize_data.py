import argparse
from os import path, makedirs

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.data import (
  load_data,
  get_destination_angle,
  draw_minimap,
  draw_arrows
)
from utils.transforms import center_crop, resize

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to data (.npz)')
parser.add_argument('--out', default='.', help='output directory')
parser.add_argument('--predictions', help='path to predictions file (.pt)')
args = parser.parse_args()
# load data
_, ext = path.splitext(args.data)
if ext != '.npz':
  args.data = f'{args.data}.npz'
if not path.isfile(args.data):
  raise ValueError(f'Could not find data {args.data}')
data = load_data(args.data)
# optionally load predictions
if args.predictions:
  predictions = torch.load(args.predictions)
else:
  predictions = None
makedirs(args.out, exist_ok=True)
# draw minimaps
original_frame_size = 360
for file, (frame_index, video, movement) in (pbar := tqdm(data.items())):
  pbar.set_description(f'Drawing {file}')
  resize_factor = original_frame_size / video.shape[1]
  video = resize(video, size=original_frame_size)
  movement = np.round(resize_factor * movement).astype(int)
  # we center the video to 2/3 of the original, which improves the quality
  video = center_crop(video, size=2/3)
  minimap = draw_minimap(video, movement)
  minimap = cv2.cvtColor(minimap, cv2.COLOR_RGB2BGR)
  destination = get_destination_angle(movement)
  # draw arrows that point to the exit
  draw_arrows(
    minimap,
    angles=destination,
    movement=movement,
    frame_size=video.shape[1],
    c=(0, 128, 0)  # green
  )
  if predictions is not None and file in predictions['files']:
    i = predictions['files'].index(file)
    predicted_destination = predictions['logits'][i]
    draw_arrows(
      minimap,
      angles=predicted_destination,
      movement=movement,
      frame_size=video.shape[1],
      c=(0, 0, 255)  # red
    )
  filename, _ = path.splitext(path.basename(file))
  cv2.imwrite(path.join(args.out, f'{filename}.png'), minimap)
