import argparse
from os import path, makedirs

import cv2
import numpy as np
import skvideo.io
import torch
from tqdm import tqdm

from model import MaskedVisionTransformer
from train_model import create_val_loader
from utils.data import load_data, draw_minimap
from utils.transforms import center_crop, resize

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to data (.npz)')
parser.add_argument('--model', required=True, help='path to a model')
parser.add_argument('--out', default='.', help='output directory')
args = parser.parse_args()
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
# load model
ckpt = torch.load(args.model, map_location=device)
config = ckpt['config']
model = MaskedVisionTransformer(config, flash_attn=False).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
# load video
data = load_data(args.data)
data = [(file, frame_index, video, movement) for file, (frame_index, video, movement) in data.items()]
dataloader = create_val_loader(data, config)
# run forward pass
attn_weights = []
with torch.inference_mode():
  for video, metadata, _, targets_mask in dataloader:
    model(
      video.to(device),
      metadata.to(device),
      targets_mask=targets_mask.to(device)
    )
    attn_weights.append(  # grab attention matrices from the last block
      model.blocks[-1].mha.attn_weights.clone().cpu()
    )
attn_weights = torch.cat(attn_weights)
# animate how the attention changes
makedirs(args.out, exist_ok=True)
original_frame_size = 360
for (file, frame_index, video, movement), attn in zip((pbar := tqdm(data)), attn_weights):
  pbar.set_description(f'Processing {file}')
  resize_factor = original_frame_size / video.shape[1]
  video = resize(video, size=original_frame_size)
  movement = np.round(resize_factor * movement).astype(int)
  video = center_crop(video, size=2/3)
  _, H, W, C = video.shape
  current_position = movement.cumsum(axis=0)
  x_min, y_min = current_position.min(axis=0)
  (x_start, y_start) = abs(x_min), abs(y_min)
  with skvideo.io.FFmpegWriter(
      path.join(args.out, file),
      inputdict={'-r': '5'},
      outputdict={'-pix_fmt': 'yuv420p'}
  ) as writer:
    for i in range(len(video)):
      minimap = draw_minimap(video[:i+1], movement)
      top_frames_per_head = attn[:, i, :i+1].argsort(descending=True, dim=1)[:, 0]  # (H,)
      for k in top_frames_per_head:  # draw circles on frames that have the largest attention weights
        center = (x_start + current_position[k, 0] + W // 2,
                  y_start + current_position[k, 1] + H // 2)
        cv2.circle(
          minimap,
          center,
          radius=min(W, H) // 2,
          color=(0, 0, 255),
          thickness=8)
      writer.writeFrame(minimap)
