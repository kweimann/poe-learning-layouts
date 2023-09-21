import argparse
from os import path, makedirs

import cv2
import numpy as np
import skvideo.io
import torch
from tqdm import tqdm

from model import MaskedVisionTransformer
from train_model import accuracy
from utils.data import stream_video, drop_frames, preprocess_video
from utils.transforms import center_crop

parser = argparse.ArgumentParser()
parser.add_argument('video', help='path to a video')
parser.add_argument('--model', required=True, help='path to a model')
parser.add_argument('--out', default='.', help='output directory')
parser.add_argument('--crop', type=int, nargs=4, help='crop minimap')
args = parser.parse_args()
if not path.isfile(args.video):
  raise ValueError(f'input video does not exist: {args.video}')
output_file = path.join(args.out, path.basename(args.video))
if path.isfile(output_file):
  raise ValueError(f'output video already exists: {output_file}')
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
# load model
ckpt = torch.load(args.model, map_location=device)
config = ckpt['config']
model = MaskedVisionTransformer(config).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
# read video
def transform(frame):
  if args.crop is not None:
    x_start, y_start, x_end, y_end = args.crop
    frame = frame[y_start:y_end, x_start:x_end]
  frame = center_crop(frame, size=2/3)
  return frame
video_stream = stream_video(
  file=args.video,
  transform=transform,
  clip_movement=10,
  verbose=True
)
video_stream = drop_frames(
  video_stream,
  movement_per_frame=50
)
frame_index, video, movement = zip(*video_stream)
frame_index = np.array(frame_index)
video = np.array(video)
if args.crop is not None:
  x_start, y_start, x_end, y_end = args.crop
  video = video[:, y_start:y_end, x_start:x_end]
movement = np.round(movement).astype(int)
video, metadata, targets, targets_mask = preprocess_video(
  video=video,
  movement=movement,
  context_size=config.context_size,
  frame_size=96,
  random_clip=False
)
video = torch.from_numpy(video).float()
metadata = torch.from_numpy(metadata).float()
targets = torch.from_numpy(targets).float()
targets_mask = torch.from_numpy(targets_mask).bool()
# predict where the exit is
with torch.inference_mode():
  logits = model(
    video.unsqueeze(0).to(device),
    metadata.unsqueeze(0).to(device),
    targets_mask=targets_mask.unsqueeze(0).to(device)
  ).squeeze(0).cpu()
# store video with predictions
makedirs(args.out, exist_ok=True)
metadata = skvideo.io.ffprobe(args.video)
num_frames = int(metadata['video']['@nb_frames'])
with skvideo.io.FFmpegReader(args.video) as reader:
  with skvideo.io.FFmpegWriter(output_file, outputdict={'-pix_fmt': 'yuv420p'}) as writer:
    for i, frame in enumerate(tqdm(reader.nextFrame(), desc=f'Writing {output_file}', total=num_frames)):
      k = np.searchsorted(frame_index, i, side='right') - 1
      H, W, C = frame.shape
      center_pt = np.array((
        W // 2,
        H // 2
      ))
      radius = max(H, W) // 12
      thickness = max(1, int(np.sqrt(radius)) - 1)
      target_pt = center_pt + np.array((
        radius * np.cos(targets[k].item()),  # dx
        radius * np.sin(-targets[k].item())  # dy
      )).round().astype(int)
      pred_pt = center_pt + np.array((
        radius * np.cos(logits[k].item()),   # dx
        radius * np.sin(-logits[k].item())   # dy
      )).round().astype(int)
      cv2.arrowedLine(frame, pt1=center_pt, pt2=target_pt, color=(0, 128, 0),
                      thickness=thickness, tipLength=0.2)
      cv2.arrowedLine(frame, pt1=center_pt, pt2=pred_pt, color=(255, 0, 0),
                      thickness=thickness, tipLength=0.2)
      writer.writeFrame(frame)
# print metrics
thresholds = [15, 30, 45, 60, 90]
for threshold in thresholds:
  acc = accuracy(targets, logits, targets_mask, threshold=threshold)
  print(f'acc@{threshold}\u00B0 {acc:.2%}')
