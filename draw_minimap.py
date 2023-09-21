import argparse
import functools
from os import path, makedirs

import cv2
import numpy as np
import skvideo.io

from utils.data import (
  stream_video,
  drop_frames,
  draw_minimap,
  draw_arrows,
  get_destination_angle
)
from utils.transforms import center_crop

parser = argparse.ArgumentParser()
parser.add_argument('video', help='path to a video file')
parser.add_argument('--out', help='where to save the minimap image instead')
parser.add_argument('--draw-targets', action='store_true', help='draw arrows that point to the exit')
args = parser.parse_args()
if not path.isfile(args.video):
  raise ValueError(f'Failed to find video: {args.video}')
# load metadata
metadata = skvideo.io.ffprobe(args.video)
num_frames = int(metadata['video']['@nb_frames'])
# read video
frame_index, video, movement = zip(*stream_video(
  file=args.video,
  # we center the video to about 2/3 of original, which improves the quality
  transform=functools.partial(center_crop, size=2/3),
  clip_movement=10,
  verbose=True
))
if args.draw_targets:
  _, _, large_movement = zip(*drop_frames(
    zip(frame_index, video, movement),
    movement_per_frame=50
  ))
  large_movement = np.round(large_movement).astype(int)
else:
  large_movement = None
video = np.array(video)
video = center_crop(video, size=2/3)
movement = np.round(movement).astype(int)
minimap = draw_minimap(video, movement)
minimap = cv2.cvtColor(minimap, cv2.COLOR_RGB2BGR)
if args.draw_targets:
  destination = get_destination_angle(large_movement)
  draw_arrows(
    minimap,
    angles=destination,
    movement=large_movement,
    frame_size=video.shape[1],
    c=(0, 128, 0)  # green
  )
if args.out:
  makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
  cv2.imwrite(args.out, minimap)
else:
  cv2.imshow(args.video, minimap)
  cv2.waitKey(0)
