import argparse
import functools
import multiprocessing as mp
from os import path, listdir, makedirs

import numpy as np
from tqdm import tqdm

from utils.data import (
  stream_video,
  drop_frames,
  load_data,
  save_data
)
from utils.transforms import center_crop, resize

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='data directory')
parser.add_argument('--out', help='output file (.npz)')
parser.add_argument('--append', action='store_true', help='append to the existing .npz archive')
parser.add_argument('--num-workers', type=int, default=0, help='number of parallel processes')
parser.add_argument('--frame-size', type=int, default=96, help='resize video')
parser.add_argument('--clip-movement', type=int, default=10, help='clip movement above the threshold')
parser.add_argument('--movement-per-frame', type=int, default=50, help='minimal movement required to jump to next frame')
parser.add_argument('--translation-threshold', type=float, default=0.7, help='threshold for how good a translation match should be')
parser.add_argument('--translation-max-matches', type=int, default=15, help='number of best translation matches')


def read_video(file, config, verbose=False):
  video_stream = stream_video(
    file=path.join(config.dir, file),
    # we center the video to 2/3 of the original, which improves the quality
    transform=functools.partial(center_crop, size=2/3),
    clip_movement=config.clip_movement,
    translation_threshold=config.translation_threshold,
    translation_max_matches=config.translation_max_matches,
    verbose=verbose
  )
  if config.movement_per_frame > 0:  # drop frames with little movement
    video_stream = drop_frames(
      video_stream,
      movement_per_frame=config.movement_per_frame
    )
  # resize video
  frames = []
  for frame_index, frame, movement in video_stream:
    if config.frame_size:
      H, W, C = frame.shape
      frame = resize(frame, config.frame_size)
      movement = movement / (H / config.frame_size)  # resize movement
    frames.append((
      frame_index,
      frame,
      movement
    ))
  frame_index, video, movement = zip(*frames)
  frame_index = np.array(frame_index)
  video = np.array(video)
  movement = np.round(movement).astype(int)
  return (
    file,
    frame_index,
    video,
    movement
  )


def main():
  args = parser.parse_args()
  # note: we assume that all training videos are in the `dir`
  #  and every file has a unique name
  if args.out is None:
    args.out = f'{args.dir}.npz'
  # find video files
  video_files = listdir(args.dir)
  # optionally load existing data
  if args.append and path.isfile(args.out):
    print(f'Loading existing data from {args.out}')
    data = load_data(args.out)
    video_files = [file for file in video_files if file not in data]
  else:
    data = {}
  if args.num_workers > 0:
    with tqdm(total=len(video_files), desc='Preprocessing videos') as pbar:
      with mp.Pool(processes=args.num_workers) as pool:
        read_video_fn = functools.partial(read_video, config=args)
        for file, frame_index, video, movement in pool.imap_unordered(read_video_fn, video_files):
          pbar.update()
          data[file] = (frame_index, video, movement)
  else:
    for file in video_files:
      _, frame_index, video, movement = read_video(file, args, verbose=True)
      data[file] = (frame_index, video, movement)
  # save the archive
  print(f'Saving data to {args.out}')
  makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
  save_data(args.out, data)


if __name__ == '__main__':
  main()
