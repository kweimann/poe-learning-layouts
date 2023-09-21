import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm

from utils import transforms

mean = np.array([0.138, 0.119, 0.119])
std = np.array([0.083, 0.087, 0.100])


def preprocess_video(video, movement, *, context_size, center_crop=None,
                     frame_size=None, random_clip=False):
  video = video[:-1]  # discard exit frame (there is no target for it)
  # get angles
  origin = get_origin_angle(movement)[:-1]  # discard exit frame (there is no target for it)
  previous = get_movement_angle(movement)[:-1]  # discard exit frame (there is no target for it)
  targets = get_destination_angle(movement)
  # create metadata for each frame
  metadata = np.stack([
    np.cos(origin),
    np.sin(origin),
    np.cos(previous),
    np.sin(previous)
  ], axis=1)
  metadata = np.concatenate([  # prepend (empty) metadata in the first position
    np.zeros((1, metadata.shape[1]), dtype=metadata.dtype),
    metadata
  ], axis=0)
  assert len(video) == len(metadata) == len(targets)
  # clip video
  if len(targets) > context_size:
    if random_clip:
      offset = np.random.randint(len(targets) - context_size + 1)
    else:
      offset = 0
    video, metadata, targets = [
      x[offset:offset + context_size]
      for x in (video, metadata, targets)
    ]
  # center crop frames
  if center_crop is not None:
    video = transforms.center_crop(video, center_crop)
  # resize frames
  if frame_size is not None:
    video = transforms.resize(video, frame_size)
  # normalize frames
  video = video / 255.
  video = transforms.normalize(video, mean, std)
  # channels first
  video = np.transpose(video, (0, 3, 1, 2))
  # pad video
  targets_mask = np.ones(len(targets))
  if len(targets) < context_size:
    video, metadata, targets, targets_mask = [
      transforms.pad(x, context_size)
      for x in (video, metadata, targets, targets_mask)
    ]
  return video, metadata, targets, targets_mask


def save_data(file, data):
  files = np.array(list(data.keys()))
  frame_index, videos, movement = zip(*data.values())
  video_length = np.array([len(video) for video in videos])
  frame_index, videos, movement = map(np.concatenate, (frame_index, videos, movement))
  np.savez_compressed(
    file,
    files=files,
    frame_index=frame_index,
    video=videos,
    movement=movement,
    video_length=video_length
  )


def load_data(file):
  archive = np.load(file)
  files, frame_index, video, movement, video_length = (
    archive['files'], archive['frame_index'], archive['video'],
    archive['movement'], archive['video_length']
  )
  data = {}
  video_end = np.cumsum(video_length)
  video_start = video_end - video_length
  for file, start, end in zip(files, video_start, video_end):
    assert file not in data  # filenames are unique
    data[file] = (
      frame_index[start:end],
      video[start:end],
      movement[start:end]
    )
  return data


def draw_minimap(video, movement):
  current_position = movement.cumsum(axis=0)
  x_min, y_min = current_position.min(axis=0)
  x_max, y_max = current_position.max(axis=0)
  assert (x_min <= 0) and (x_max >= 0)
  assert (y_min <= 0) and (y_max >= 0)
  (x_start, y_start) = abs(x_min), abs(y_min)
  _, H_frame, W_frame, C = video.shape  # frame size
  (H, W) = (y_max - y_min + H_frame, x_max - x_min + W_frame)  # map size
  H += H % 2  # make height divisible by 2
  W += W % 2  # make width divisible by 2
  minimap = np.zeros((H, W, C), dtype=np.uint8)
  for frame, (x, y) in zip(video, current_position):
    minimap[y_start + y:y_start + y + H_frame, x_start + x:x_start + x + W_frame] = frame
  return minimap


def draw_arrows(minimap, angles, movement, frame_size, c=(0, 0, 0)):
  current_position = movement.cumsum(axis=0)
  x_min, y_min = current_position.min(axis=0)
  (x_start, y_start) = abs(x_min), abs(y_min)
  R = 25  # we assume that frame size is 240x240 or larger
  for (x, y), angle in zip(current_position, angles):
    (center_x, center_y) = (
      x_start + x + frame_size // 2,
      y_start + y + frame_size // 2
    )
    (dx, dy) = np.array((
      R * np.cos(angle),
      R * np.sin(-angle)
    )).round().astype(int)
    cv2.arrowedLine(
      minimap,
      (center_x, center_y),
      (center_x + dx, center_y + dy),
      color=c,
      thickness=4,
      tipLength=0.4
    )


def stream_video(file,
                 transform=None,
                 clip_movement=None,
                 translation_threshold=0.7,
                 translation_max_matches=15,
                 verbose=False):
  if transform is None:
    def transform(frame): return frame
  metadata = skvideo.io.ffprobe(file)
  num_frames = int(metadata['video']['@nb_frames'])
  assert num_frames > 0
  video = skvideo.io.vreader(file, num_frames=num_frames)
  last_yielded = None
  last = (0, next(video), np.array([0., 0.]))  # (frame_index, frame, movement)
  current_frame = next(video, None)
  frame_index = 1
  with tqdm(total=num_frames, desc=f'Streaming {file}', disable=not verbose) as pbar:
    while current_frame is not None:
      pbar.update(frame_index - pbar.n)
      _, last_frame, _ = last
      movement = find_translation(
        transform(last_frame),
        transform(current_frame),
        threshold=translation_threshold,
        max_matches=translation_max_matches
      )
      if movement is not None:
        # usually large movement indicates that the algorithm badly matched the frames;
        #  therefore, we clip the movement
        if clip_movement is not None:
          movement = np.clip(movement, -clip_movement, clip_movement)
        yield last
        last_yielded = last
        last = (frame_index, current_frame, movement)
        current_frame = next(video, None)
        frame_index += 1
      elif last_yielded is not None:  # try last yielded frame on the first mismatch
        last = last_yielded
        last_yielded = None
      else:  # try next frame
        current_frame = next(video, None)
        frame_index += 1
    pbar.update()
  yield last


def drop_frames(video_stream, movement_per_frame):  # minimal movement required to jump to next frame
  yield next(video_stream)
  frame_index, frame, total_movement = None, None, (0, 0)
  for frame_index, frame, movement in video_stream:
    new_total_movement = (dx, dy) = total_movement + movement
    if abs(dx) >= movement_per_frame or abs(dy) >= movement_per_frame:
      yield frame_index, frame, new_total_movement
      total_movement = (0, 0)
    else:
      total_movement = new_total_movement
  if np.any(total_movement != 0):
    yield frame_index, frame, total_movement


def get_destination_angle(movement):
  # at which angle from the current frame is the exit
  # note: last frame (exit) is discarded
  current_position = movement.cumsum(axis=0)
  current_position, end_position = current_position[:-1], current_position[-1]
  # computes the angle between current position and the exit
  xy_diff = end_position - current_position
  angles = np.arctan2(-xy_diff[:, 1], xy_diff[:, 0])
  return angles


def get_origin_angle(movement):
  # at which angle from the current frame was the entrance
  # note: first frame (entrance) is discarded
  current_position = movement.cumsum(axis=0)
  start_position, current_position = current_position[0], current_position[1:]
  # computes the angle between the entrance and current position
  xy_diff = start_position - current_position
  angles = np.arctan2(-xy_diff[:, 1], xy_diff[:, 0])
  return angles


def get_movement_angle(movement):
  # at which angle from the current frame was previous frame
  # note: first frame (entrance) is discarded
  frame_position = movement.cumsum(axis=0)
  prev_position, current_position = frame_position[:-1], frame_position[1:]
  # computes the angle between previous and current position
  xy_diff = current_position - prev_position
  angles = np.arctan2(-xy_diff[:, 1], xy_diff[:, 0])
  return angles


def find_translation(image_a, image_b, threshold=0.7, max_matches=15):
  # determine the (dx, dy) vector required to move from image_a to image_b
  # noinspection PyUnresolvedReferences
  sift = cv2.SIFT_create()
  # convert images to grayscale
  image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
  image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
  # find SIFT features
  keypoints_a, descriptors_a = sift.detectAndCompute(image_a_gray, None)
  keypoints_b, descriptors_b = sift.detectAndCompute(image_b_gray, None)
  if len(keypoints_a) == 0 or len(keypoints_b) == 0:
    return None
  # match the SIFT features
  matcher = cv2.BFMatcher(crossCheck=False)
  matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
  # filter bad matches and sort
  matches = [
    match[0] for match in matches
    if len(match) == 2 and match[0].distance < threshold * match[1].distance
  ]
  if len(matches) == 0:
    return None  # failed to find translation
  matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
  # select the matching points
  points_a = np.array([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 2)
  points_b = np.array([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 2)
  # get the mean translation
  dx, dy = np.median(points_a - points_b, axis=0)
  return dx, dy
