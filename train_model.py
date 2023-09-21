import argparse
import itertools
import math
import random
from datetime import datetime
from os import path, makedirs
from time import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader, TensorDataset

from model import MaskedVisionTransformer, Config
from utils.data import load_data, preprocess_video

# noinspection PyUnresolvedReferences
torch.backends.cuda.matmul.allow_tf32 = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('training_data', help='path to training data (.npz)')
parser.add_argument('--config', default='default', help='config name')
parser.add_argument('--out', default='.', help='output directory')
parser.add_argument('--val-size', type=float, default=None, help='single train-validation split')
parser.add_argument('--k-fold', type=int, default=None, help='k-fold cross-validation')
parser.add_argument('--seed', type=int, default=None, help='random state')
parser.add_argument('--silent', action='store_true', help='do not print messages')


def train_model(training_data, config, validation_data=None,
                device=None, verbose=False):
  assert len(training_data) > 0
  uses_cuda = device is not None and device.type == 'cuda'
  # prepare dataloaders
  training_data = create_train_loader(training_data, config)
  training_data = itertools.cycle(training_data)
  if uses_cuda:
    training_data = non_blocking_cuda_loader(training_data, device)
  if validation_data is not None:
    assert len(validation_data) > 0
    validation_data = create_val_loader(validation_data, config)
  # initialize the model
  model = MaskedVisionTransformer(config).to(device)
  optimizer = model.get_optimizer(fused=uses_cuda)
  # start training
  train_step, train_loss = Meter(), Meter()
  best_val_acc = 0.
  for step in range(config.steps):
    step_start = time()
    update_lr_with_cosine_schedule_(
      optimizer=optimizer,
      step=step,
      max_steps=config.steps,
      max_lr=config.learning_rate,
      min_lr=config.min_learning_rate,
      warmup_steps=config.warmup_steps
    )
    # video: (B, N, C, H, W); metadata: (B, N, F); targets: (B, N); targets_mask: (B, N)
    video, metadata, targets, targets_mask = next(training_data)
    logits, loss = model(
      video, metadata,
      targets=targets,
      targets_mask=targets_mask,
      drop_ratio=0.8
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_loss.update(loss.item())
    step_end = time()
    train_step.update(step_end - step_start)
    if (step + 1) % config.eval_interval == 0:
      if validation_data is not None:
        val_targets, val_logits, val_mask = run_inference(
          model, validation_data, device=device
        )
        val_acc = accuracy(
          val_targets,
          val_logits,
          val_mask
        )
      else:
        val_acc = torch.nan
      if val_acc > best_val_acc:
        best_val_acc = val_acc
        new_best_val_acc = True
      else:
        new_best_val_acc = False
      if verbose:
        print(f'[{datetime.now()}] [{step + 1:05d}] {"(*)" if new_best_val_acc else "   "} '
              f'train_step {train_step.value:.4f} train_loss {train_loss.value:.4f}',
              f'val_acc {val_acc:.2%}' if not np.isnan(val_acc) else '')
  return model


@torch.inference_mode()
def run_inference(model, dataloader, device=None):
  targets, logits, targets_mask = [], [], []
  model.eval()
  for video, metadata, batch_targets, batch_mask in dataloader:
    batch_logits = model(
      video.to(device),
      metadata.to(device),
      targets_mask=batch_mask.to(device)
    )
    targets.append(batch_targets.clone())
    logits.append(batch_logits.clone().cpu())
    targets_mask.append(batch_mask.clone())
  model.train()
  targets = torch.cat(targets)
  logits = torch.cat(logits)
  targets_mask = torch.cat(targets_mask)
  return targets, logits, targets_mask


def accuracy(targets, logits, targets_mask, threshold=45.0):
  assert targets.shape == logits.shape
  assert targets.ndim in (1, 2)
  if targets.ndim == 1:
    targets = targets.unsqueeze(dim=0)
    logits = logits.unsqueeze(dim=0)
    targets_mask = targets_mask.unsqueeze(dim=0)
  sum_of_accuracies = 0.
  for sample_targets, sample_logits, sample_mask in zip(targets, logits, targets_mask):
    sample_targets = torch.masked_select(sample_targets, sample_mask)
    sample_logits = torch.masked_select(sample_logits, sample_mask)
    angle_diff = torch.rad2deg(
      torch.arctan2(
        torch.sin(sample_targets - sample_logits),
        torch.cos(sample_targets - sample_logits)
      )
    )
    sum_of_accuracies += (angle_diff.abs() <= threshold).float().mean()
  return sum_of_accuracies / len(targets)


def update_lr_with_cosine_schedule_(
    optimizer,
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.,
    warmup_steps: int = 0
) -> None:
  if step < warmup_steps:
    lr = max_lr * step / warmup_steps
  elif step > max_steps:
    lr = min_lr
  else:  # cosine learning rate schedule
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps - 1)
    coefficient = 0.5 * (1. + math.cos(math.pi * decay_ratio))
    lr = min_lr + coefficient * (max_lr - min_lr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def create_train_loader(training_data, config):
  train_dataset = LayoutDataset(
    data=training_data,
    context_size=config.context_size,
    random_clip=True
  )
  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1,
    pin_memory=True
  )
  return train_loader


def create_val_loader(validation_data, config):
  val_dataset = LayoutDataset(
    data=validation_data,
    context_size=config.context_size,
  )
  val_dataset = TensorDataset(
    *map(torch.stack, zip(*val_dataset))
  )
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config.batch_size,
  )
  return val_loader


def non_blocking_cuda_loader(dataloader, device):
  def to_device(batch):
    return tuple(map(lambda x: x.to(device, non_blocking=True), batch))
  dataloader = map(to_device, dataloader)
  prefetched_batch = next(dataloader)
  for next_batch in dataloader:
    yield prefetched_batch
    prefetched_batch = next_batch
  yield prefetched_batch


class LayoutDataset(Dataset):
  def __init__(self, data, *, context_size, random_clip=False):
    self.data = data
    self.context_size = context_size
    self.random_clip = random_clip

  def __getitem__(self, index):
    _, _, video, movement = self.data[index]
    video, metadata, targets, targets_mask = preprocess_video(
      video, movement, context_size=self.context_size,
      random_clip=self.random_clip
    )
    video = torch.from_numpy(video).float()
    metadata = torch.from_numpy(metadata).float()
    targets = torch.from_numpy(targets).float()
    targets_mask = torch.from_numpy(targets_mask).bool()
    return video, metadata, targets, targets_mask

  def __len__(self):
    return len(self.data)


class Meter:
  def __init__(self, m=0.9):
    self.m = m
    self.value = None

  def update(self, item):
    if self.value is None:
      self.value = item
    else:
      self.value = self.m * self.value + (1 - self.m) * item
    return self.value


if __name__ == '__main__':
  args = parser.parse_args()
  if args.val_size and args.k_fold:
    raise ValueError('Choose either `val_size` or `k-fold`')
  verbose = not args.silent
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  makedirs(args.out, exist_ok=True)
  # load training data
  _, ext = path.splitext(args.training_data)
  if ext != '.npz':
    args.training_data = f'{args.training_data}.npz'
  if not path.isfile(args.training_data):
    raise ValueError(f'Could not find training data {args.training_data}')
  training_data = load_data(args.training_data)
  training_data = [
    (file, frame_index, video, movement)
    for file, (frame_index, video, movement) in training_data.items()
  ]
  # load config file
  if not path.isfile(args.config):
    # maybe config is the name of a default config file
    config_file = path.join(path.dirname(__file__), f'{args.config}.yml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file
  config = Config.from_file(args.config)
  # train (and validate) the model
  if args.k_fold:  # run k-fold cross-validation
    validation_files, val_targets, val_logits, val_mask = [], [], [], []
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    for i, (train_idx, val_idx) in enumerate(kf.split(training_data)):
      if verbose:
        print(f'   --- Fold {i+1:02d} ---')
      fold_training_data = [training_data[i] for i in train_idx]
      fold_validation_data = [training_data[i] for i in val_idx]
      model = train_model(
        training_data=fold_training_data,
        validation_data=fold_validation_data if verbose else None,
        config=config,
        device=device,
        verbose=verbose
      )
      fold_val_loader = create_val_loader(fold_validation_data, config)
      fold_val_targets, fold_val_logits, fold_val_mask = run_inference(
        model, fold_val_loader, device=device
      )
      torch.save({
        'model': model.state_dict(),
        'config': config
      }, path.join(args.out, f'checkpoint_{i+1:02d}.pt'))
      validation_files += [file for file, _, _, _ in fold_validation_data]
      val_targets.append(fold_val_targets)
      val_logits.append(fold_val_logits)
      val_mask.append(fold_val_mask)
    val_targets = torch.cat(val_targets)
    val_logits = torch.cat(val_logits)
    val_mask = torch.cat(val_mask)
  else:
    if args.val_size:
      if args.val_size >= 1:
        args.val_size = int(args.val_size)
      training_data, validation_data = train_test_split(
        training_data, test_size=args.val_size, random_state=args.seed, shuffle=True
      )
    else:
      validation_data = None
    model = train_model(
      training_data=training_data,
      validation_data=validation_data if verbose else None,
      config=config,
      device=device,
      verbose=verbose
    )
    if validation_data is not None:
      val_loader = create_val_loader(validation_data, config)
      validation_files = [file for file, _, _, _ in validation_data]
      val_targets, val_logits, val_mask = run_inference(
        model, val_loader, device=device
      )
    else:
      validation_files = val_logits = val_targets = val_mask = None
    torch.save({
      'model': model.state_dict(),
      'config': config
    }, path.join(args.out, 'checkpoint.pt'))
  # save validation results
  if validation_files:
    if verbose:
      thresholds = [15, 30, 45, 60, 90]
      for threshold in thresholds:
        val_acc = accuracy(
          val_targets,
          val_logits,
          val_mask,
          threshold=threshold
        )
        print(f'val_acc@{threshold}\u00B0 {val_acc:.2%}')
    torch.save({
      'files': validation_files,
      'targets': val_targets,
      'logits': val_logits,
      'mask': val_mask
    }, path.join(args.out, 'val_predictions.pt'))
