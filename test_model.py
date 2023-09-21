import argparse
from os import path

import torch

from model import MaskedVisionTransformer
from train_model import accuracy
from train_model import create_val_loader, run_inference
from utils.data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to data (.npz)')
parser.add_argument('--model', required=True, help='path to a model')
parser.add_argument('--out', default='.', help='output directory')
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load model
ckpt = torch.load(args.model, map_location=device)
config = ckpt['config']
model = MaskedVisionTransformer(config).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
# load data
data = load_data(args.data)
files = list(data.keys())
data = [(file, frame_index, video, movement) for file, (frame_index, video, movement) in data.items()]
dataloader = create_val_loader(data, config)
# predict where the exit is
targets, logits, targets_mask = run_inference(
  model, dataloader, device=device
)
thresholds = [15, 30, 45, 60, 90]
for threshold in thresholds:
  acc = accuracy(targets, logits, targets_mask, threshold=threshold)
  print(f'acc@{threshold}\u00B0 {acc:.2%}')
torch.save({
  'files': files,
  'targets': targets,
  'logits': logits,
  'mask': targets_mask
}, path.join(args.out, 'predictions.pt'))
