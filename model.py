import math
from dataclasses import dataclass

import torch
import yaml
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
  # model
  blocks: tuple[int]
  channels: tuple[int]
  stride: tuple[int]
  context_size: int
  dim: int
  depth: int
  num_heads: int
  bias: bool
  mlp_ratio: int
  dropout: float
  attn_dropout: float
  # training
  steps: int
  eval_interval: int
  batch_size: int
  learning_rate: float
  min_learning_rate: float
  weight_decay: float
  warmup_steps: int

  @classmethod
  def from_file(cls, file, **kwargs):
    with open(file) as fh:
      kwargs_from_file = yaml.safe_load(fh)
    kwargs = {**kwargs_from_file, **kwargs}  # override file config with kwargs
    return cls(**kwargs)


class MaskedVisionTransformer(nn.Module):
  num_meta_features = 4
  num_outputs = 1

  def __init__(self, config: Config, flash_attn=True):
    super().__init__()
    self.config = config
    self.frame_encoder = FrameEncoder(config)
    # self.frame_embed = nn.Linear(config.channels[-1], config.dim, bias=config.bias)
    self.meta_embed = nn.Linear(self.num_meta_features, config.dim, bias=True)
    self.pos_embed = nn.Parameter(get_1d_pos_embed(config.context_size, config.dim), requires_grad=False)
    self.dropout = nn.Dropout(p=config.dropout)
    self.blocks = nn.ModuleList([
      TransformerBlock(
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        bias=config.bias,
        flash_attn=flash_attn
      ) for _ in range(config.depth)
    ])
    self.norm = LayerNorm(config.dim, bias=config.bias)
    self.fc = nn.Linear(config.dim, self.num_outputs, bias=True)
    self.mask_token = nn.Parameter(torch.zeros(1, config.dim))
    nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)
    self.init_weights()

  def forward(self, video, metadata, targets=None, targets_mask=None, drop_ratio=0.):
    B, N, C, H, W = video.shape
    if targets_mask is None:
      targets_mask = torch.ones(B, N, dtype=torch.bool, device=video.device)
    if drop_ratio > 0:
      targets_mask = torch.bernoulli((1 - drop_ratio) * targets_mask.float()).bool()
    keep_ids = targets_mask.flatten().nonzero().squeeze(-1)
    drop_ids = (~targets_mask).flatten().nonzero().squeeze(-1)
    restore_ids = torch.argsort(torch.cat([keep_ids, drop_ids]))
    video = video.view(B * N, C, H, W)[keep_ids]
    video = self.frame_encoder(video)
    # video = self.frame_embed(video)
    metadata = metadata.view(B * N, -1)[keep_ids]
    metadata = self.meta_embed(metadata)
    video = video + metadata
    video = torch.cat([video, self.mask_token.repeat(len(drop_ids), 1)])
    video = torch.gather(video, dim=0, index=restore_ids.unsqueeze(-1).repeat(1, self.config.dim))
    video = video.view(B, N, self.config.dim)
    video = video + self.pos_embed[:, :N]
    video = self.dropout(video)
    for block in self.blocks:
      video = block(video)
    video = self.norm(video)
    logits = self.fc(video).squeeze(dim=-1)
    if targets is not None:
      logits_ = torch.masked_select(logits, targets_mask)
      targets_ = torch.masked_select(targets, targets_mask)
      loss = (
          F.mse_loss(logits_.sin(), targets_.sin()) +
          F.mse_loss(logits_.cos(), targets_.cos())
      )
      return logits, loss
    return logits

  def get_optimizer(self, fused=False):
    decay = set()
    decay_modules = (nn.Linear, nn.Conv2d)
    for module_name, module in self.named_modules():
      for param_name, param in module.named_parameters():
        param_name = f'{module_name}.{param_name}' if module_name else param_name
        if isinstance(module, decay_modules) and param_name.endswith('weight'):
          decay.add(param_name)
    params = dict(self.named_parameters())
    optim_groups = [
      {'params': [param for name, param in params.items() if name in decay], 'weight_decay': self.config.weight_decay},
      {'params': [param for name, param in params.items() if name not in decay], 'weight_decay': 0.}
    ]
    optimizer = torch.optim.AdamW(
      params=optim_groups,
      lr=self.config.learning_rate,
      fused=fused)
    return optimizer

  def init_weights(self):
    for name, module in self.named_modules():
      if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, (GroupNorm, LayerNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)


def get_1d_pos_embed(context_size, dim):
  assert dim % 2 == 0
  position = torch.arange(context_size).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
  pos_embed = torch.zeros(1, context_size, dim)
  pos_embed[0, :, 0::2] = torch.sin(position * div_term)
  pos_embed[0, :, 1::2] = torch.cos(position * div_term)
  return pos_embed


class CausalSelfAttention(nn.Module):
  def __init__(self, dim, num_heads, dropout=0.,
               attn_dropout=0., bias=True, flash_attn=True):
    super().__init__()
    assert dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.attn_dropout = attn_dropout
    self.flash_attn = flash_attn
    self.W_qkv = nn.Linear(dim, 3 * dim, bias=bias)
    self.W_o = nn.Linear(dim, dim, bias=bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, N, D = x.size()
    qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if self.flash_attn:
      attn = F.scaled_dot_product_attention(
        q, k, v, dropout_p=self.attn_dropout, is_causal=True)
    else:  # use standard implementation of attention to get attention weights for visualization
      attn_mask = torch.ones(N, N, device=q.device).tril(diagonal=0)
      attn_mask = attn_mask.masked_fill(attn_mask == 0, -float('inf'))
      attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
      attn_weight = torch.dropout(attn_weight, self.attn_dropout, train=self.training)
      attn = attn_weight @ v
      self.attn_weights = attn_weight
    attn = attn.transpose(1, 2).reshape(B, N, D)
    attn = self.dropout(self.W_o(attn))
    return attn


class MLP(nn.Module):
  def __init__(self, dim, mlp_ratio, dropout=0., bias=True):
    super().__init__()
    self.W_1 = nn.Linear(dim, mlp_ratio * dim, bias=bias)
    self.W_2 = nn.Linear(mlp_ratio * dim, dim, bias=bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.W_2(F.gelu(self.W_1(x))))


class LayerNorm(nn.LayerNorm):
  # same as nn.LayerNorm, but allows setting bias to None
  def __init__(self, *args, **kwargs):
    bias = kwargs.pop('bias', True)
    super().__init__(*args, **kwargs)
    if not bias:
      self.register_parameter('bias', None)


class TransformerBlock(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio, dropout=0.,
               attn_dropout=0., bias=True, flash_attn=True):
    super().__init__()
    self.mha_norm = LayerNorm(dim, bias=bias)
    self.mha = CausalSelfAttention(
      dim=dim,
      num_heads=num_heads,
      dropout=dropout,
      attn_dropout=attn_dropout,
      bias=bias,
      flash_attn=flash_attn
    )
    self.mlp_norm = LayerNorm(dim, bias=bias)
    self.mlp = MLP(
      dim=dim,
      mlp_ratio=mlp_ratio,
      dropout=dropout,
      bias=bias
    )

  def forward(self, x):
    # pre-norm residual units as per: https://arxiv.org/abs/1906.01787
    x = x + self.mha(self.mha_norm(x))
    x = x + self.mlp(self.mlp_norm(x))
    return x


class FrameEncoder(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.patch_dropout = PatchDropout(patch_size=16, p=config.dropout)
    in_channels = out_channels = config.channels[0]
    self.conv1 = conv2d(
      in_channels=3,
      out_channels=out_channels,
      kernel_size=7,
      stride=2,
      bias=config.bias
    )
    self.blocks = []
    for out_channels, num_blocks, stride in zip(config.channels, config.blocks, config.stride):
      self.blocks.append(ResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        kernel_size=3,
        bias=config.bias
      ))
      for _ in range(num_blocks - 1):
        self.blocks.append(ResidualBlock(
          in_channels=out_channels,
          out_channels=out_channels,
          kernel_size=3,
          bias=config.bias
        ))
      in_channels = out_channels
    self.blocks = nn.ModuleList(self.blocks)
    self.norm1 = norm2d(out_channels, bias=config.bias)
    self.relu1 = nn.ReLU()
    self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

  def forward(self, x):
    x = self.patch_dropout(x)
    x = self.conv1(x)
    for block in self.blocks:
      x = block(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.global_pool(x).squeeze(dim=(-1, -2))
    return x


class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
    super().__init__()
    if stride > 1 or in_channels != out_channels:
      self.shortcut = conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
    else:
      self.shortcut = nn.Identity()
    self.norm1 = norm2d(in_channels, bias=bias)
    self.relu1 = nn.ReLU()
    self.conv1 = conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    self.norm2 = norm2d(out_channels, bias=bias)
    self.relu2 = nn.ReLU()
    self.conv2 = conv2d(out_channels, out_channels, kernel_size, bias=bias)

  def forward(self, x):
    shortcut = self.shortcut(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.relu2(x)
    x = self.conv2(x)
    out = x + shortcut
    return out


class PatchDropout(nn.Module):
  def __init__(self, patch_size, p=0.5):
    super().__init__()
    self.patch_size = patch_size
    self.p = p

  def forward(self, x):
    if self.training and self.p > 0:
      mask = drop_patches(x, self.patch_size, self.p)
      return x * mask * (1.0 / (1 - self.p))
    return x


def drop_patches(x, patch_size, drop_ratio):
  B, C, H, W = x.size()
  assert H == W
  assert H % patch_size == 0
  keep_ratio = 1 - drop_ratio
  num_patches = H // patch_size
  patches = torch.bernoulli(  # randomly drop patches
    torch.full((B, 1, num_patches, 1, num_patches, 1),
               fill_value=keep_ratio, device=x.device)
  )
  patches = patches.repeat(1, 1, 1, patch_size, 1, patch_size)  # upscale patches
  mask = patches.reshape(B, 1, H, W)  # reshape the patches into a mask
  return mask


class GroupNorm(nn.GroupNorm):
  # same as nn.GroupNorm, but allows setting bias to None
  def __init__(self, *args, **kwargs):
    bias = kwargs.pop('bias', True)
    super().__init__(*args, **kwargs)
    if not bias:
      self.register_parameter('bias', None)


def conv2d(in_channels, out_channels, kernel_size, stride=1, bias=True, **kwargs):
  assert kernel_size % 2 == 1  # for simplicity
  padding = (kernel_size - 1) // 2
  return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs)


def norm2d(in_channels, num_groups=None, bias=True):
  # replace batch normalization with group normalization as per: https://arxiv.org/abs/2003.00295
  if num_groups is None:
    num_groups = in_channels // 16
  return GroupNorm(num_groups, in_channels, bias=bias)
