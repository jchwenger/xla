import args_parse

FLAGS = args_parse.parse_common_options(
    datadir="/tmp/mingpt-data",
    num_cores=8,
    batch_size=16,
    momentum=0.5,
    lr=6e-4,
    num_epochs=18,
)

FLAGS.lr_decay = True
FLAGS.block_size = 64
FLAGS.vocab_size = 256
FLAGS.n_embd = 128
FLAGS.n_layer = 4
FLAGS.n_head = 4
FLAGS.embd_pdrop = 0.1
FLAGS.resid_pdrop = 0.1
FLAGS.attn_pdrop = 0.1
FLAGS.betas = [0.9, 0.95]
FLAGS.grad_norm_clip = 1
FLAGS.warmup_tokens = 20000
FLAGS.final_tokens = 260e9
FLAGS.weight_decay = 0.1
FLAGS.train_test_split = 100

import os
import sys
import glob
import math
import shutil
import psutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset

import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# --------------------------------------------------------------------------------
# ByteLevel Dataset


class BytesDataset(Dataset):

  def __init__(self, data, input_path, block_size):
    self.input_path = input_path
    self.bytes = data.encode("utf-8")
    self.block_size = block_size
    self.vocab_size = 256
    self.stoi = {"bytes": True}  # for saving purposes

  def reload(data):
    self.bytes = load_files(files).encode("utf-8")

  @staticmethod
  def encode(text):
    return [b for b in text.encode("utf-8")]

  @staticmethod
  def decode(data, errors="strict"):
    return bytes(data).decode(errors=errors)

  def __len__(self):
    return len(self.bytes) - self.block_size

  def __getitem__(self, idx):
    chunk = [b for b in self.bytes[idx:idx + self.block_size + 1]]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y


# --------------------------------------------------------------------------------
# Karpathy's MinGPT


class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(config.n_embd, config.n_embd)
    self.query = nn.Linear(config.n_embd, config.n_embd)
    self.value = nn.Linear(config.n_embd, config.n_embd)
    # regularization
    self.attn_drop = nn.Dropout(config.attn_pdrop)
    self.resid_drop = nn.Dropout(config.resid_pdrop)
    # output projection
    self.proj = nn.Linear(config.n_embd, config.n_embd)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer(
        "mask",
        torch.tril(torch.ones(config.block_size,
                              config.block_size)).view(1, 1, config.block_size,
                                                       config.block_size),
    )
    self.n_head = config.n_head

  def forward(self, x, layer_past=None):
    B, T, C = x.size()

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # (B, nh, T, hs)
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    # re-assemble all head outputs side by side
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    # output projection
    y = self.resid_drop(self.proj(y))
    return y


class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.mlp = nn.Sequential(
        nn.Linear(config.n_embd, 4 * config.n_embd),
        nn.GELU(),
        nn.Linear(4 * config.n_embd, config.n_embd),
        nn.Dropout(config.resid_pdrop),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config

    # input embedding stem
    self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
    self.pos_emb = nn.Parameter(
        torch.zeros(1, config.block_size, config.n_embd))
    self.drop = nn.Dropout(config.embd_pdrop)
    # transformer
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.block_size = config.block_size
    self.apply(self._init_weights)

    xm.master_print(
        f"number of parameters: {sum(p.numel() for p in self.parameters()):,}")

  def save(self, model_dir, model_name):
    self.config.save(model_dir, model_name)

  def log(self):
    xm.master_print("-" * 40)
    msg = "Model's state_dict:"
    xm.master_print(msg)
    xm.master_print("-" * len(msg))
    longest = len(max(self.state_dict().keys(), key=len))
    for param_tensor in self.state_dict():
      xm.master_print(
          f"{param_tensor:{longest}} {list(self.state_dict()[param_tensor].size())}"
      )

  def to_file(self, model_dir, model_name):
    fname = "innards.log"
    msg = f"Saving model's state_dict to {model_dir}/{model_name}/{fname}"
    xm.master_print("-" * 40)
    xm.master_print(msg)
    xm.master_print("-" * 40)
    longest = len(max(self.state_dict().keys(), key=len))
    if not os.path.isdir(os.path.join(model_dir, model_name)):
      os.makedirs(os.path.join(model_dir, model_name))
    with open(os.path.join(model_dir, model_name, fname), "w") as o:
      for param_tensor in self.state_dict():
        o.write(
            f"{param_tensor:{longest}} {list(self.state_dict()[param_tensor].size())}\n"
        )

  def get_block_size(self):
    return self.block_size

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

  def configure_optimizers(self, train_config):

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    # mn: module name, m: module object
    for mn, m in self.named_modules():
      # pn: parameters name, p: parameters
      for pn, p in m.named_parameters():
        fpn = f"{mn}.{pn}" if mn else pn  # full param name

        if pn.endswith("bias"):
          # all biases will not be decayed
          no_decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
          # weights of whitelist modules will be weight decayed
          decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
          # weights of blacklist modules will NOT be weight decayed
          no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add("pos_emb")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=train_config.lr, betas=train_config.betas)
    return optimizer

  def forward(self, idx, targets=None):
    b, t = idx.size()
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."

    # forward the GPT model
    token_embeddings = self.tok_emb(
        idx)  # each index maps to a (learnable) vector
    # each position maps to a (learnable) vector
    position_embeddings = self.pos_emb[:, :t, :]
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x)

    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss


# --------------------------------------------------------------------------------
# utils


# from OpenAI's GPT-2
def load_files(path):
  paths = []
  if os.path.isfile(path):
    # Simple file
    paths.append(path)
  elif os.path.isdir(path):
    # Directory
    for (dirpath, _, fnames) in os.walk(path):
      for fname in fnames:
        paths.append(os.path.join(dirpath, fname))
  else:
    # Assume glob
    paths = glob.glob(path)

  raw_text = ""
  random.shuffle(paths)
  total = len(paths)
  # xm.master_print(f"found {total} files")
  for i, path in enumerate(paths):
    # xm.master_print(f"{i} | used mem: {psutil.virtual_memory().percent}% | {path}")
    # sys.stdout.write("\033[K\033[F")
    # Plain text
    if sys.getsizeof(raw_text) > psutil.virtual_memory().available / 1000:
      # xm.master_print(
      #     f"\nfiles loaded: {i}/{total} | used mem: {psutil.virtual_memory().percent}%"
      # )
      break
    try:
      with open(path, "r", encoding="utf-8") as fp:
        raw_text += fp.read()
    except UnicodeDecodeError as e:
      xm.master_print(f"\e[31m{e}")
      xm.master_print(f"file: {path}\e[0m")
      # sys.exit()

  return raw_text


def split_dataset(dataset, divisor=100, seed=42):
  ds_len = len(dataset)
  # xm.master_print("before random split, len:", ds_len)
  test_len = ds_len // divisor
  test, train = torch.utils.data.random_split(
      dataset,
      [test_len, ds_len - test_len],
      generator=torch.Generator().manual_seed(seed),
  )
  return {
      "train_dataset": train,
      "test_dataset": test,
      "train_test_split": divisor,
  }


def _train_update(device, x, loss, tracker, writer):
  test_utils.print_training_update(
      device,
      x,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      summary_writer=writer,
  )


def train_mingpt(flags, **kwargs):
  torch.manual_seed(1)

  if flags.fake_data:
    # raise NotImplementedError("Fake data not implemented yet.")
    train_loader = xu.SampleGenerator(
        data=(
            torch.zeros(flags.batch_size, flags.block_size, dtype=torch.int64),
            torch.zeros(flags.batch_size, flags.block_size, dtype=torch.int64),
        ),
        sample_count=60000 // flags.batch_size // xm.xrt_world_size(),
    )
    test_loader = xu.SampleGenerator(
        data=(
            torch.zeros(flags.batch_size, flags.block_size, dtype=torch.int64),
            torch.zeros(flags.batch_size, flags.block_size, dtype=torch.int64),
        ),
        sample_count=10000 // flags.batch_size // xm.xrt_world_size(),
    )
  else:

    data = split_dataset(
        BytesDataset(
            load_files(flags.datadir), flags.datadir, flags.block_size),
        divisor=flags.train_test_split,
    )
    train_dataset, test_dataset = data["train_dataset"], data["test_dataset"]

    train_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True,
      )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags.batch_size,
        sampler=train_sampler,
        drop_last=flags.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=flags.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags.batch_size,
        drop_last=flags.drop_last,
        shuffle=False,
        num_workers=flags.num_workers,
    )

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  device = xm.xla_device()
  model = GPT(flags).to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = model.configure_optimizers(flags)

  best_loss = float("inf")
  tokens = 0  # counter used for learning rate decay

  def train_loop_fn(loader):
    tracker = xm.RateTracker()
    model.train()
    tokens = 0
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      logits, loss = model(data, target)
      loss.backward()
      xm.optimizer_step(optimizer)

      # decay the learning rate based on our progress
      if flags.lr_decay:
        # number of tokens processed this step
        tokens += ((flags.block_size + 1) * flags.batch_size *
                   xm.xrt_world_size())
        if tokens < flags.warmup_tokens:
          # linear warmup
          lr_mult = float(tokens) / float(max(1, flags.warmup_tokens))
        else:
          # cosine learning rate decay
          progress = float(tokens - flags.warmup_tokens) / float(
              max(1, flags.final_tokens - flags.warmup_tokens))
          lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        lr = flags.learning_rate * lr_mult
        for param_group in optimizer.param_groups:
          param_group["lr"] = lr
      else:
        lr = flags.learning_rate

      tracker.add(flags.batch_size)
      if step % flags.log_steps == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, writer))

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    losses = []
    for data, target in loader:
      logits, loss = model(data, target)
    test_loss = float(np.mean(losses))
    test_loss = xm.mesh_reduce("test_loss", test_loss, np.mean)
    return test_loss

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  test_loss, min_test_loss = 0.0, 0.0
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print("Epoch {} train begin {}".format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader)
    xm.master_print("Epoch {} train end {}".format(epoch, test_utils.now()))

    test_loss = test_loop_fn(test_device_loader)
    ppl = math.exp(test_loss)
    xm.master_print(
        "Epoch {} test end {}, Test Loss={:.2f}, Perplexity={:.2f}".format(
            epoch, test_utils.now(), test_loss, ppl))
    min_test_loss = min(test_loss, min_test_loss)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={
            "Test Loss/test": test_loss,
            "Perplexity/test": math.exp(test_loss),
        },
        write_xla_metrics=True,
    )
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print("Min Loss: {:.2f}, Min Perplexity: {:.2f}".format(
      min_test_loss, math.exp(min_test_loss)))
  return min_test_loss


def _mp_fn(index, flags):
  # torch.set_default_tensor_type("torch.FloatTensor")
  # try:
  loss = train_mingpt(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if loss < flags.target_loss:
    print("Loss {} is below target {}. Perplexity: ".format(
        loss, math.exp(loss)))
    sys.exit(21)
  # except:
  #   print(met.metrics_report())


if __name__ == "__main__":
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)

