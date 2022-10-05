from hybrik.datasets import MixDataset, PW3D, H36MDataset

# Docker
config_file = './configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_single_nc_test.yaml'

import yaml
from easydict import EasyDict as edict

import numpy as np
import random
import torch
import torch.profiler
def _init_fn(worker_id):
  np.random.seed(33)
  random.seed(33)

with open(config_file) as f:
  cfg = edict(yaml.load(f, Loader=yaml.FullLoader))


train_dataset = H36MDataset(cfg=cfg,train=True)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=_init_fn, pin_memory=False)


with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(
          wait=1,
          warmup=1,
          active=2),
      on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
      record_shapes=True,
      profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
      with_stack=True
  ) as p:
    for step, data in enumerate(train_loader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device="cuda"), data[1].to(device="cuda")
        if step + 1 >= 4:
            break
        p.step()