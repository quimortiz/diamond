import pathlib
from qutils import save_checkpoint, load_checkpoint, get_diffusion_batch
import torch
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

from torch.utils.data import DataLoader
import torchvision
import einops


import sys


sys.path.append("/home/quim/code/diffusion_planning_v2/src")
import qdataset

# load the world model.


path = "qoutput/2024-12-03/11-04-29_l75jja/ckpt/ckpt_00000027.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = load_checkpoint(path, device)
model = D["model"]
diffusion_sampler = DiffusionSampler(denoiser=model, cfg=D["diffusion_sampler_cfg"])

datapath = "/home/quim/code/diffusion_planning_v2/data/pusht_more_data/traj_v3/"

tgt_traj_len = 16
dataset = qdataset.DiskDatasetTrajv2(datapath, tgt_traj_len=tgt_traj_len)

batch_size = 4
dataloader = DataLoader(
    dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True
)


# simulate the next image of a batch.


batch = next(iter(dataloader))

batch_diffusion = get_diffusion_batch(
    batch, act_is_vel=True, act_is_vel_discretized=False, device=device
)

num_steps_conditioning = D["denoiser_cfg"].inner_model.num_steps_conditioning

out = diffusion_sampler.sample(
    prev_obs=batch_diffusion.obs[:, :num_steps_conditioning],
    prev_act=batch_diffusion.act[:, :num_steps_conditioning],
)


x = out[0]

x_01 = 0.5 * (x + 1)

real_imgs = 0.5 * (batch_diffusion.obs[:, num_steps_conditioning] + 1)
real_imgs = real_imgs.minimum(
    0.9 * torch.ones_like(real_imgs)
)  # we real images a bit darker

all_imgs = []
for img, real_img in zip(x_01, real_imgs):
    all_imgs.append(real_img)
    all_imgs.append(img)
all_imgs = torch.stack(all_imgs)


img_dir = pathlib.Path("tmp/play/")
img_dir.mkdir(parents=True, exist_ok=True)

step = 0
fout = img_dir / f"new_samples_{step:08d}.png"
torchvision.utils.save_image(all_imgs, fout, nrow=8)


# lets simulate a velocity!

num_steps = 20

prev_obs = batch_diffusion.obs[:, :num_steps_conditioning]
prev_act = batch_diffusion.act[:, :num_steps_conditioning]

vel = torch.ones(batch_size, 2).to(device) * 0.1

traj_list = []
traj_list_real = []

for i in range(num_steps):
    out = diffusion_sampler.sample(prev_obs=prev_obs, prev_act=prev_act)
    new_obs = out[0]
    prev_obs = torch.roll(prev_obs, shifts=-1, dims=1)
    prev_obs[:, -1] = new_obs
    prev_act = torch.roll(prev_act, shifts=-1, dims=1)
    prev_act[:, -1] = vel
    traj_list.append(0.5 * new_obs + 0.5)


traj = torch.stack(traj_list, dim=1)


imgs = einops.rearrange(traj, "b t ...  -> (b t) ...")

fout_long_rollout = img_dir / f"long_rollout_{step:08d}.png"

torchvision.utils.save_image(imgs, fout_long_rollout, nrow=num_steps)
