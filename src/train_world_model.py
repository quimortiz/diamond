import torch
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
import torchvision
from data import Batch

from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from torch.utils.data import DataLoader
import einops
from collections import deque
import utils
import numpy as np


import sys

sys.path.append("/home/quim/code/diffusion_planning_v2/src")

import qdataset
import diff_utils
import pathlib
import matplotlib.pyplot as plt

rand_id = diff_utils.get_unique_id()
timestamp = diff_utils.get_time_stamp()

output_dir = pathlib.Path(f"qoutput/{timestamp}_{rand_id}")
output_dir.mkdir(parents=True, exist_ok=True)

img_dir = output_dir / "img"
ckpt_dir = output_dir / "ckpt"

img_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

num_steps_conditioning = 2
num_autoregressive_steps = 8
num_actions = 2

act_is_vel = True
act_is_vel_discretized = False

inner_model_config = InnerModelConfig(
    img_channels=3,
    num_steps_conditioning=num_steps_conditioning,  # go to 2?
    cond_channels=256,
    # TODO: i have reduced w.r.t yaml!
    depths=[2, 2, 2, 2],
    channels=[64, 64, 64, 64],
    attn_depths=[0, 0, 0, 0],
    # depths=[2, 2, 2],
    # channels=[64, 64, 64],
    # attn_depths=[0, 0, 0],
    num_actions=num_actions,
    num_hidden_layers_action=0,
)


denoiser_config = DenoiserConfig(
    inner_model=inner_model_config,
    sigma_data=0.5,
    sigma_offset_noise=0.3,
)

sigma_distribution_config = SigmaDistributionConfig(
    loc=-0.4,
    scale=1.2,
    sigma_min=2e-3,
    sigma_max=20,
)

diffusion_sampler_cfg = DiffusionSamplerConfig(
    num_steps_denoising=3,
    sigma_min=2e-3,
    sigma_max=5.0,
    rho=7,
    order=1,  # 1: Euler, 2: Heun
    s_churn=0.0,  # Amount of stochasticity
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
)

diffusion_sampler_cfg_v2 = DiffusionSamplerConfig(
    num_steps_denoising=6,
    sigma_min=2e-3,
    sigma_max=5.0,
    rho=7,
    order=1,  # 1: Euler, 2: Heun
    s_churn=0.0,  # Amount of stochasticity
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
)


device = torch.device("cuda:0")

model = Denoiser(denoiser_config)
model.to(device)

batch_size = 32
batch = Batch(
    obs=torch.rand(batch_size, 6, 3, 64, 64).to(device),
    act=torch.zeros(batch_size, 6, num_actions).to(device),
    mask_padding=torch.ones(batch_size, 6, dtype=torch.long).to(device),
    trunc=torch.zeros(batch_size, 6).to(device),
    rew=torch.zeros(batch_size, 6).to(device),
    end=torch.zeros(batch_size, 6).to(device),
    info=[],
    segment_ids=[],
)


model.setup_training(sigma_distribution_config)


loss, metric = model(batch)

# print("loss", loss)
# print("metric", metric)


diffusion_sampler = DiffusionSampler(denoiser=model, cfg=diffusion_sampler_cfg)
diffusion_sampler_v2 = DiffusionSampler(denoiser=model, cfg=diffusion_sampler_cfg_v2)


out = diffusion_sampler.sample(
    prev_obs=batch.obs[:, :num_steps_conditioning],
    prev_act=batch.act[:, :num_steps_conditioning],
)

# print("out")
# print(type(out))
# x = out[0]
# print("max x ", x.max())
# print("min x", x.min())


# lets train the model.

# load the data

visualize_data = False

datapath = "/home/quim/code/diffusion_planning_v2/data/pusht_more_data/traj_v3/"

tgt_traj_len = num_steps_conditioning + 1 + num_autoregressive_steps
batch_size = 16

dataset = qdataset.DiskDatasetTrajv2(datapath, tgt_traj_len=tgt_traj_len)


# def add_action(img, act):
#     img_out = torch.clone(img)

#     x_cord = int(act[1] / 512 * 64)
#     y_cord = int(act[0] / 512 * 64)
#     img_out[:, x_cord, y_cord] = 0

#     return img_out


def add_action(img, act, cross_size=2, color=0):
    """
    Adds a cross to the image at the location specified by the action.

    Parameters:
    - img (torch.Tensor): The input image tensor with shape (C, H, W).
    - act (iterable): The action coordinates, where act[0] is y and act[1] is x.
    - cross_size (int): The size of the cross arms extending from the center.
    - color (int or float): The value to set for the cross pixels.

    Returns:
    - torch.Tensor: The modified image with a cross added.
    """
    img_out = img.clone()

    # Map action coordinates from [0, 512] to [0, 64]
    y_cord = int(act[1] / 512 * 64)
    x_cord = int(act[0] / 512 * 64)

    # Ensure coordinates are within image bounds
    x_cord = max(0, min(x_cord, img_out.shape[2] - 1))
    y_cord = max(0, min(y_cord, img_out.shape[1] - 1))

    # Draw horizontal line of the cross
    x_start = max(x_cord - cross_size, 0)
    x_end = min(x_cord + cross_size + 1, img_out.shape[2])
    img_out[:, y_cord, x_start:x_end] = color

    # Draw vertical line of the cross
    y_start = max(y_cord - cross_size, 0)
    y_end = min(y_cord + cross_size + 1, img_out.shape[1])
    img_out[:, y_start:y_end, x_cord] = color

    return img_out


if visualize_data:

    first_image = dataset[0]["observation.image"][0]

    fout = "/home/quim/code/diamond/fig_state.png"

    torchvision.utils.save_image(first_image, fout)

    dir_tmp = pathlib.Path("/home/quim/code/diamond/tmp/")
    dir_tmp.mkdir(parents=True, exist_ok=True)

    # generate a new image with the action

    traj_id = 1
    first_image = dataset[traj_id]["observation.image"][0]
    first_action = dataset[traj_id]["action"][0]

    first_image_w_action = add_action(first_image, first_action)

    fout = dir_tmp / "image_w_action.png"

    torchvision.utils.save_image(first_image_w_action, fout)

    img_w_action = []
    for img, act in zip(
        dataset[traj_id]["observation.image"], dataset[traj_id]["action"]
    ):
        img_w_action.append(add_action(img, act))

    fout = dir_tmp / "image_w_action_traj.png"

    torchvision.utils.save_image(img_w_action, fout)

    sys.exit()

    # for i in range(tgt_traj_len):

    #     fig, ax = plt.subplots(2, 2)
    #     first_image = dataset[1]["observation.image"][i]

    #     first_action = dataset[1]["action"][i]
    #     first_state = dataset[1]["observation.state"][i]
    #     ax[0, 0].imshow(torch.permute(first_image, (1, 2, 0)))
    #     ax[0, 1].plot([first_action[0] / 512], [1 - first_action[1] / 512], "*")
    #     ax[0, 1].plot([first_state[0] / 512], [1 - first_state[1] / 512], "+")
    #     ax[0, 0].plot([first_action[0] / 512 * 64], [(first_action[1] / 512) * 64], "*")
    #     ax[0, 0].plot([first_state[0] / 512 * 64], [(first_state[1] / 512) * 64], "+")
    #     ax[0, 1].set_aspect("equal")
    #     ax[0, 1].set(xlim=(0, 1), ylim=(0, 1))

    #     fig.tight_layout()

    #     fig.savefig(dir_tmp / f"/home/quim/code/diamond/tmp/fig_action_{i:03d}.png")
    #     plt.close()

    sys.exit()

dataset_longtraj = qdataset.DiskDatasetTrajv2(datapath, tgt_traj_len=64)

dataloader = DataLoader(
    dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True
)
dataloader_longtraj = DataLoader(
    dataset_longtraj, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True
)

dl = diff_utils.cycle(dataloader)

num_steps = int(1e6)


action_normalization = 512


def get_diffusion_batch(batch):
    """ """
    vel_normalization = 0.3

    batch_size = batch["observation.image"].shape[0]

    image_1 = 2.0 * batch["observation.image"].to(device) - 1.0

    if act_is_vel:
        act_01 = (
            (batch["action"] - batch["observation.state"]).to(device)
            / action_normalization
            / vel_normalization
        )

        # i want to discretize the input in 7 bins, 
        # going from -vel_normalization to vel_normalization.
        # the input has two channels, so i will end up with 7 bins for each channel.

    elif act_is_vel_discretized:
        # Compute the normalized velocity
        act_01 = (
            (batch["action"] - batch["observation.state"]).to(device)
            / action_normalization
            / vel_normalization
        )

        # Discretize the input into 7 bins ranging from -vel_normalization to vel_normalization
        bin_width = (2 * vel_normalization) / 7  # Width of each bin
        act_01_clamped = act_01.clamp(-vel_normalization, vel_normalization)
        act_01_shifted = act_01_clamped + vel_normalization  # Shift to range [0, 2 * vel_normalization]
        act_bins = torch.floor(act_01_shifted / bin_width).long()
        act_bins = act_bins.clamp(max=6)  # Ensure bin indices are within [0, 6]
        act_01 = act_bins  # Discretized actions
    else:
        act_01 = 2.0 * batch["action"].to(device) / action_normalization - 1.0
    
    tgt_traj_len = batch["observation.image"].shape[1]

    batch = Batch(
        obs=image_1,
        act=act_01,
        mask_padding=torch.ones(batch_size, tgt_traj_len, dtype=torch.long).to(device),
        trunc=torch.zeros(batch_size, tgt_traj_len).to(device),
        rew=torch.zeros(batch_size, tgt_traj_len).to(device),
        end=torch.zeros(batch_size, tgt_traj_len).to(device),
        info=[],
        segment_ids=[],
    )
    return batch


opt = utils.configure_opt(model, lr=1e-4, weight_decay=1e-2, eps=1e-8)
num_warmup_steps = 1000
scheduler = utils.get_lr_sched(opt, num_warmup_steps=num_warmup_steps)


evaluate_every = 400
save_every = 400


loss_queue = deque(maxlen=200)


# state_dict_path = "/home/quim/code/diamond/qoutput/2024-12-02/17-25-07_4zyx5f/ckpt/state_dict_00007200.pth"
# state_dict = torch.load(state_dict_path, weights_only=True)
# model.load_state_dict(state_dict)


# state_dict_path = "/home/quim/code/diamond/qoutput/2024-11-27/11-18-25_dnx5h6/ckpt/state_dict_00016600.pth"
# state_dict = torch.load(state_dict_path, weights_only=True)
# model.load_state_dict(state_dict)

# Note:
# they only use one autoregressive step, but I use quite a lot. does this matter?
# num_autoregressive_steps: 1
# TODO: try with less steps!
# TODO: try with action as the velocity! (dx with respect to the current one?)

max_grad_norm = 1.0

step = 0
while step < num_steps:
    opt.zero_grad()
    model.train()
    step += 1
    batch = next(dl)
    batch_diffusion = get_diffusion_batch(batch)
    # print("batch diffusion max and min")
    # print(batch_diffusion.act.max())
    # print(batch_diffusion.act.min())
    loss, metric = model(batch_diffusion)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )

    opt.step()
    loss_queue.append(loss.item())
    scheduler.step()

    if step % evaluate_every == 0:

        model.eval()

        print(f"Step {step} Loss {sum(loss_queue) / len(loss_queue)}")
        batch = next(dl)
        batch_diffusion = get_diffusion_batch(batch)
        out = diffusion_sampler.sample(
            prev_obs=batch_diffusion.obs[:, :num_steps_conditioning],
            prev_act=batch_diffusion.act[:, :num_steps_conditioning],
        )

        x = out[0]

        # difference.
        print(
            "average difference ",
            torch.linalg.norm(batch_diffusion.obs[:, num_steps_conditioning] - out[0])
            / batch_diffusion.obs.shape[0],
        )

        # visualize the difference using matlplotlib for the first image. 
        # show three images

        # Visualize the difference using matplotlib for the first image
        # Show three images: Original Image, Output Image, Difference Image

        # Extract the images
        original_image = batch_diffusion.obs[0, num_steps_conditioning].cpu().numpy()
        output_image = x[0].cpu().detach().numpy()
        difference_image = original_image - output_image

        # Since images are typically in (C, H, W), transpose them to (H, W, C)
        original_image = np.transpose(original_image, (1, 2, 0))
        output_image = np.transpose(output_image, (1, 2, 0))
        difference_image = np.transpose(difference_image, (1, 2, 0))

        # Rescale images from [-1, 1] to [0, 1] for visualization
        original_image = (original_image + 1) / 2
        output_image = (output_image + 1) / 2
        # For the difference image, shift the range to [0, 1] for visualization
        difference_image = (difference_image - difference_image.min()) / (difference_image.max() - difference_image.min())

        # Clamp values to [0, 1] to avoid any artifacts
        original_image = np.clip(original_image, 0, 1)
        output_image = np.clip(output_image, 0, 1)
        difference_image = np.clip(difference_image, 0, 1)

        # Plot the images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(original_image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(output_image)
        axs[1].set_title('Output Image')
        axs[1].axis('off')

        axs[2].imshow(difference_image)
        axs[2].set_title('Difference Image')
        axs[2].axis('off')

        plt.savefig("tmp_fig_diff.png")

       


        out_v2 = diffusion_sampler_v2.sample(
            prev_obs=batch_diffusion.obs[:, :num_steps_conditioning],
            prev_act=batch_diffusion.act[:, :num_steps_conditioning],
        )
        print(
            "average difference v2 ",
            torch.linalg.norm(
                batch_diffusion.obs[:, num_steps_conditioning] - out_v2[0]
            )
            / batch_diffusion.obs.shape[0],
        )

        x_01 = 0.5 * (x + 1)

        real_imgs = 0.5 * (batch_diffusion.obs[:, num_steps_conditioning] + 1)
        real_imgs = real_imgs.maximum(
            0.9 * torch.ones_like(real_imgs)
        )  # we real images a bit darker

        all_imgs = []
        for img, real_img in zip(x_01, real_imgs):
            all_imgs.append(real_img)
            all_imgs.append(img)
        all_imgs = torch.stack(all_imgs)

        fout = img_dir / f"new_samples_{step:08d}.png"
        torchvision.utils.save_image(all_imgs, fout, nrow=8)

        # lets generate a long trajectory!

        long_batch = next(iter(dataloader_longtraj))
        traj_len = long_batch["observation.image"].shape[1]

        batch_diffusion = get_diffusion_batch(long_batch)

        prev_obs = batch_diffusion.obs[:, :num_steps_conditioning]
        prev_act = batch_diffusion.act[:, :num_steps_conditioning]

        traj_list = []
        traj_list_real = []

        for i in range(num_steps_conditioning, traj_len):
            out = diffusion_sampler.sample(prev_obs=prev_obs, prev_act=prev_act)
            # roll prev obs
            new_obs = out[0]
            prev_obs = torch.roll(prev_obs, shifts=-1, dims=1)
            prev_obs[:, -1] = new_obs
            prev_act = torch.roll(prev_act, shifts=-1, dims=1)
            prev_act[:, -1] = batch_diffusion.act[:, i]
            traj_list.append(
                torch.stack(
                    [
                        add_action(
                            0.5 * new_obs[j] + 0.5,
                            (0.5 * batch_diffusion.act[j, i] + 0.5) * 512,
                        )
                        for j in range(new_obs.shape[0])
                    ]
                )
            )
            traj_list_real.append(
                torch.stack(
                    [
                        add_action(
                            0.5 * batch_diffusion.obs[j, i] + 0.5,
                            (0.5 * batch_diffusion.act[j, i] + 0.5) * 512,
                        )
                        for j in range(new_obs.shape[0])
                    ]
                )
            )

            # breakpoint()
        # traj is list of (batch, 3 , 64 , 64)

        traj_list = torch.stack(traj_list, dim=1)
        traj_list_real = torch.stack(traj_list_real, dim=1)

        all_trajs = []
        for traj_real, traj_fake in zip(traj_list_real, traj_list):
            all_trajs.append(traj_real.maximum(0.9 * torch.ones_like(traj_real)))
            all_trajs.append(traj_fake)
        traj = torch.stack(all_trajs)

        # traj = 0.5 * (traj + 1)

        # lets put them all togher
        imgs = einops.rearrange(traj, "b t ...  -> (b t) ...")

        fout_long_rollout = img_dir / f"long_rollout_{step:08d}.png"

        torchvision.utils.save_image(
            imgs, fout_long_rollout, nrow=traj_len - num_steps_conditioning
        )

        fout_long_rollout_gif = img_dir / f"long_rollout_{step:08d}.gif"

        diff_utils.generate_gif_multiple_trajectories(
            traj, fout_long_rollout_gif, fps=5
        )

    if step % save_every == 0:

        fout = ckpt_dir / f"state_dict_{step:08d}.pth"
        torch.save(model.state_dict(), fout)
