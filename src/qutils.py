import torch
from dataclasses import asdict, dataclass
from typing import List, Tuple
from torch import Tensor
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from data import Batch



def get_diffusion_batch(batch, act_is_vel: bool, act_is_vel_discretized: bool, device):
    """ """
    action_normalization = 512
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
        act_01_shifted = (
            act_01_clamped + vel_normalization
        )  # Shift to range [0, 2 * vel_normalization]
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


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    save_path,
    denoiser_cfg,
    diffusion_sampler_cfg,
    sigma_distribution_cfg,
):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "denoiser_cfg": denoiser_cfg.to_dict(),
        "diffusion_sampler_cfg": diffusion_sampler_cfg.to_dict(),
        "sigma_distribution_cfg": sigma_distribution_cfg.to_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at step {step} to {save_path}")


def load_checkpoint(load_path, device):
    checkpoint = torch.load(load_path, map_location=device)

    # Reconstruct cfgurations
    denoiser_cfg = DenoiserConfig.from_dict(checkpoint["denoiser_cfg"])
    diffusion_sampler_cfg = DiffusionSamplerConfig.from_dict(
        checkpoint["diffusion_sampler_cfg"]
    )
    sigma_distribution_cfg = SigmaDistributionConfig.from_dict(
        checkpoint["sigma_distribution_cfg"]
    )

    # Instantiate the model with the loaded cfguration
    model = Denoiser(denoiser_cfg)
    model.to(device)

    # Load the state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    model.setup_training(sigma_distribution_cfg)

    # Optionally, load optimizer and scheduler states
    # opt.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    step = checkpoint["step"]

    print(f"Checkpoint loaded from {load_path} at step {step}")

    return {
        "model": model,
        "step": step,
        "denoiser_cfg": denoiser_cfg,
        "diffusion_sampler_cfg": diffusion_sampler_cfg,
        "sigma_distribution_cfg": sigma_distribution_cfg,
    }
