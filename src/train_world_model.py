import torch
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
import torchvision
from data import Batch

from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from torch.utils.data import DataLoader, ConcatDataset
import einops
from collections import deque
import utils
import numpy as np


from qutils import save_checkpoint, load_checkpoint, add_action, get_diffusion_batch

import sys

sys.path.append("/home/quim/code/diffusion_planning_v2/src")

import qdataset
import diff_utils
import pathlib
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


repo_base = pathlib.Path(__file__).parent.parent


torch.set_num_threads(4)


@hydra.main(config_path="../config/", config_name="world_model")
def main(cfg: DictConfig):

    rand_id = diff_utils.get_unique_id()
    timestamp = diff_utils.get_time_stamp()

    print("Time stamp:", timestamp)
    print("Unique ID:", rand_id)

    # Print the configuration (optional)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb if enabled
    if cfg.wandb.enabled:

        # Prepare wandb configuration by filtering relevant hyperparameters
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["time_stamp"] = timestamp
        wandb_config["unique_id"] = rand_id

        wandb.init(
            project=cfg.wandb.project,
            config=wandb_config,
        )

    output_dir = repo_base / pathlib.Path(f"qoutput/{timestamp}_{rand_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = output_dir / "img"
    ckpt_dir = output_dir / "ckpt"

    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_steps_conditioning = 2
    num_autoregressive_steps = 4
    num_actions = 2

    act_is_vel = True
    act_is_vel_discretized = False

    inner_model_cfg = InnerModelConfig(**OmegaConf.to_object(cfg.inner_model_cfg))

    denoiser_cfg = DenoiserConfig(
        inner_model=inner_model_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.3,
    )

    sigma_distribution_cfg = SigmaDistributionConfig(
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
    # device = torch.device("cpu")

    model = Denoiser(denoiser_cfg)
    model.to(device)

    batch_size = cfg.train.batch_size
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

    model.setup_training(sigma_distribution_cfg)

    loss, metric = model(batch)

    # print("loss", loss)
    # print("metric", metric)

    diffusion_sampler = DiffusionSampler(denoiser=model, cfg=diffusion_sampler_cfg)
    diffusion_sampler_v2 = DiffusionSampler(
        denoiser=model, cfg=diffusion_sampler_cfg_v2
    )

    out = diffusion_sampler.sample(
        prev_obs=batch.obs[:, :num_steps_conditioning],
        prev_act=batch.act[:, :num_steps_conditioning],
    )

    visualize_data = False

    data_ids = [
        "2024-12-06__16-33-06",
        "2024-12-06__19-11-22",
        "2024-12-08__13-00-39",
        "2024-12-06__19-39-10",
        "2024-12-08__12-33-54",
    ]

    tgt_traj_len = num_steps_conditioning + 1 + num_autoregressive_steps

    dataset = ConcatDataset(
        [
            qdataset.DiskDatasetTrajv2(
                f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/diamond/",
                tgt_traj_len=tgt_traj_len,
            )
            for data_id in data_ids
        ]
    )

 

    dataset_longtraj = ConcatDataset(
        [
            qdataset.DiskDatasetTrajv2(
                f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/diamond/",
                tgt_traj_len=64,
            )
            for data_id in data_ids
        ]
    )

    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dataloader_longtraj = DataLoader(
        dataset_longtraj,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dl = diff_utils.cycle(dataloader)

    num_steps = int(1e6)

    opt = utils.configure_opt(model, lr=1e-4, weight_decay=1e-2, eps=1e-8)
    num_warmup_steps = 1000
    scheduler = utils.get_lr_sched(opt, num_warmup_steps=num_warmup_steps)

    evaluate_every = 400
    save_every = 400

    loss_queue = deque(maxlen=1000)
    path = repo_base /   "qoutput/2024-12-08/18-54-06_8ctpfr/ckpt/ckpt_00067200.pth"

    D = load_checkpoint(path, device)
    model = D["model"]

    # NOTE: this is looking well
    # Checkpoint saved at step 104000 to qoutput/2024-12-08/11-56-23_6w5ru1/ckpt/ckpt_00104000.pth
    # Step 104400 Loss 0.0008428849202755373
    # average difference  tensor(0.3025, device='cuda:0')
    # average difference v2  tensor(0.3047, device='cuda:0')
    # Checkpoint saved at step 104400 to qoutput/2024-12-08/11-56-23_6w5ru1/ckpt/ckpt_00104400.pth
    # Step 104800 Loss 0.0008612034496036358
    # average difference  tensor(0.2524, device='cuda:0')
    # average difference v2  tensor(0.2995, device='cuda:0')
    # Checkpoint saved at step 104800 to qoutput/2024-12-08/11-56-23_6w5ru1/ckpt/ckpt_00104800.pth

    # i started from
    # state_dict_path  = "qoutput/2024-12-06/18-26-49_vthfi6/ckpt/state_dict_00019200.pth"

    # state_dict_path = "/home/quim/code/diamond/qoutput/2024-12-02/18-31-10_ky80fb/ckpt/state_dict_00151200.pth"
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

    _get_diffusion_batch = get_diffusion_batch

    def get_diffusion_batch_real(batch):
        """ """
        # NOTE: s
        # max_action tensor(0.1418)
        # min_action tensor(-0.1339)
        max_action = 0.15  # this was the first dataset i have.
        max_action = 0.2  # this is with the new data!

        batch_size = batch["observation.image"].shape[0]

        image_1 = 2.0 * batch["observation.image"].to(device) - 1.0

        act_01 = batch["action"].to(device) / max_action

        tgt_traj_len = batch["observation.image"].shape[1]

        batch = Batch(
            obs=image_1,
            act=act_01,
            mask_padding=torch.ones(batch_size, tgt_traj_len, dtype=torch.long).to(
                device
            ),
            trunc=torch.zeros(batch_size, tgt_traj_len).to(device),
            rew=torch.zeros(batch_size, tgt_traj_len).to(device),
            end=torch.zeros(batch_size, tgt_traj_len).to(device),
            info=[],
            segment_ids=[],
        )
        return batch

    _get_diffusion_batch = get_diffusion_batch_real

    best_loss = 1e6
    while step < num_steps:
        opt.zero_grad()
        model.train()
        batch = next(dl)
        batch_diffusion = _get_diffusion_batch(batch)
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

        if step > 0 and step % evaluate_every == 0:

            model.eval()

            loss_avg = sum(loss_queue) / len(loss_queue)
            if loss_avg < best_loss:
                fout = ckpt_dir / f"state_dict_best.pth"
                torch.save(model.state_dict(), fout)

                fout = ckpt_dir / f"ckpt_best.pth"
                save_checkpoint(
                    model,
                    opt,
                    scheduler,
                    step,
                    fout,
                    denoiser_cfg,
                    diffusion_sampler_cfg,
                    sigma_distribution_cfg,
                )

            best_loss = min(best_loss, loss_avg)

            print(f"Step {step} Loss {loss_avg}")
            batch = next(dl)
            batch_diffusion = _get_diffusion_batch(batch)
            out = diffusion_sampler.sample(
                prev_obs=batch_diffusion.obs[:, :num_steps_conditioning],
                prev_act=batch_diffusion.act[:, :num_steps_conditioning],
            )

            x = out[0]

            diff = torch.abs(batch_diffusion.obs[:, num_steps_conditioning] - x)
            diff = diff.clamp(0.0, 0.1)
            # normalize diff so that max (0.1) maps to 1.0
            diff = diff / 0.1

            _out = torch.stack(
                (
                    0.5 * batch_diffusion.obs[:, num_steps_conditioning] + 0.5,
                    0.5 * x + 0.5,
                    diff,
                )
            )
            _out = einops.rearrange(_out, "t b ... -> (b t) ...")
            fout_difference = img_dir / f"diff_img_{step:08d}.png"
            torchvision.utils.save_image(_out, fout_difference, nrow=3)

            # difference.
            print(
                "average difference ",
                torch.linalg.norm(
                    batch_diffusion.obs[:, num_steps_conditioning] - out[0]
                )
                / batch_diffusion.obs.shape[0],
            )


            long_batch = next(iter(dataloader_longtraj))
            traj_len = long_batch["observation.image"].shape[1]

            batch_diffusion = _get_diffusion_batch(long_batch)

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
                all_trajs.append(traj_real.minimum(0.9 * torch.ones_like(traj_real)))
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

            if cfg.wandb.enabled:

                wandb.log({"step": step, "loss": sum(loss_queue) / len(loss_queue)})

                wandb.log(
                    {
                        "Fake/Real Image": wandb.Image(
                            str(fout_difference),
                            caption=f"step {step + 1} Real Images",
                        ),
                        "Long Rollout": wandb.Image(
                            str(fout_long_rollout),
                            caption=f"step {step + 1} Fake Images",
                        ),
                        "Long Rollout GIF": wandb.Video(
                            str(fout_long_rollout_gif),
                            caption=f"step {step + 1} Gif Rollout",
                        ),
                    }
                )

        if step % save_every == 0 or step == num_steps - 1:
            # now we only store the best one and the last one
            pass 


            # fout = ckpt_dir / f"state_dict_{step:08d}.pth"
            # torch.save(model.state_dict(), fout)

            # fout = ckpt_dir / f"ckpt_{step:08d}.pth"
            # save_checkpoint(
            #     model,
            #     opt,
            #     scheduler,
            #     step,
            #     fout,
            #     denoiser_cfg,
            #     diffusion_sampler_cfg,
            #     sigma_distribution_cfg,
            # )

            # should I save the model

        step += 1

    fout = ckpt_dir / f"state_dict_last.pth"
    torch.save(model.state_dict(), fout)

    fout = ckpt_dir / f"ckpt_last.pth"
    save_checkpoint(
        model,
        opt,
        scheduler,
        step,
        fout,
        denoiser_cfg,
        diffusion_sampler_cfg,
        sigma_distribution_cfg,
    )


if __name__ == "__main__":
    main()
