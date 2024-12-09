import os
import glob
import imageio
import pickle
import pathlib
import torch
import numpy as np
import torchvision

# max_action tensor(0.1624)
# min_action tensor(-0.1795)

dt_original = 0.1
dt_new = 0.2  # Adjust this value as desired

# original dt is at 0.1
# we want to store data taking a new dt (e.g., 0.2, or 0.3)
# This means we will downsample our data accordingly.

traj_len = 64

data_ids = [
    "2024-12-06__19-11-22",
    "2024-12-08__13-00-39/",
    "2024-12-06__19-39-10",
    "2024-12-08__12-33-54",
]

for data_id in data_ids:

    img_path = (
        f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/img"
    )

    # Get a sorted list of all PNG files in the directory
    png_files = sorted(glob.glob(os.path.join(img_path, "*.png")))
    print("png_files", png_files)

    # Load the images into a list
    images = [imageio.imread(file) for file in png_files]

    data_file = f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/data/data_extended.pkl"
    with open(data_file, "rb") as f:
        Din = pickle.load(f)

    outputdir = pathlib.Path(
        f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/diamond_dt2/"
    )

    outputdir.mkdir(exist_ok=True, parents=True)

    num_frames = len(images)
    num_data = len(Din["q"])
    assert num_data == num_frames, "Mismatch between frames and data length"

    def imageio_to_tensor(img):
        image_np = np.array(img)  # Ensure it's a NumPy array
        image_tensor = torch.from_numpy(image_np)
        if len(image_tensor.shape) == 3:  # If RGB or RGBA
            image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0
        return image_tensor

    # Compute the step for downsampling
    step = int(dt_new / dt_original)
    if step < 1:
        step = 1  # At least we take every frame if dt_new <= dt_original

    # Downsample images and data according to dt_new
    images_downsampled = images[::step]
    actions_downsampled = Din["vs"][::step]
    # If there are other keys in Din that need downsampling, do it similarly.
    # For example, if needed: q_downsampled = Din["q"][::step]

    # Now we have a reduced dataset at the new dt
    num_frames_new = len(images_downsampled)
    num_actions_new = len(actions_downsampled)
    # assert num_frames_new == num_actions_new, "Mismatch after downsampling"

    num_pieces = num_frames_new // traj_len
    print("num pieces: ", num_pieces)

    max_action = -np.inf
    min_action = +np.inf

    for i in range(num_pieces):
        D = {}
        start = traj_len * i
        end = traj_len * (i + 1)

        _imgs = torch.stack(
            [imageio_to_tensor(img) for img in images_downsampled[start:end]]
        )
        D["observation.image"] = _imgs

        _actions = torch.stack(
            [torch.tensor(v) for v in actions_downsampled[start:end]]
        )
        D["action"] = _actions

        if _actions.max().item() > max_action:
            max_action = _actions.max().item()
        if _actions.min().item() < min_action:
            min_action = _actions.min().item()

        fout = outputdir / f"trajectory_{i:09d}.pth"
        torch.save(D, fout)

    print("max_action", max_action)
    print("min_action", min_action)
