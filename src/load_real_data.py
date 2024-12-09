#
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
# dt_new = 0.2 not use yet!

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

    # Define the path

    # Get a sorted list of all PNG files in the directory
    png_files = sorted(glob.glob(os.path.join(img_path, "*.png")))
    print("png_files", png_files)

    # Load the images into a list
    images = [imageio.imread(file) for file in png_files]

    data_file = f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/data/data_extended.pkl"
    with open(data_file, "rb") as f:
        Din = pickle.load(f)

    outputdir = pathlib.Path(
        f"/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/{data_id}/diamond/"
    )

    outputdir.mkdir(exist_ok=True, parents=True)

    num_frames = len(images)
    num_data = len(Din["q"])
    assert num_data == num_frames

    def imageio_to_tensor(img):

        image_np = np.array(img)  # Ensure it's a NumPy array

        # Step 3: Convert to a PyTorch tensor
        image_tensor = torch.from_numpy(image_np)

        # Step 4: Permute dimensions (H, W, C) -> (C, H, W) if needed
        if len(image_tensor.shape) == 3:  # If RGB or RGBA
            image_tensor = image_tensor.permute(2, 0, 1)

        # Step 5: Convert to float and scale to [0, 1] (optional)
        image_tensor = image_tensor.float() / 255.0
        return image_tensor

    num_pieces = num_frames // traj_len
    import sys

    print("num pieces: ", num_pieces)

    max_action = -np.inf
    min_action = +np.inf

    for i in range(num_pieces):
        D = {}
        _imgs = torch.stack(
            [
                imageio_to_tensor(img)
                for img in images[traj_len * i : traj_len * (i + 1)]
            ]
        )
        # torchvision.utils.save_image(
        #         _imgs, "tmp.png"
        #     )
        # print("max img")
        # print(_imgs.max())
        # print(_imgs.min())
        D["observation.image"] = _imgs
        _actions = torch.stack(
            [torch.tensor(v) for v in Din["vs"][traj_len * i : traj_len * (i + 1)]]
        )
        # print(_actions.max())
        # print(_actions.min())
        D["action"] = _actions
        if _actions.max() > max_action:
            max_action = _actions.max()
        if _actions.min() < min_action:
            min_action = _actions.min()
        fout = outputdir / f"trajectory_{i:09d}.pth"
        torch.save(D, fout)

    print("max_action", max_action)
    print("min_action", min_action)
    # convert to torch array
