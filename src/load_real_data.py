#
import os
import glob
import imageio
import pickle
import pathlib
import torch
import numpy as np
import torchvision

dt_original = 0.1
dt_new = 0.2

traj_len = 64


img_path = "/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/2024-12-06__16-33-06/img"

# Define the path

# Get a sorted list of all PNG files in the directory
png_files = sorted(glob.glob(os.path.join(img_path, "*.png")))
print("png_files", png_files)

# Load the images into a list
images = [imageio.imread(file) for file in png_files]


data_file = "/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/2024-12-06__16-33-06/data/data_extended.pkl"
with open(data_file, "rb") as f:
    Din = pickle.load(f)

outputdir = pathlib.Path(
    "/home/quim/code/diamond/data/real_robot_pusht_v0/data_t_v0/2024-12-06__16-33-06/diamond/"
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

num_pieces =  num_frames //  traj_len
import sys

print("num pieces: " , num_pieces)

for i in range(num_pieces):
    D = {}
    _imgs = torch.stack([imageio_to_tensor(img) for img in images[traj_len * i: traj_len * (i + 1)]])
    # torchvision.utils.save_image(
    #         _imgs, "tmp.png"
    #     )
    # print("max img")
    # print(_imgs.max())
    # print(_imgs.min())
    D["observation.image"] = _imgs
    _actions =  torch.stack([ torch.tensor(v) for v in Din["vs"][traj_len * i: traj_len * (i + 1)]])
    # print(_actions.max())
    # print(_actions.min())
    D["action"] = _actions
    fout = outputdir / f"trajectory_{i:09d}.pth"
    torch.save(D, fout)
    


    # convert to torch array
