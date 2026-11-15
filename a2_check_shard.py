import numpy as np
from pathlib import Path
from PIL import Image
from a1_prepare_data import (
    safe_file_name,
    build_class_dict,
    split_path,
    prepare_data,
    IMG_RESIZE,
)

# ------------------------ check one shard ------------------------

image_size = 128
writer = "1052-f"
if image_size not in IMG_RESIZE:
    raise ValueError(f"{image_size} is not in IMG_RESIZE {IMG_RESIZE}")
prepare_data()
root = split_path(image_size, "train")


# ------------------------ business ------------------------

x = np.load(f"{root}/x_{writer}.npy")
y = np.load(f"{root}/y_{writer}.npy")
w = np.load(f"{root}/w_{writer}.npy")
classes = list(build_class_dict())

print(x.shape, x.dtype)  # ~3900, 128x128, uint8
print(y.shape)  # ~3900;
print(f"writer = {w[:3]}")

Path("debug").mkdir(parents=True, exist_ok=True)
DIR = "debug"

for i in range(0, 2000, 50):
    label = classes[y[i]]
    Image.fromarray(x[i]).save(f"{DIR}/{safe_file_name(label)}_{y[i]}.png")

print(f">> images saved to {DIR}")
