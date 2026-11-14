import numpy as np
from pathlib import Path
from PIL import Image
from a1_prepare_data import safe_file_name, build_class_dict

# ------------------------ check one shard ------------------------

writer = "1052-f"


# ------------------------ business ------------------------

x = np.load(f"train_set/x_{writer}.npy")
y = np.load(f"train_set/y_{writer}.npy")
w = np.load(f"train_set/w_{writer}.npy")
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
