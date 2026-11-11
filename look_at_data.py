"""
Step 1: look at the HWDB1.1 data before we train anything.

What this does:
  1. Reads one training .gnt file and one test .gnt file (each file = one writer).
  2. Prints simple facts: how many samples, image size range, how many different characters.
  3. Saves a grid of random samples to samples.png (and shows it).

Only needs:  pip install numpy matplotlib
"""

import random
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- change this if your folder is somewhere else ----
DATA_DIR = Path(r"D:\Main\DS_data\HWDB1.1")
TRAIN_DIR = DATA_DIR / "HWDB1.1trn_gnt"
TEST_DIR = DATA_DIR / "HWDB1.1tst_gnt"

# Make matplotlib able to draw Chinese characters (Windows ships these fonts).
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def read_gnt(path):
    """Yield (label, image) for every character stored in one .gnt file.

    Each sample in the file is laid out like this:
        4 bytes  total size of this sample
        2 bytes  the character, encoded in GB2312
        2 bytes  image width
        2 bytes  image height
        w*h bytes  the image, one byte per pixel (255 = paper, small numbers = ink)
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(10)
            if len(header) < 10:  # end of file
                break
            size, tag, width, height = struct.unpack("<I2sHH", header)
            pixels = np.frombuffer(f.read(width * height), dtype=np.uint8)
            image = pixels.reshape(height, width)
            label = tag.decode("gb2312", errors="replace").strip("\x00")
            yield label, image


def summarize(name, gnt_path):
    samples = list(read_gnt(gnt_path))
    widths = [img.shape[1] for _, img in samples]
    heights = [img.shape[0] for _, img in samples]
    labels = {label for label, _ in samples}
    print(f"{name}: {gnt_path.name}")
    print(f"  samples          : {len(samples)}")
    print(f"  different labels : {len(labels)}")
    print(f"  width  min/max   : {min(widths)} / {max(widths)}")
    print(f"  height min/max   : {min(heights)} / {max(heights)}")
    return samples


def main():
    train_files = sorted(TRAIN_DIR.glob("*.gnt"))
    test_files = sorted(TEST_DIR.glob("*.gnt"))
    print(f"train files found: {len(train_files)} (expect 240)")
    print(f"test files found : {len(test_files)} (expect 60)\n")
    if not train_files or not test_files:
        raise SystemExit(
            "No .gnt files found - check DATA_DIR at the top of this script."
        )

    train_samples = summarize("TRAIN writer", train_files[0])
    print()
    summarize("TEST writer", test_files[0])

    # Draw 40 random characters from the training writer.
    random.seed(0)
    picks = random.sample(train_samples, 40)
    fig, axes = plt.subplots(5, 8, figsize=(12, 8))
    for ax, (label, img) in zip(axes.ravel(), picks):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=14)
        ax.axis("off")
    fig.suptitle(f"Random samples from {train_files[0].name}")
    fig.tight_layout()
    fig.savefig("samples.png", dpi=120)
    print("\nSaved samples.png")
    plt.show()


if __name__ == "__main__":
    main()
