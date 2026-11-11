import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from a1_prepare_data import RAW_TRAIN_DIR, RAW_TEST_DIR


def scan_image_dim():
    """
    scan and plot image dimensions:
        to decide compression size
        to have a better idea about our data
    """

    def read_sizes(folder):
        """return (heights, widths) of every example in a folder, reading headers only"""
        heights, widths = [], []
        for path in Path(folder).iterdir():
            if not path.is_file():
                continue
            with open(path, "rb") as file:
                while True:
                    header = file.read(10)
                    if len(header) < 10:
                        break
                    _, _, width, height = struct.unpack("<I2sHH", header)
                    file.seek(
                        width * height, 1
                    )  # skip the pixels, 1 = from current position
                    heights.append(height)
                    widths.append(width)
        return np.array(heights), np.array(widths)

    h_train, w_train = read_sizes(RAW_TRAIN_DIR)
    h_test, w_test = read_sizes(RAW_TEST_DIR)
    h = np.concatenate([h_train, h_test])
    w = np.concatenate([w_train, w_test])
    side = np.maximum(h, w)  # the longer side is what matters for a square pad

    print(f"examples: {len(h)}")
    print(f"height: mean={h.mean():.1f}  max={h.max()}")
    print(f"width:  mean={w.mean():.1f}  max={w.max()}")
    for p in (50, 90, 99, 99.9):
        print(f"longer side, {p}th percentile: {np.percentile(side, p):.0f}")

    plt.figure()
    plt.hist2d(
        w,
        h,
        bins=[np.arange(0, w.max() + 2), np.arange(0, h.max() + 2)],
        norm="log",
        cmap="viridis",
    )
    plt.colorbar(label="number of examples (log scale)")
    plt.plot(
        [0, 253], [0, 253], "w--", linewidth=0.8
    )  # square characters sit on this line
    plt.xlabel("width (px)")
    plt.ylabel("height (px)")
    plt.title("width x height density")
    plt.show()


if __name__ == "__main__":
    scan_image_dim()
