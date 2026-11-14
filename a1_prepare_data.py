import struct
import numpy as np
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------   config   ---------------------------------

# raw data
"""
max image dim w=215 h=253
938679 examples
3922 unique charactors
"""
RAW_DATA_DIR = Path("raw_dataset")
RAW_TRAIN_DIR = RAW_DATA_DIR / "HWDB1.1trn_gnt"
RAW_TEST_DIR = RAW_DATA_DIR / "HWDB1.1tst_gnt"


# prepared data
CLASS_DICT = Path("label_to_class.txt")
TRAIN_SET_DIR = "train_set"
VAL_SET_DIR = "val_set"
TEST_SET_DIR = "test_set"

# image
W_PADDING = 256  # padding (before resize)
H_PADDING = 256
W_RESIZE = 128  # resize
H_RESIZE = 128


# ---------------------------------   unpack data   ---------------------------------


def process_gnt_file(gnt_DIR, class_dict, save_DIR):
    """
    Process a single .gnt file
    """

    x = []  # images
    labels = []  # charactors
    writer = gnt_DIR.stem

    print(f"Processing: {gnt_DIR.name}")

    for label, unpad_image in _unpack_gnt(gnt_DIR):

        image = image_padding(unpad_image, W_PADDING, H_PADDING)
        small_image = np.array(
            Image.fromarray(image).resize((W_RESIZE, H_RESIZE), Image.BOX)
        )

        x.append(small_image)
        labels.append(label)

    x = np.array(x)
    y = np.array([class_dict.get(l, -1) for l in labels])
    w = np.full(len(x), writer)  # who wrote this

    np.save(f"{save_DIR}/x_{writer}.npy", x)
    np.save(f"{save_DIR}/y_{writer}.npy", y)
    np.save(f"{save_DIR}/w_{writer}.npy", w)


def _unpack_gnt(path, label_only=False):
    """
    Handle the bits
    """
    with open(path, "rb") as file:
        while True:
            header = file.read(10)
            if len(header) < 10:
                break

            size, label_raw, width, height = struct.unpack("<I2sHH", header)
            label = label_raw.decode("gb2312", errors="replace").strip("\x00")

            if label_only:
                file.seek(width * height, 1)  # skip images
                yield label, None
            else:
                image = np.frombuffer(
                    file.read(width * height), dtype=np.uint8
                ).reshape(height, width)
                yield label, image


# ---------------------------------   image padding   ---------------------------------


def image_padding(image, width, height):
    """
    Add padding around image
    """

    h, w = image.shape
    assert h <= height and w <= width

    top = (height - h) // 2
    bot = height - h - top
    left = (width - w) // 2
    right = width - w - left

    pad = ((top, bot), (left, right))
    padded = np.pad(image, pad, mode="constant", constant_values=255)

    h2, w2 = padded.shape
    assert h2 == height and w2 == width

    return padded


def image_padding_test():
    print(">> image padding test")

    arr = np.random.randint(0, 255, size=(500, 41), dtype=np.uint8)

    img = Image.fromarray(arr)
    img.save("image_padding_test0.png")

    result = image_padding(arr, 500, 500)
    Image.fromarray(result).save("image_padding_test1.png")


# ---------------------------------   only used occationally   ---------------------------------


def safe_file_name(label):
    if (
        label
        and label not in '<>:"/\\|?*'
        and (label.isalnum() or ord(label[0]) > 0x2E80)
    ):
        return label
    return "U" + "_".join(f"{ord(c):04X}" for c in label)


# ---------------------------------   main   ---------------------------------


def build_class_dict():
    """
    label -> class index

    Load dict if it exists
    Otherwise scan all gnt, create and return the new dict
    """

    if CLASS_DICT.is_file():
        labels = CLASS_DICT.read_text(encoding="utf-8").splitlines()
        if labels:
            print(f">> Class dict loaded")
            return {label: i for i, label in enumerate(labels)}

    print("[Notice] Class dict missing, rebuilding... ")

    labels = set()
    for folder in (RAW_TRAIN_DIR, RAW_TEST_DIR):
        for gnt in sorted(f for f in folder.iterdir() if f.is_file()):
            for label, _ in _unpack_gnt(gnt, label_only=True):
                labels.add(label)
    class_dict = {l: i for i, l in enumerate(sorted(labels))}
    CLASS_DICT.write_text("\n".join(class_dict), encoding="utf-8")

    print(f"[Notice] Class dict ({len(class_dict)}) saved to {CLASS_DICT} ")
    return class_dict


def main():

    Path(TRAIN_SET_DIR).mkdir(parents=True, exist_ok=True)
    Path(VAL_SET_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_SET_DIR).mkdir(parents=True, exist_ok=True)

    # ----------   collect labels, generate class   ----------

    class_dict = build_class_dict()

    # ----------   split train / val from RAW_TRAIN_SET   ----------

    random.seed(42)
    _writers = [f.stem for f in RAW_TRAIN_DIR.iterdir() if f.is_file()]
    val_set = random.sample(sorted(set(_writers)), 20)
    print(">> val_set chosen: ", val_set)
    Path("val_set.txt").write_text("\n".join(val_set), encoding="utf-8")

    # ----------   prepare data   ----------

    for gnt in [f for f in Path(RAW_TRAIN_DIR).iterdir() if f.is_file()]:
        if gnt.stem in val_set:
            process_gnt_file(gnt, class_dict, VAL_SET_DIR)
        else:
            process_gnt_file(gnt, class_dict, TRAIN_SET_DIR)

    for gnt in [f for f in Path(RAW_TEST_DIR).iterdir() if f.is_file()]:
        process_gnt_file(gnt, class_dict, TEST_SET_DIR)

    print("\nDONE!")


if __name__ == "__main__":
    main()
