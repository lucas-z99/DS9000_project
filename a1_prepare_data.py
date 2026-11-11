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
TRAIN_SET_DIR = "train_set"
VAL_SET_DIR = "val_set"
TEST_SET_DIR = "test_set"

# image
W_PADDING = 256  # padding (before resize)
H_PADDING = 256
W_RESIZE = 128  # resize
H_RESIZE = 128


# ---------------------------------   unpack data   ---------------------------------


def unpack_gnt(path, label_only=False):
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


def process_data_set(gnt_DIR, class_dict, save_DIR):

    x = []  # images
    labels = []  # charactors
    writer = gnt_DIR.stem

    print(f"WIP: {writer}")

    for label, unpad_image in unpack_gnt(gnt_DIR):

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


# ---------------------------------   image padding   ---------------------------------


def image_padding(image, width, height):
    """
    add padding around image
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


def padding_test():
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
    labels = set()
    for folder in (RAW_TRAIN_DIR, RAW_TEST_DIR):
        for gnt in sorted(f for f in folder.iterdir() if f.is_file()):
            for label, _ in unpack_gnt(gnt, label_only=True):
                labels.add(label)
    return {l: i for i, l in enumerate(sorted(labels))}


def main():

    Path(TRAIN_SET_DIR).mkdir(parents=True, exist_ok=True)
    Path(VAL_SET_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_SET_DIR).mkdir(parents=True, exist_ok=True)

    # ----------   collect labels, generate class   ----------

    class_dict = build_class_dict()
    Path("label_to_class.txt").write_text("\n".join(class_dict), encoding="utf-8")
    print(f"class = {len(class_dict)}, should be ~3922")

    # ----------   split train / val from RAW_TRAIN_SET   ----------

    random.seed(42)
    _writers = [f.stem for f in RAW_TRAIN_DIR.iterdir() if f.is_file()]
    val_writers = random.sample(sorted(set(_writers)), 20)
    print(">> Validation = ", val_writers)
    Path("val_writers.txt").write_text("\n".join(val_writers), encoding="utf-8")

    # ----------   prepare data   ----------

    for gnt in [f for f in Path(RAW_TRAIN_DIR).iterdir() if f.is_file()]:
        if gnt.stem in val_writers:
            process_data_set(gnt, class_dict, VAL_SET_DIR)
        else:
            process_data_set(gnt, class_dict, TRAIN_SET_DIR)

    for gnt in [f for f in Path(RAW_TEST_DIR).iterdir() if f.is_file()]:
        process_data_set(gnt, class_dict, TEST_SET_DIR)

    # ----------   prepare data [SMALL TEST]   ----------

    # files = sorted(f for f in RAW_TRAIN_SET_DIR.iterdir() if f.is_file())
    # test_files = [f for f in files if f.stem not in val_writers][:2] + [
    #     f for f in files if f.stem in val_writers
    # ][:1]
    # for gnt in test_files:
    #     if gnt.stem in val_writers:
    #         process_data_set(gnt, class_dict, VAL_SET_DIR)
    #     else:
    #         process_data_set(gnt, class_dict, TRAIN_SET_DIR)

    print("\nDONE!")
    # print(f"train set       = {len(x_train)}")
    # print(f"val set         = {len(x_val)}")
    # print(f"test set        = {len(x_test)}")
    # print(f"unique char     = {len(class_dict)}")

    # print("\nmissing char in each set:")
    # print(f"train: {int((y_train == -1).sum())} char are never learned")
    # print(f"val:   {int((y_val == -1).sum())}")
    # print(f"test:  {int((y_test == -1).sum())} char are never tested")

    # assert len(x_train) == len(y_train) == len(y_writer)
    # assert len(x_val) == len(y_val) == len(y_writer_val)
    # assert len(x_test) == len(y_test) == len(y_writers_test)


if __name__ == "__main__":
    main()
