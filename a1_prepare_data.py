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


# image
PADDING = 256  # padding the original images (before resize)
IMG_RESIZE = [128, 96, 64, 48, 32, 24]  # [128, 96, 64, 48, 32, 24]

# ---------------------------------   config   ---------------------------------

# others
CLASS_DICT_DIR = Path("label_to_class.txt")
VAL_LIST_DIR = Path("val_set.txt")


# ---------------------------------   unpack data   ---------------------------------


def split_path(size, split):
    """
    prepared/<size>/<split>
    """

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val, or test, got {split}")
    return (Path("prepared") / str(size) / split).as_posix()


def resize_image(image, size):
    """
    Resize from the padded image. BOX keeps the area average.
    """

    if image.shape == (size, size):
        return image
    resized = Image.fromarray(image).resize((size, size), Image.BOX)
    return np.array(resized, dtype=np.uint8)


def process_gnt_file(gnt_DIR, class_dict, outputs):
    """
    Process a single .gnt file.

    outputs: list of (image_size, save_dir).
    Pad each character to 256 once, then resize to every size.
    Skip a size whose x_ file is already saved.
    """

    writer = gnt_DIR.stem
    pending = []
    for size, save_dir in outputs:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if (save_dir / f"x_{writer}.npy").is_file():
            continue
        pending.append((size, save_dir))
    if not pending:
        return

    print(f"Processing: {gnt_DIR.name} -> {[size for size, _ in pending]}")

    buckets = {size: [] for size, _ in pending}
    labels = []

    for label, unpad_image in _unpack_gnt(gnt_DIR):
        image = image_padding(unpad_image, PADDING, PADDING)
        labels.append(label)
        for size, _ in pending:
            buckets[size].append(resize_image(image, size))

    y = np.array([class_dict.get(l, -1) for l in labels])
    w = np.full(len(labels), writer)  # who wrote this

    for size, save_dir in pending:
        np.save(save_dir / f"x_{writer}.npy", np.array(buckets[size]))
        np.save(save_dir / f"y_{writer}.npy", y)
        np.save(save_dir / f"w_{writer}.npy", w)


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


# ---------------------------------   image   ---------------------------------


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


# ---------------------------------   others   ---------------------------------


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

    if CLASS_DICT_DIR.is_file():
        labels = CLASS_DICT_DIR.read_text(encoding="utf-8").splitlines()
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
    CLASS_DICT_DIR.write_text("\n".join(class_dict), encoding="utf-8")

    print(f"[Notice] Class dict ({len(class_dict)}) saved to {CLASS_DICT_DIR} ")
    return class_dict


def _gnt_stems(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"raw data not found: {folder}")
    stems = {f.stem for f in folder.iterdir() if f.is_file()}
    if not stems:
        raise FileNotFoundError(f"no .gnt files in {folder}")
    return stems


def val_writers():
    """
    Pick 20 writers as val_set (from HWDB1.1trn_gnt)
    Use a fixed seed
    """

    if not VAL_LIST_DIR.is_file():
        random.seed(42)
        chosen = random.sample(sorted(_gnt_stems(RAW_TRAIN_DIR)), 20)
        VAL_LIST_DIR.write_text("\n".join(chosen), encoding="utf-8")
        print(">> val_set chosen:", chosen)

    return {
        line.strip()
        for line in VAL_LIST_DIR.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def writers_by_split():
    val = val_writers()
    train = _gnt_stems(RAW_TRAIN_DIR)
    missing = val - train
    if missing:
        raise ValueError(f"val_set.txt has writers not in raw train: {sorted(missing)}")
    return {"train": train - val, "val": val, "test": _gnt_stems(RAW_TEST_DIR)}


def _saved_writers(size, split):
    folder = Path(split_path(size, split))
    return {path.stem.removeprefix("x_") for path in folder.glob("x_*.npy")}


def size_ready(size, groups=None):
    if groups is None:
        groups = writers_by_split()
    return all(groups[split] <= _saved_writers(size, split) for split in groups)


def prepare_data():
    """
    Prepare image for requested sizes
    Saved in: prepared/<size>/{train, val, test}
    Skip if already there

    Load one gnt -> unpack images/label -> pad to 256x256 -> resizes
    """

    groups = writers_by_split()
    have = [size for size in IMG_RESIZE if size_ready(size, groups)]
    need = [size for size in IMG_RESIZE if size not in have]
    # if have:
    #     print(f">> already on disk: {have}")
    if not need:
        return

    print(f">> preparing: {need}")
    class_dict = build_class_dict()

    def outputs_for(split):
        return [(size, split_path(size, split)) for size in need]

    for gnt in sorted(f for f in RAW_TRAIN_DIR.iterdir() if f.is_file()):
        split = "val" if gnt.stem in groups["val"] else "train"
        process_gnt_file(gnt, class_dict, outputs_for(split))

    for gnt in sorted(f for f in RAW_TEST_DIR.iterdir() if f.is_file()):
        process_gnt_file(gnt, class_dict, outputs_for("test"))

    print("\nDONE!")


if __name__ == "__main__":
    prepare_data()
