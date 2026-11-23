import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
from a1_prepare_data import (
    CLASS_DICT_DIR,
    IMG_RESIZE,
    build_class_dict,
    prepare_data,
    split_path,
)


def n_labels():
    """
    How many lines are in label_to_class.txt.
    """

    if CLASS_DICT_DIR.is_file():
        labels = CLASS_DICT_DIR.read_text(encoding="utf-8").splitlines()
        if labels:
            return len(labels)
    return len(build_class_dict())


def resolve_class_set(class_set, label_n):
    """
    None means every class. Anything else becomes sorted unique ids.
    """

    if class_set is None:
        return None
    ids = sorted({int(i) for i in class_set})
    if not ids:
        raise ValueError("class_set is empty")
    if ids[0] < 0 or ids[-1] >= label_n:
        raise ValueError(f"class id outside 0..{label_n - 1}: {ids[0]}..{ids[-1]}")
    return ids


def class_set_note(class_ids):
    """
    Readable inclusive spans. range(1000, 1200) becomes "1000-1199".
    """

    if class_ids is None:
        return "all"
    parts = []
    start = prev = class_ids[0]
    for i in class_ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def run_tag(class_ids, label_n):
    """
    Folder tag. A range uses the half-open end, so range(1000, 1200) is 1000-1200.
    A gap becomes 400mix.
    """

    if class_ids is None:
        return str(label_n)
    lo, hi = class_ids[0], class_ids[-1]
    if class_ids == list(range(lo, hi + 1)):
        return f"{lo}-{hi + 1}"
    return f"{len(class_ids)}mix"


def class_remap(class_ids, label_n):
    """
    Original class id -> output index. Missing ids are -1.
    """

    remap = np.full(label_n, -1, dtype=np.int64)
    remap[np.asarray(class_ids, dtype=np.int64)] = np.arange(len(class_ids))
    return remap


class ShardDataset:
    """
    Provides file look up
    """

    def __init__(self, split, image_size=128, class_set=None):
        """
        scan all data files and map { index -> files }
        so others can __getitem__(i)

        split = "train", "val", or "test"
        image_size picks prepared/<size>/<split>.
        class_set = None (every class), a range, or a list of class ids.
        Returned labels are remapped to 0 .. n-1.
        """

        if image_size not in IMG_RESIZE:
            raise ValueError(f"{image_size} is not in IMG_RESIZE {IMG_RESIZE}")
        prepare_data()
        data_DIR = split_path(image_size, split)
        x_DIRs = sorted(Path(data_DIR).glob("x_*.npy"))
        self.XX = []
        self.YY = []
        cur_index = []

        for index, x_DIR in enumerate(x_DIRs):
            y_DIR = x_DIR.with_name(x_DIR.name.replace("x_", "y_"))
            x = np.load(x_DIR, mmap_mode="r")  # just open (not loaded in RAM)
            y = np.load(y_DIR)

            self.XX.append(x)  # image files (not loaded in RAM)
            self.YY.append(y)  # class files

            sample_size = len(y)
            cur_index.append(
                np.stack(
                    [
                        np.full(sample_size, index),  # [0,0,0....]
                        np.arange(sample_size),  # [0,1,2...]
                    ],
                    axis=1,
                )
            )

        self.index = np.concatenate(cur_index)

        label_n = n_labels()
        class_ids = resolve_class_set(class_set, label_n)
        self.remap = None
        if class_ids is not None:
            self.remap = class_remap(class_ids, label_n)
            ids = np.array([self.YY[file][row] for file, row in self.index])
            self.index = self.index[self.remap[ids] >= 0]
            if len(self.index) == 0:
                raise ValueError(
                    f"no {split} samples in class_set {class_set_note(class_ids)}"
                )

    def __getitem__(self, i):
        file_i, row = self.index[i]
        image = np.array(self.XX[file_i][row])
        x = torch.from_numpy(1 - image / 255).float().flatten()
        y = int(self.YY[file_i][row])  # class
        if self.remap is not None:
            y = int(self.remap[y])

        return x, y

    def __len__(self):
        return len(self.index)


class DeviceLoader:
    """
    Fast replacement for DataLoader(ShardDataset(...)).

    Loads every image of the dataset once into one uint8 tensor that lives on the
    device (GPU). Each batch is then a single indexing op on the GPU, instead of
    hundreds of small python __getitem__ calls on the CPU.

    Memory: 48x48 train set is about 2 GB as uint8.
    Yields the same (x, y) as before: x = 1 - pixel/255, flattened; y = class.
    """

    def __init__(self, dataset, batch_size, shuffle, device, flatten=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.flatten = flatten

        index = dataset.index
        xs, ys = [], []
        for file_i in np.unique(index[:, 0]):
            rows = index[index[:, 0] == file_i, 1]
            xs.append(np.asarray(dataset.XX[file_i][rows]))  # read this shard once
            ys.append(np.asarray(dataset.YY[file_i][rows]))
        y = np.concatenate(ys).astype(np.int64)
        if dataset.remap is not None:
            y = dataset.remap[y]

        self.X = torch.from_numpy(np.concatenate(xs)).to(device)  # (N, H, W) uint8
        self.Y = torch.from_numpy(y).to(device)

    def __len__(self):
        return (len(self.Y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.Y)
        if self.shuffle:
            order = torch.randperm(n, device=self.device)
        else:
            order = torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            x = 1 - self.X[idx].float() / 255
            if self.flatten:
                x = x.flatten(1)
            yield x, self.Y[idx]


def load_test():

    print("load_test()")

    # load shards
    dataset = ShardDataset("train")
    print(len(dataset))  # should be ~860000

    # PyTorch loader (random pick)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    t = time.time()
    x, y = next(iter(loader))

    print(f"DONE: t = {time.time() - t:.2f} s")
    print(x[:5])
    print(y[:20])


if __name__ == "__main__":
    load_test()
