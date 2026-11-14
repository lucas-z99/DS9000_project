import numpy as np
from pathlib import Path
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import time


class ShardDataset:
    """
    Provides file look up
    """

    def __init__(self, data_DIR, cap_y=None):
        """
        scan all data files and map { index -> files }
        so others can __getitem__(i)

        cap_y = load a smaller set by capping the class y
        """

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

        # only allow load a smaller batch
        if cap_y is not None:
            ids = np.array([self.YY[file][row] for file, row in self.index])
            self.index = self.index[ids < cap_y]

    def __getitem__(self, i):
        file_i, row = self.index[i]
        image = np.array(self.XX[file_i][row])
        x = torch.from_numpy(1 - image / 255).float().flatten()
        y = int(self.YY[file_i][row])  # class

        return x, y

    def __len__(self):
        return len(self.index)


def load_test():

    print("load_test()")

    # load shards
    dataset = ShardDataset("train_set")
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
