import csv
import torch.nn as nn
from dataloader import ShardDataset
from torch.utils.data import DataLoader
import torch
import time


class NN_Classifier(nn.Module):

    def __init__(self, n_class, n_input, n1, n2):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, n1),  # W1 b1
            nn.ReLU(),
            nn.Linear(n1, n2),  # W2 b2
            nn.ReLU(),
            nn.Linear(n2, n_class),  # outputs raw scores (logits)
        )

    def forward(self, x):
        return self.model(x)


class CsvWriter:
    """
    Write a csv row by row.
    Each row is flushed, so a stopped run keeps the log.
    """

    def __init__(self, path, header):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def evaluate(model, loader):
    model.eval()  # eval mode (matters later for dropout etc.)
    correct, total = 0, 0

    with torch.no_grad():  # no graph, no grads: faster, less memory
        for x, y in loader:  # walk the whole val set once
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

    model.train()  # back to training mode
    return correct / total


def eval_and_log(step, loss_value, t):
    """
    run a evaluation()
    and log .csv
    """
    val_acc = evaluate(model, val_loader)
    now = time.time()
    sec = now - t
    elapsed = now - t0
    print(f"[{step}] loss={loss_value:.3f}, val_acc = {val_acc:.3f}, t={sec:.1f}")
    log.write(
        [
            step,
            f"{loss_value:.6f}",
            f"{val_acc:.6f}",
            f"{sec:.1f}",
            f"{elapsed:.1f}",
        ]
    )
    return now


if __name__ == "__main__":

    # config
    n_class = 3900  # number of class we'll consider
    traning_loops = 2500
    eval_per = 100  # evaluate per loops. [Warning] too many will slow-down training

    # init model
    model = NN_Classifier(
        n_class=n_class,
        n_input=128 * 128,
        n1=512,
        n2=256,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(model)

    # train set
    dataset = ShardDataset("train_set", cap_y=n_class)
    loader = DataLoader(
        dataset,
        batch_size=256,  # smaller batch size = less noise = better training?
        shuffle=True,
    )

    # val set
    val_set = ShardDataset("val_set", cap_y=n_class)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)

    # loop
    print(f">> Config: {n_class=}, {traning_loops=}")
    print(">> Training start:")
    t0 = t = time.time()
    log = CsvWriter(
        "train_log.csv",
        ["step", "loss", "val_acc", "sec", "elapsed_sec"],
    )

    try:
        for i in range(traning_loops):
            optimizer.zero_grad()  # clear gradiants

            x, y = next(iter(loader))  # draw data

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            if i % eval_per == 0:
                t = eval_and_log(i, loss_value, t)

        last_step = traning_loops - 1
        if last_step % eval_per != 0:
            eval_and_log(last_step, loss_value, t)
    finally:
        log.close()

    print(f"total t={(time.time() - t0)/60:.1f} min")
