import json
from datetime import datetime
from pathlib import Path
import torch.nn as nn
from dataloader import ShardDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def resume_file(resume):
    """
    resume is the model .pt file. A folder path is rejected.
    """

    path = Path(resume)
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise FileNotFoundError(
            f"resume must be an existing .pt model file, got {resume}"
        )
    return path


def save_run(
    run_dir,
    model_path,
    model,
    optimizer,
    config,
    step,
    epoch,
    last_loss,
    last_val_acc,
    last_val_loss,
    last_top5,
    finished,
    write_model,
):
    """
    config.json describes the folder.
    model.pt is the weights and the Adam state.
    """

    note = {
        "note": (
            "This folder is one training run. "
            "model.pt is the checkpoint. "
            "TensorBoard events are the curves."
        ),
        "finished": finished,
        "model_file": Path(model_path).name,
        "step": step,
        "epoch": epoch,
        "last_loss": last_loss,
        "last_val_acc": last_val_acc,
        "last_val_loss": last_val_loss,
        "last_top5": last_top5,
        **config,
    }
    Path(run_dir, "config.json").write_text(
        json.dumps(note, indent=2),
        encoding="utf-8",
    )
    if write_model:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "step": step,
                "epoch": epoch,
                "last_loss": last_loss,
                "last_val_acc": last_val_acc,
                "last_val_loss": last_val_loss,
                "last_top5": last_top5,
            },
            model_path,
        )


def evaluate(model, loader):
    model.eval()  # eval mode (matters later for dropout etc.)
    correct, top5, total = 0, 0, 0
    loss_sum = 0.0

    with torch.no_grad():  # no graph, no grads: faster, less memory
        for x, y in loader:  # walk the whole val set once
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            k = min(5, logits.size(1))
            top = logits.topk(k, dim=1).indices
            top5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)

    model.train()  # back to training mode
    return correct / total, loss_sum / total, top5 / total


def eval_and_log(step, epoch, loss_value, train_acc, t):
    """
    run a evaluation() and log the curves
    """
    global last_loss, last_val_acc, last_val_loss, last_top5
    val_acc, val_loss, val_top5 = evaluate(model, val_loader)
    now = time.time()
    sec = now - t
    print(
        f"[{step}] epoch={epoch} loss={loss_value:.3f}, acc={train_acc:.3f}, "
        f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, val_top5={val_top5:.3f}, t={sec:.1f}"
    )
    tb.add_scalar("loss/val", val_loss, step)
    tb.add_scalar("acc/val", val_acc, step)
    tb.add_scalar("acc/val_top5", val_top5, step)
    tb.flush()
    last_loss = loss_value
    last_val_acc = val_acc
    last_val_loss = val_loss
    last_top5 = val_top5
    return now


if __name__ == "__main__":

    # -----------------------   config   -----------------------

    n_class = 3922
    epochs = 2
    eval_per = 100  # evaluate() and log every X batches
    image_size = 64  # [128, 96, 64, 48, 32, 24]
    n1 = 512
    n2 = 256
    batch_size = 256
    batch_size_val = 1024
    lr = 1e-3

    # -----------------------   RESUME training   -----------------------
    """
    RESUME = None                   // train a new model
    RESUME = "runs/<run>/model.pt"  // continue training
    """

    RESUME = "runs/64_3922_20261114_191011/model.pt"

    """
    "runs/64_3922_20261114_191011/model.pt"
    """

    # -----------------------   Tensorboard   -----------------------

    """
    Run local server:
    tensorboard --logdir runs
    http://localhost:6006/    
    """

    # ----------------------------------------------------------

    if RESUME is None:
        run_dir = Path(
            f"runs/{image_size}_{n_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pt"
        next_epoch = 0
        step = 0
        model = NN_Classifier(
            n_class=n_class,
            n_input=image_size * image_size,
            n1=n1,
            n2=n2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model_path = resume_file(RESUME)
        run_dir = model_path.parent
        saved = torch.load(model_path, weights_only=False)
        saved_config = saved["config"]
        n_class = saved_config["n_class"]
        image_size = saved_config["image_size"]
        n1 = saved_config["n1"]
        n2 = saved_config["n2"]
        lr = saved_config["lr"]
        next_epoch = saved["epoch"]
        step = saved["step"]
        model = NN_Classifier(
            n_class=n_class,
            n_input=saved_config["n_input"],
            n1=n1,
            n2=n2,
        )
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(saved["optimizer"])
        print(f">> resume: {model_path}")

    loss_fn = nn.CrossEntropyLoss()
    print(f"Model:\n{model}")

    # train set
    dataset = ShardDataset("train", image_size=image_size, cap_y=n_class)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # traning batch size. small = less noise = better training?
        shuffle=True,
    )

    # val set
    val_set = ShardDataset("val", image_size=image_size, cap_y=n_class)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)

    # logs
    config = {
        "n_class": n_class,
        "epochs": epochs,
        "eval_per": eval_per,
        "image_size": image_size,
        "n_input": image_size * image_size,
        "n1": n1,
        "n2": n2,
        "batch_size": batch_size,
        "val_batch_size": batch_size_val,
        "lr": lr,
        "optimizer": "Adam",
    }
    print(f">> Config: {n_class=}, {image_size=}, {epochs=}, {RESUME=}")
    print(f">> TensorBoard: {run_dir}")

    t0 = t = time.time()
    tb = SummaryWriter(run_dir)
    tb.add_text(
        "config",
        "\n".join(
            [
                f"n_class: {n_class}",
                f"epochs: {epochs}",
                f"eval_per: {eval_per}",
                f"image_size: {image_size}",
                f"n1: {n1}",
                f"n2: {n2}",
                f"batch_size: {batch_size}",
                f"val_batch_size: {batch_size_val}",
                f"lr: {lr}",
                "optimizer: Adam",
            ]
        ),
    )
    last_loss = None
    last_val_acc = None
    last_val_loss = None
    last_top5 = None
    finished = False

    # training loop
    print(">> Training start:")
    print(f">> Total Batch: {step} -> {step + 938679/batch_size * epochs:.0f}")

    try:
        save_run(
            run_dir,
            model_path,
            model,
            optimizer,
            config,
            step,
            next_epoch,
            last_loss,
            last_val_acc,
            last_val_loss,
            last_top5,
            finished=False,
            write_model=False,
        )
        for epoch in range(next_epoch, next_epoch + epochs):
            last_logged = None
            for x, y in loader:
                optimizer.zero_grad()  # clear gradiants

                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                train_acc = (y_hat.argmax(dim=1) == y).float().mean().item()
                tb.add_scalar("loss/train", loss_value, step)
                tb.add_scalar("acc/train", train_acc, step)
                if step % eval_per == 0:
                    t = eval_and_log(step, epoch, loss_value, train_acc, t)
                    last_logged = step
                step += 1

            if last_logged != step - 1:
                t = eval_and_log(step - 1, epoch, loss_value, train_acc, t)
            next_epoch = epoch + 1

        if last_val_acc is not None:
            tb.add_hparams(
                {
                    "n_class": n_class,
                    "epochs": epochs,
                    "eval_per": eval_per,
                    "image_size": image_size,
                    "n1": n1,
                    "n2": n2,
                    "batch_size": batch_size,
                    "val_batch_size": batch_size_val,
                    "lr": lr,
                    "optimizer": "Adam",
                },
                {
                    "acc/val": last_val_acc,
                    "acc/val_top5": last_top5,
                    "loss/train": last_loss,
                    "loss/val": last_val_loss,
                },
            )
        finished = True
    finally:
        save_run(
            run_dir,
            model_path,
            model,
            optimizer,
            config,
            step,
            next_epoch,
            last_loss,
            last_val_acc,
            last_val_loss,
            last_top5,
            finished,
            write_model=True,
        )
        tb.close()

    print(f"total t={(time.time() - t0)/60:.1f} min")
