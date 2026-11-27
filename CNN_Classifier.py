"""
CNN training file. Same data, optimizer, early stopping and seeds as the MLP
(NN_Classifier.py); everything that can be shared is imported from there.

    python3 CNN_Classifier.py --n1 1024 --n2 512 --seed 1
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloader import (
    DeviceLoader,
    ShardDataset,
    class_set_note,
    n_labels,
    resolve_class_set,
    run_tag,
)
from NN_Classifier import NN_Classifier, model_checkpoint, resume_file, save_run
from util import get_device, stay_awake


class CNN_Classifier(nn.Module):
    """
    3 conv blocks + the SAME fully connected head as the MLP.

    input 1 x 48 x 48
      -> [conv3x3 -> BN -> ReLU -> maxpool]  32 ch -> 24 x 24
      -> [conv3x3 -> BN -> ReLU -> maxpool]  64 ch -> 12 x 12
      -> [conv3x3 -> BN -> ReLU -> maxpool] 128 ch ->  6 x  6
      -> flatten (128 * 6 * 6 = 4608) -> BatchNorm1d
      -> NN_Classifier head: n1 -> n2 -> n_class (with the same dropout)

    The only difference from the MLP is the conv part in front, so the
    comparison shows what convolution adds.
    """

    channels = (32, 64, 128)

    def __init__(self, n_class, image_size, n1, n2, dropout_p=0, input_mean=0.0, input_std=1.0):
        super().__init__()
        # input standardization (computed from the train set, saved with the model).
        # Without it the first BatchNorm sees variance ~1e-4 (near its eps) and
        # val accuracy jumps around wildly in eval mode.
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be a multiple of 8, got {image_size}")

        blocks = []
        c_in = 1
        for c_out in self.channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # BN has the bias
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),  # halves height and width
            ]
            c_in = c_out
        side = image_size // 8  # three pools
        # normalize the flattened conv features before the head. Without it the
        # features are large (max ~20) and ~95% of the head's first ReLU units die.
        # eps=1e-2 (not 1e-5): ~60% of features have variance ~1e-6, the default
        # eps blows them up and eval mode breaks.
        blocks += [nn.Flatten(), nn.BatchNorm1d(c_in * side * side, eps=1e-2)]
        self.features = nn.Sequential(*blocks)

        self.head = NN_Classifier(
            n_class=n_class,
            n_input=c_in * side * side,
            n1=n1,
            n2=n2,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        if x.dim() == 3:  # (batch, H, W) -> (batch, 1, H, W)
            x = x.unsqueeze(1)
        x = (x - self.input_mean) / self.input_std
        return self.head(self.features(x))


def eval_and_log(step, epoch, loss_value, train_acc):
    """
    evaluate and log for dashboard (same as NN_Classifier.py)
    """
    global t
    model.eval()  # eval mode: dropout off, BatchNorm uses running stats
    correct, top5, total = 0, 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            top = logits.topk(min(5, logits.size(1)), dim=1).indices
            top5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)
    model.train()
    val_acc = correct / total
    val_loss = loss_sum / total
    val_top5 = top5 / total

    now = time.time()
    print(
        f"[{step}] epoch={epoch} loss={loss_value:.3f}, acc={train_acc:.3f}, "
        f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, val_top5={val_top5:.3f}, t={now - t:.1f}"
    )
    tb.add_scalar("loss/val", val_loss, step)
    tb.add_scalar("acc/val", val_acc, step)
    tb.add_scalar("acc/val_top5", val_top5, step)
    tb.flush()
    t = now

    return loss_value, val_acc, val_loss, val_top5


if __name__ == "__main__":

    # ---------------------------    RESUME training    ---------------------------

    """
    RESUME = None                           // create a new model
    RESUME = "runs/cnn_12345/model.pt"      // load a model and continue training
    """
    RESUME = None

    # ---------------------------    Dynamic Config    ---------------------------
    # keep these the same as NN_Classifier.py for a fair comparison

    epochs = 200
    lr = 1e-3
    patience_max = 5

    batch_size = 1024
    batch_size_val = 1024
    eval_per = 500  # eval_and_log() every X batches

    # ---------------------------    Fixed Config    ---------------------------

    image_size = 48
    n1 = 1024  # head layer size (same meaning as the MLP)
    n2 = 512
    dropout_p = 0.3  # in the head only, same as the MLP
    seed = 1
    use_class_set = None

    # ---------------------------    command line    ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", type=int)
    parser.add_argument("--n2", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    n1 = args.n1 if args.n1 is not None else n1
    n2 = args.n2 if args.n2 is not None else n2
    seed = args.seed if args.seed is not None else seed
    epochs = args.epochs if args.epochs is not None else epochs

    # ---------------------------    load model    ---------------------------

    t0 = t = time.time()
    device = get_device()
    print(f">> Device:  {device}")
    label_n = n_labels()
    if RESUME is None:
        class_ids = resolve_class_set(use_class_set, label_n)
        n_class = label_n if class_ids is None else len(class_ids)
        run_dir = Path(
            f"runs/cnn_{image_size}_{run_tag(class_ids, label_n)}_{n1}-{n2}_s{seed}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pt"
        next_epoch, step, best_val_loss = 0, 0, None
        torch.manual_seed(seed)
        stats = ShardDataset("train", image_size=image_size, class_set=class_ids)
        sample = DeviceLoader(stats, 200_000, False, device, flatten=False)
        x_mean, x_std = (lambda x: (x.mean().item(), x.std().item()))(next(iter(sample))[0])
        del stats, sample
        print(f">> Input:   mean={x_mean:.4f}, std={x_std:.4f}")
        model = CNN_Classifier(
            n_class, image_size, n1, n2, dropout_p, x_mean, x_std
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        patience = patience_max
        print("\n------------------   NEW CNN MODEL   ------------------")
    else:
        model_path = resume_file(RESUME)
        run_dir = model_path.parent
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        n_class, image_size = saved["n_class"], saved["image_size"]
        n1, n2, dropout_p = saved["n1"], saved["n2"], saved["dropout_p"]
        class_ids = resolve_class_set(saved.get("class_set"), label_n)
        seed = saved.get("seed", seed)
        torch.manual_seed(seed + saved["epoch"])
        next_epoch, step = saved["epoch"], saved["step"]
        patience = saved.get("patience", patience_max)
        patience_max = saved.get("patience_max", patience_max)
        best_val_loss = saved.get("best_val_loss")
        model = CNN_Classifier(n_class, image_size, n1, n2, dropout_p)
        model.load_state_dict(saved["model"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        print("\n------------------   RESUME CNN   ------------------")

    loss_fn = nn.CrossEntropyLoss()
    n_input = image_size * image_size
    n_params = sum(p.numel() for p in model.parameters())

    print(f">> Path:    {model_path}")
    print(f">> Config:  {n_class=}, {n1=}, {n2=}, {dropout_p=}, {seed=}")
    print(f">> Params:  {n_params:,}")
    print(f">> Model:\n{model}\n")

    # ---------------------------    load dataset    ---------------------------
    # same loader as the MLP, but keep the 2D image (no flatten)
    dataset = ShardDataset("train", image_size=image_size, class_set=class_ids)
    loader = DeviceLoader(dataset, batch_size, True, device, flatten=False)
    val_set = ShardDataset("val", image_size=image_size, class_set=class_ids)
    val_loader = DeviceLoader(val_set, batch_size_val, False, device, flatten=False)

    tb = SummaryWriter(run_dir)
    train_loss = val_acc = val_loss = val_top5 = None
    finished = False

    # ---------------------------    training loop    ---------------------------
    print(f">> Training start:")
    print(f'(to stop after this epoch, create a "STOP" file in: {run_dir})\n')

    def checkpoint_args():
        checkpoint = model_checkpoint(
            model, optimizer, n_class, class_ids, image_size, n_input, n1, n2,
            batch_size, batch_size_val, dropout_p, step, next_epoch, train_loss,
            val_acc, val_loss, val_top5, patience, patience_max, best_val_loss,
            finished,
        )
        checkpoint["seed"] = seed
        checkpoint["model_type"] = "cnn"
        checkpoint["n_params"] = n_params
        return checkpoint

    def write_progress(write_model):
        save_run(run_dir, model_path, checkpoint_args(), write_model)

    with stay_awake():
        try:
            write_progress(False)
            for epoch in range(next_epoch, next_epoch + epochs):
                last_logged = None
                for x, y in loader:
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    optimizer.step()

                    loss_value = loss.item()
                    train_acc = (y_hat.argmax(dim=1) == y).float().mean().item()
                    tb.add_scalar("loss/train", loss_value, step)
                    tb.add_scalar("acc/train", train_acc, step)
                    tb.add_scalar("lr", lr, step)
                    if step % eval_per == 0:
                        train_loss, val_acc, val_loss, val_top5 = eval_and_log(
                            step, epoch, loss_value, train_acc
                        )
                        last_logged = step
                    step += 1

                if last_logged != step - 1:
                    train_loss, val_acc, val_loss, val_top5 = eval_and_log(
                        step - 1, epoch, loss_value, train_acc
                    )
                next_epoch = epoch + 1

                # patience. one epoch, one count. (same rule as the MLP)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = patience_max
                    torch.save(checkpoint_args(), run_dir / "best.pt")
                    print(f">> Best val_loss={best_val_loss:.3f}, saved best.pt")
                else:
                    patience -= 1

                write_progress(False)
                if patience <= 0:
                    print(f">> Patience {patience_max} reached, STOP training")
                    break

                stop_file = run_dir / "STOP"
                if stop_file.is_file():
                    stop_file.unlink()
                    print(f">> STOP found, ending...")
                    break

            finished = True
        finally:
            write_progress(True)
            tb.close()

    print(f"Total t={(time.time() - t0)/60:.1f} min")
