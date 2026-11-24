import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import (
    ShardDataset,
    class_set_note,
    n_labels,
    resolve_class_set,
    run_tag,
)
from util import stay_awake
from cnn.CNN_Classifier import CNN_Classifier


class NN_Classifier(nn.Module):

    def __init__(self, n_class, n_input, n1, n2, dropout_p=0):

        super().__init__()
        if not 0 <= dropout_p <= 1:
            raise ValueError(f"dropout_p must be from 0 to 1, got {dropout_p}")
        self.dropout_p = dropout_p

        self.model = nn.Sequential(
            nn.Linear(n_input, n1),  # W1 b1
            nn.ReLU(),
            nn.Linear(n1, n2),  # W2 b2
            nn.ReLU(),
            nn.Linear(n2, n_class),  # outputs raw scores (logits)
        )

    def forward(self, x):
        if self.dropout_p == 0:
            return self.model(x)
        x = self.model[0](x)
        x = self.model[1](x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.model[2](x)
        x = self.model[3](x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.model[4](x)



def build_model(model_type, n_class, n_input, n1, n2, dropout_p=0):
    if model_type in ("cnn", "cnn_v2"):
        size = int(n_input ** 0.5)
        if size * size != n_input:
            raise ValueError("CNN input must be a square image")
        return CNN_Classifier(n_class, size, dropout_p, leaky=model_type == 'cnn_v2')
    if model_type == "mlp":
        return NN_Classifier(n_class, n_input, n1, n2, dropout_p)
    raise ValueError(f"Unknown model_type: {model_type}")

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


def model_checkpoint(
    model,
    optimizer,
    n_class,
    class_set,
    image_size,
    n_input,
    n1,
    n2,
    batch_size,
    val_batch_size,
    dropout_p,
    step,
    epoch,
    train_loss,
    val_acc,
    val_loss,
    val_top5,
    patience,
    patience_max,
    best_val_loss,
    finished,
):
    return {
        "model_type": model.model_type if isinstance(model, CNN_Classifier) else "mlp",
        "n_class": n_class,
        "class_set": class_set,
        "image_size": image_size,
        "n_input": n_input,
        "n1": n1,
        "n2": n2,
        "dropout_p": dropout_p,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "val_top5": val_top5,
        "patience": patience,
        "patience_max": patience_max,
        "best_val_loss": best_val_loss,
        "finished": finished,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def save_run(run_dir, model_path, checkpoint, write_model):
    preview = {
        key: value
        for key, value in checkpoint.items()
        if key not in ("model", "optimizer")
    }
    Path(run_dir, "config_preview.json").write_text(
        json.dumps(preview, indent=2),
        encoding="utf-8",
    )
    if write_model:
        torch.save(checkpoint, model_path)


def eval_and_log(step, epoch, loss_value, train_acc):
    """
    evaluate and log for dashboard
    """
    global t
    model.eval()  # eval mode (matters later for dropout etc.)
    correct, top5, total = 0, 0, 0
    loss_sum = 0.0
    with torch.no_grad():  # no graph, no grads: faster, less memory
        for x, y in val_loader:  # walk the whole val set once
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            k = min(5, logits.size(1))
            top = logits.topk(k, dim=1).indices
            top5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)
    model.train()  # back to training mode
    val_acc = correct / total
    val_loss = loss_sum / total
    val_top5 = top5 / total

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
    t = now

    return loss_value, val_acc, val_loss, val_top5


if __name__ == "__main__":

    # ---------------------------    RESUME training    ---------------------------

    """
    RESUME = None                           // create a new model
    RESUME = "runs/12345/model.pt"          // load a model and continue training
    """
    RESUME = None
    model_type = "cnn_v2"  # "cnn_v2", "cnn" or "mlp"; resume uses checkpoint architecture

    # ---------------------------    Dynamic Config    ---------------------------

    epochs = 2
    lr = 1e-3  # -3 ~ -5
    patience_max = 10  # 3-10

    batch_size = 256
    batch_size_val = 1024
    eval_per = 100  # eval_and_log() every X batches

    # ---------------------------    Fixed Config    ---------------------------
    # Fixed when a model is created (existing models always ignore these)

    image_size = 48  # [128, 96, 64, 48, 32, 24]
    n1 = 512  # layer size
    n2 = 256

    dropout_p = 0.2  # 0 = off

    use_class_set = None  # use a smaller set for quick test
    """
    class_set = None                               // (default) train on every class
    class_set = range(1000, 1200)                  // only train on class 1000..1199
    class_set = list(range(0, 200)) + [205, 210]
    """

    # ---------------------------    Tensorboard    ---------------------------
    # tensorboard --logdir runs
    # http://localhost:6006/

    # ---------------------------    load model    ---------------------------

    t0 = t = time.time()
    label_n = n_labels()
    if RESUME is None:
        class_ids = resolve_class_set(use_class_set, label_n)
        n_class = label_n if class_ids is None else len(class_ids)
        run_dir = Path(
            f"runs/{model_type}_{image_size}_{run_tag(class_ids, label_n)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.pt"
        next_epoch = 0
        step = 0
        best_val_loss = None
        n_input = image_size * image_size
        model = build_model(
            model_type=model_type,
            n_class=n_class,
            n_input=n_input,
            n1=n1,
            n2=n2,
            dropout_p=dropout_p,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        patience = patience_max
        print("\n------------------   NEW MODEL   ------------------")
    else:
        model_path = resume_file(RESUME)
        run_dir = model_path.parent
        saved = torch.load(model_path, weights_only=False)
        model_type = saved.get("model_type", "mlp")
        n_class = saved["n_class"]
        image_size = saved["image_size"]
        n1 = saved["n1"]
        n2 = saved["n2"]
        n_input = saved["n_input"]
        saved_class_set = saved.get("class_set", range(0, n_class))
        class_ids = resolve_class_set(saved_class_set, label_n)
        got = label_n if class_ids is None else len(class_ids)
        if got != n_class:
            raise ValueError(
                f"checkpoint n_class={n_class} does not match class_set ({got})"
            )
        dropout_p = saved.get("dropout_p", dropout_p)
        next_epoch = saved["epoch"]
        step = saved["step"]
        patience = saved.get("patience", patience_max)
        patience_max = saved.get("patience_max", patience_max)
        best_val_loss = saved.get("best_val_loss", None)
        model = build_model(
            model_type=model_type,
            n_class=n_class,
            n_input=n_input,
            n1=n1,
            n2=n2,
            dropout_p=dropout_p,
        )
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        print("\n------------------   RESUME   ------------------")

    loss_fn = nn.CrossEntropyLoss()
    class_set = class_set_note(class_ids)

    print(f">> Path:    {model_path}")
    print(f">> Config:  {n_class=}, {dropout_p=}")
    print(f">> Model:\n{model}\n")

    # ---------------------------    load dataset    ---------------------------
    # train set
    dataset = ShardDataset("train", image_size=image_size, class_set=class_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # traning batch size. small = less noise = better training?
        shuffle=True,
    )

    # val set
    val_set = ShardDataset("val", image_size=image_size, class_set=class_ids)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)

    # ---------------------------    others    ---------------------------

    tb = SummaryWriter(run_dir)
    train_loss = None
    val_acc = None
    val_loss = None
    val_top5 = None
    finished = False

    # ---------------------------    training loop    ---------------------------
    print(f">> Training start:")
    print(f'(to stop after this epoch, create a "STOP" file in: {run_dir})\n')

    def checkpoint_args():
        return model_checkpoint(
            model,
            optimizer,
            n_class,
            class_ids,
            image_size,
            n_input,
            n1,
            n2,
            batch_size,
            batch_size_val,
            dropout_p,
            step,
            next_epoch,
            train_loss,
            val_acc,
            val_loss,
            val_top5,
            patience,
            patience_max,
            best_val_loss,
            finished,
        )

    def write_progress(write_model):
        save_run(run_dir, model_path, checkpoint_args(), write_model)

    with stay_awake():
        try:
            write_progress(False)
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

                # patience. one epoch, one count.
                if val_loss is not None and (
                    best_val_loss is None or val_loss < best_val_loss
                ):
                    best_val_loss = val_loss
                    patience = patience_max
                    torch.save(checkpoint_args(), run_dir / "best.pt")
                    print(f">> Best val_loss={best_val_loss:.3f}, saved best.pt")
                elif val_loss is not None:
                    patience -= 1

                write_progress(False)
                if val_loss is not None and patience <= 0:
                    print(f">> Patience {patience_max} reached, STOP training")
                    break

                # manual early STOP
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
