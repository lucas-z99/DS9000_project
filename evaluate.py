from NN_Classifier import NN_Classifier, resume_file, resolve_classes
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from dataloader import (
    ShardDataset,
    classes_note,
    n_labels,
    resolve_classes,
    run_tag,
)
from a1_prepare_data import CLASS_DICT_DIR
import csv
from collections import Counter

# ----------------    CONFIG    ----------------

RESUME = "runs/48_3922_20261115_145210 copy/model.pt"
testset_name = "val"  # test, val
CN_begin = 155  # before this are +-*\abcd


def evaluate(model, loader, n_class):

    c_appear = torch.zeros(n_class)  # appearance [class1, class2, class3 ..]
    c_correct = torch.zeros(n_class)  # correctness [class1, class2,  class3 ...]
    c_mistaken_as = [
        Counter() for _ in range(n_class)
    ]  # mistaken as [class1, class2,  class3 ...]

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            predict = logits.argmax(dim=1)  # index of max element
            bool_mask = predict == y  # [T, T, T, F...]

            c_correct += torch.bincount(y[bool_mask], minlength=n_class)
            c_appear += torch.bincount(y, minlength=n_class)

            # mistake classA as this class [class1, class2, class3 ..]
            # c_mistaken_as[y[~bool_mask]].append(predict[~bool_mask])
            for true_c, pred_c in zip(
                y[~bool_mask].tolist(), predict[~bool_mask].tolist()
            ):
                c_mistaken_as[true_c][pred_c] += 1

    c_acc = c_correct / c_appear  # acc [class1, class2, class3 ..]
    c_dict = CLASS_DICT_DIR.read_text(encoding="utf-8").splitlines()  # {class:label}

    # save a csv for eyeball
    rows = [
        (
            i,
            int(c_appear[i]),
            int(c_correct[i]),
            c_acc[i].item(),
            c_dict[i],
            "".join(c_dict[p] * n + " " for p, n in c_mistaken_as[i].most_common()),
        )
        for i in range(CN_begin, n_class)
    ]
    rows.sort(key=lambda r: r[4])  # worst accuracy first
    with open("eval_result_Classifier.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "n", "n_correct", "acc", "charactor", "mistaken as"])
        writer.writerows(rows)

    print("DONE!")


if __name__ == "__main__":

    # ----------------    load checkpoint & model    ----------------
    saved = torch.load(resume_file(RESUME), map_location="cpu")
    saved_config = saved["config"]
    n_class = saved_config["n_class"]
    n_input = saved_config["n_input"]
    image_size = int(n_input**0.5)
    n1 = saved_config["n1"]
    n2 = saved_config["n2"]

    print(f"{saved.keys()=}")

    model = NN_Classifier(
        n_class=n_class,
        n_input=n_input,
        n1=n1,
        n2=n2,
        dropout=False,  # turn off
        dropout_p=0,
    )
    model.load_state_dict(saved["model"])
    model.eval()
    print(f"{model=}")

    # ----------------    test data    ----------------
    dataset = ShardDataset(
        testset_name,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    print(f"{len(dataset)=}")

    # ----------------    evaluate    ----------------

    evaluate(model, loader, n_class)
