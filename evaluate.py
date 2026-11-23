from collections import Counter

import torch

from a1_prepare_data import CLASS_DICT_DIR
from dataloader import DeviceLoader, ShardDataset
from NN_Classifier import NN_Classifier, resume_file
from util import get_device

DEVICE = get_device()

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
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            predict = logits.argmax(dim=1).cpu()  # index of max element
            y = y.cpu()  # counting below happens on CPU
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

    # # save a csv for eyeball
    # rows = [
    #     (
    #         i,
    #         int(c_appear[i]),
    #         int(c_correct[i]),
    #         c_acc[i].item(),
    #         c_dict[i],
    #         "".join(c_dict[p] * n + " " for p, n in c_mistaken_as[i].most_common()),
    #     )
    #     for i in range(CN_begin, n_class)
    # ]
    # rows.sort(key=lambda r: r[4])  # worst accuracy first
    # with open("eval_result_Classifier.csv", "w", newline="", encoding="utf-8-sig") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["label", "n", "n_correct", "acc", "charactor", "mistaken as"])
    #     writer.writerows(rows)

    print("DONE!")
    return c_acc


def make_ablate_hook(unit):
    def hook(module, input, output):
        output = output.clone()
        output[:, unit] = 0  # switch this unit off for every image in the batch
        return output

    return hook


def load_model(resume):
    saved = torch.load(resume_file(resume), map_location="cpu")
    model = NN_Classifier(
        n_class=saved["n_class"],
        n_input=saved["n_input"],
        n1=saved["n1"],
        n2=saved["n2"],
        dropout_p=0,
    )
    model.load_state_dict(saved["model"])
    model.to(DEVICE)
    model.eval()
    return model, saved


def load_loader(split, image_size, batch_size=1024):
    dataset = ShardDataset(split, image_size=image_size)
    # whole split kept on the GPU (ablate.py walks it 512 times)
    return DeviceLoader(dataset, batch_size, shuffle=False, device=DEVICE)


def count_correct(model, loader, n_class):
    c_correct = torch.zeros(n_class, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            predict = model(x).argmax(dim=1)
            c_correct += torch.bincount(y[predict == y], minlength=n_class).cpu()
    return c_correct


if __name__ == "__main__":

    model, saved = load_model(RESUME)
    loader = load_loader(testset_name, int(saved["n_input"] ** 0.5))
    evaluate(model, loader, saved["n_class"])
