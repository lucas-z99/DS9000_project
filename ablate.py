import csv, time
import torch
from a1_prepare_data import CLASS_DICT_DIR
from evaluate import load_model, load_loader, count_correct, make_ablate_hook

# ----------------    can I poke your brain?    ----------------

RESUME = "runs/48_3922_20261115_145210 copy/model.pt"
SPLIT = "val"
CN_begin = 155  # before this are symbols and letters
LAYER = 1  # model.model[1] = first ReLU

if __name__ == "__main__":
    model, cfg = load_model(RESUME)
    n_class, n_units = cfg["n_class"], cfg["n1"]
    loader = load_loader(SPLIT, int(cfg["n_input"] ** 0.5))
    c_dict = CLASS_DICT_DIR.read_text(encoding="utf-8").splitlines()

    print("[Notice] Baseline...")
    base = count_correct(model, loader, n_class)

    print("[Warning] Brain poking...")
    table = torch.zeros(n_class, n_units, dtype=torch.long)
    for unit in range(n_units):
        t0 = time.time()
        handle = model.model[LAYER].register_forward_hook(make_ablate_hook(unit))
        try:
            table[:, unit] = count_correct(model, loader, n_class)
        finally:
            handle.remove()
        print(f"unit {unit + 1}/{n_units}  {time.time() - t0:.1f}s", flush=True)

    torch.save({"base": base, "table": table}, f"ablation_layer{LAYER}.pt")

    with open(f"ablation_layer{LAYER}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "char", "base"] + [f"u{u}" for u in range(n_units)])
        writer.writerow(
            ["", "TOTAL", int(base[CN_begin:].sum())]
            + table[CN_begin:].sum(dim=0).tolist()
        )
        for i in range(CN_begin, n_class):
            writer.writerow([i, c_dict[i], int(base[i])] + table[i].tolist())
