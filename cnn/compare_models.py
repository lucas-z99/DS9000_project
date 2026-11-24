"""Train both architectures for 100 epochs and compare final test accuracy."""
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import ShardDataset, n_labels
from NN_Classifier import build_model
from util import stay_awake
from cnn.experiment_paths import model_run_dir


def score(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    if not total:
        raise ValueError("Empty evaluation split")
    return correct / total


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def update_best(checkpoint, path):
    """Keep the earliest epoch on ties; select using validation only."""
    accuracy = checkpoint['history'][-1]['val_acc']
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=False)
        if accuracy <= previous['history'][-1]['val_acc']:
            return
    save_checkpoint(checkpoint, path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Existing comparison directory to resume')
    parser.add_argument('--cnn-version', choices=['cnn', 'cnn_v2'], default='cnn_v2')
    args = parser.parse_args()
    root = args.run_dir or Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('comparison_%Y%m%d_%H%M%S')
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; output: {root}', flush=True)
    # This experiment needs only 48px shards, avoiding five unused resolutions.
    import a1_prepare_data
    a1_prepare_data.IMG_RESIZE[:] = [48]
    datasets = {s: ShardDataset(s, image_size=48) for s in ('train', 'val', 'test')}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError('All three splits must contain data')
    validation = DataLoader(datasets['val'], batch_size=256)
    test = DataLoader(datasets['test'], batch_size=256)
    results = {}
    for kind in ('mlp', 'cnn'):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = dict(model_type=args.cnn_version if kind == 'cnn' else kind, n_class=n_labels(), n_input=48*48,
                      n1=512, n2=256, dropout_p=0.2)
        model = build_model(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        artifact_dir = model_run_dir(root, kind)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f'{kind}.pt'
        best_path = artifact_dir / f'{kind}_best.pt'
        history = []
        if path.exists():
            saved = torch.load(path, map_location=device, weights_only=False)
            if saved.get('model_type', 'mlp') != config['model_type']:
                raise ValueError('Checkpoint architecture differs; use a new run directory')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            history = saved['history']
            if not best_path.exists() and history[-1]['val_acc'] < max(r['val_acc'] for r in history):
                raise RuntimeError('Historical best weights are missing; cannot reconstruct from metrics')
            update_best(saved, best_path)
            print(f'Resuming {kind} after epoch {len(history)}', flush=True)
        for epoch in range(len(history), 100):
            epoch_start = time.monotonic()
            # Epoch-specific seeds preserve batch order across models and restarts.
            torch.manual_seed(42 + epoch)
            generator = torch.Generator().manual_seed(42 + epoch)
            loader = DataLoader(datasets['train'], batch_size=256,
                                shuffle=True, generator=generator)
            model.train()
            loss_sum = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)
                total += len(y)
            row = dict(epoch=epoch + 1, train_loss=loss_sum / total,
                       val_acc=score(model, validation, device))
            row['seconds'] = time.monotonic() - epoch_start
            history.append(row)
            checkpoint = dict(**config, image_size=48, class_set=None,
                              epoch=epoch+1, model=model.state_dict(),
                              optimizer=optimizer.state_dict(), history=history)
            update_best(checkpoint, best_path)
            save_checkpoint(checkpoint, path)
            (artifact_dir / f'{kind}_history.json').write_text(json.dumps(history, indent=2))
            print(kind, row, flush=True)
        results[kind] = dict(epochs=100, final_val_acc=history[-1]['val_acc'],
                             final_test_acc=score(model, test, device))
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best['model'])
        results[kind].update(best_epoch=best['epoch'],
                             best_val_acc=best['history'][-1]['val_acc'],
                             best_test_acc=score(model, test, device))
        (root / 'results.json').write_text(json.dumps(results, indent=2))
    difference = results['cnn']['final_test_acc'] - results['mlp']['final_test_acc']
    print(json.dumps(results, indent=2), flush=True)
    print(f'CNN minus MLP test accuracy: {difference * 100:.3f} percentage points')


if __name__ == '__main__':
    with stay_awake():
        main()
