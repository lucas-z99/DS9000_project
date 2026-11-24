import tempfile
from pathlib import Path
import torch
from cnn.compare_models import update_best
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / 'best.pt'
    for epoch, acc in [(1, .6), (2, .5), (3, .7), (4, .7)]:
        update_best(dict(epoch=epoch, history=[dict(val_acc=acc)]), p)
    assert torch.load(p, weights_only=False)['epoch'] == 3
saved = torch.load(Path(__file__).parent / 'runs/fixed_100epochs/mlp.pt', map_location='cpu', weights_only=False)
assert saved['history'][-1]['val_acc'] == max(r['val_acc'] for r in saved['history'])
print('Best selection passed; existing checkpoint is historical best:', saved['epoch'])
