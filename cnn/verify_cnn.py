import tempfile
from pathlib import Path
import torch
from NN_Classifier import build_model
from evaluate import load_model

torch.set_num_threads(2)
for kind in ("cnn", "cnn_v2", "mlp"):
    m = build_model(kind, 7, 48*48, 32, 16, 0.2)
    x = torch.rand(2, 48*48)
    out = m(x)
    assert out.shape == (2, 7)
    torch.nn.functional.cross_entropy(out, torch.tensor([1, 2])).backward()
    m.eval()
    cfg = dict(n_class=7, n_input=48*48, n1=32, n2=16, model=m.state_dict())
    if kind in ("cnn", "cnn_v2"):
        cfg["model_type"] = kind
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
        p = Path(d)/"model.pt"
        torch.save(cfg, p)
        loaded, _ = load_model(p)
        assert torch.allclose(m(x), loaded(x))
    print(kind, "forward/backward and checkpoint reload passed")
for size in (24, 32, 48, 64, 96, 128):
    m = build_model("cnn", 7, size*size, 32, 16)
    assert m(torch.rand(1, size*size)).shape == (1, 7)
print("All image sizes passed")
