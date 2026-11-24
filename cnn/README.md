# CNN and MLP comparison experiment

This folder contains the code and outputs added for the 100-epoch comparison.
Run commands from the project root with the project Python interpreter.

## Contents

- `CNN_Classifier.py`: original and repaired CNN architectures.
- `compare_models.py`: 100-epoch MLP/CNN training, best-checkpoint selection and test evaluation.
- `experiment_paths.py`: artifact path resolution.
- `plot_training_comparison.py`: English comparison plots.
- `diagnose_cnn.py`: activation and gradient inspection.
- `verify_cnn.py`, `verify_cnn_learning.py`, `verify_best_checkpoint.py`: verification tools.
- `runs/fixed_100epochs/`: both models' weights, histories and comparison results.
- `figures/`: comparison curves in PNG, PDF and SVG.

## Saved results

| Model | Checkpoint | Epoch | Validation accuracy | Test accuracy |
|---|---|---:|---:|---:|
| MLP | `mlp.pt`, `mlp_best.pt` | 100 | 68.87% | 62.36% |
| Repaired CNN best | `cnn_best.pt` | 36 | 86.02% | 81.03% |
| Repaired CNN final | `cnn.pt` | 100 | 85.25% | 80.25% |

All checkpoints above are in `runs/fixed_100epochs/` relative to this folder.
That directory also contains `mlp_history.json`, `cnn_history.json`, `results.json`,
`training.log`, `errors.log` and the historical `process_id.txt` (training is complete).

## Commands

```text
python -m cnn.plot_training_comparison
python -m cnn.verify_cnn
python -m cnn.verify_best_checkpoint
python -m cnn.diagnose_cnn
python -m cnn.verify_cnn_learning
python -m cnn.compare_models --run-dir cnn/runs/fixed_100epochs --cnn-version cnn_v2
```

The final command loads this completed experiment; it does not start another 100 epochs.
For a new experiment, omit `--run-dir`. Learning verification requires CUDA.

The repaired CNN uses LeakyReLU (slope 0.1) to avoid the zero activations and
zero upstream gradients observed in the original ReLU CNN. Old `cnn` checkpoints
retain their original activation behavior; the repaired model type is `cnn_v2`.

The original MLP implementation (`NN_Classifier.py`), shared data preparation and
loading code, datasets and Python environment remain at the project root.
Downloaded archives remain in `downloads/`.

The repository includes JSON metrics and exported figures. Large checkpoint files, datasets and run logs are local-only and excluded from Git.
