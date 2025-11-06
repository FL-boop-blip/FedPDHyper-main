FedPDHyper: Adaptive Federated Primal-Dual Learning with Hypergradient Ascent
-----------------------------------------------------------------------------

Overview
- This repository provides a federated learning simulation framework and multiple method implementations (including FedPDHyper, FedAvg, FedDyn, A_FedPD, FedSpeed, etc.), supporting MNIST, CIFAR-10/100, and AG_News datasets with IID and non-IID splits.
- The entry script is `train.py`. You can control the dataset, model, method, and training configuration via command-line arguments. `run.sh` contains a ready-to-run example.

Environment and Installation
- Python 3.8+ (3.10/3.11/3.12 recommended)
- Install dependencies:
  - Use `requirements.txt` in this directory
  - Command: `pip install -r requirements.txt`

Quick Start
- Run the example script directly:
  - `bash run.sh`
- Or specify arguments manually (example with CIFAR-10, ResNet18, FedPDHyper):
  - `python train.py --method FedPDHyper --dataset CIFAR10 --model ResNet18 --comm-rounds 300 --batchsize 50 --local-learning-rate 0.1 --local-epochs 5 --non-iid --lamb 0.1`

Common Arguments (excerpt; see `train.py`)
- `--dataset`: `mnist` | `CIFAR10` | `CIFAR100`
- `--model`: `mnist_2NN` | `ResNet18`
- `--method`: `FedAvg` | `FedDyn` | `FedSpeed` | `A_FedPD` | `A_FedPD_SAM` | `FedPDHyper` | `FedPDHyper_SAM` | `FedPDHyper_SCA`
- Training: `--comm-rounds` (global rounds), `--local-epochs` (local iterations), `--batchsize`, `--local-learning-rate`, `--global-learning-rate`
- Data split: `--non-iid`, `--split-rule` (`Dirichlet`/`Pathological`), `--split-coef`
- Others: `--total-client`, `--active-ratio`, `--seed`, `--cuda`, `--out-file`, `--save-model`

Outputs and Logs
- Training logs and results are saved under `out/` by default (can be changed via `--out-file`).
- Example structure: `out/<Method>/T=<Rounds>/<Dataset-Setting>/active-<ratio>/`
  - `Performance/`: saves test results `tst-<Method>.npy`
  - `Divergence/`: consistency metrics during training
  - `Time/`: per-round time
  - `Model/`: best model `best.pth`
  - Root contains `summary.txt`

Project Structure (root)
- `train.py`: main entry; parses arguments and starts federated training
- `client/`: client-side local training logic (e.g., `fedpdhyper.py`, `fedavg.py`)
- `server/`: server-side coordination and aggregation (e.g., `FedPDHyper.py`, `FedAvg.py`, and the common base `server.py`)
- `optimizer/`: optional optimizers and variants (SAM/LESAM/ESAM, etc.)
- `dataset.py`: dataset download/preprocessing and IID/non-IID splitting (Dirichlet/Pathological, etc.)
- `models.py`: model definitions (ResNet18, LeNet, etc.)
- `utils.py` / `utils_models.py`: utilities and helper modules
- `run.sh`: example run script
- `requirements.txt`: dependency list

Data
- `mnist`/`CIFAR10`/`CIFAR100` are automatically downloaded via `torchvision` to `./Data/Raw/`.

Notes
- For GPU usage, set `--cuda <gpu_id>` (e.g., `--cuda 0`); it falls back to CPU if no GPU is available.
- Use `--seed` to reproduce data splits and sampling.
