# aml2026-group-10

Group Project Advanced Machine Learning

! Current baseline results are all without finetunning and with out the actual eval (75/15/15) split => Reruns with finetunning

## TODOs
- Run HP search for `cnn_transformer` (and rerun baselines on 70/15/15 split for fair comparison) (can take days)
    - Setup proper hyperparameter finetunning for all models, let all run 
- Fill result tables in `docs/CNN_TRANSFORMER.md`, `docs/VIT_BASELINE.md`, `docs/RF_BASELINE.md`
- Check against [problemsetting.md](PROBLEMSETTING.md) for missing steps
- Create Kaggle submission notebook (deadline **June 3, 2026**)
    - [Submission FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)
    - CPU Notebook <= 90 minutes run-time
    - Submission file called submission.csv

Optional:
- Test submit to kaggle, final submissions **June 3, 2026**
- Check [perch_v2_cpu](https://www.kaggle.com/models/google/bird-vocalization-classifier) suitability
- Check top kaggle notebooks for inspo / other models
- Pretraining ViT for Audio?


See [PROBLEMSETTING.md](PROBLEMSETTING.md) for the full problem definition, formal setup,
proposed architecture, baselines, evaluation protocol, and HP tuning plan.

---

## Quick start
Use the attached SwitchDrive for persistant storage:
- Final model weights
- Parameter and training run data

On Renku, download the kaggle dataset directly this is quicker than the connection to SwitchDrive
Either with this [script](scripts/stash_birdclef_data.py) or kaggle competitions download -c birdclef-2026

### Renku: GPU / PyTorch fix

Renku's Paketo buildpack bakes a pre-installed torch (CUDA 13) into a read-only layer that shadows anything `pip install` puts in the venv. The bundled torch requires a newer NVIDIA driver than the cluster provides (driver supports CUDA 12.4 max), so `torch.cuda.is_available()` returns `False` out of the box.

**Workaround — prepend the venv to `PYTHONPATH` once per session:**

```bash
export PYTHONPATH=/home/renku/work/.venv/lib/python3.11/site-packages:$PYTHONPATH
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-cache-dir
```

After this `torch.cuda.is_available()` returns `True`. Add the `export` to `~/.bashrc` to avoid repeating it each session. The buildpack layer is read-only so `pip uninstall` of the old torch will fail — that is expected.

When adding models use train.py and create file under /models/.

Example for CNN:
```bash
pip install -r requirements.txt # if not renku session with all installed
python scripts/stash_birdclef_data.py --kaggle-download

# Train CNN baseline (smoke test)
python train.py --model cnn_baseline --epochs 1 --batch_size 8 --limit_train_batches 4

# Train main model
python train.py --model cnn_transformer --epochs 15 --warmup_epochs 5 --label_smoothing 0.1 \
    --batch_size 256 --compile

# HP search (6 trials, K=69 subset, ≤8h)
python scripts/hp_search.py --model cnn_transformer

# Final test-set evaluation
python evaluate.py --model cnn_transformer --split test
```

---

## Documentation

| File | Contents |
|---|---|
| [PROBLEMSETTING.md](PROBLEMSETTING.md) | Problem definition, architecture motivation, evaluation protocol |
| [docs/TRAINING.md](docs/TRAINING.md) | Setup, training commands, all CLI args, HP search, adding models |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Data pipeline, preprocessing constants, split strategy, API |
| [docs/CNN_BASELINE.md](docs/CNN_BASELINE.md) | ResNet-18 baseline results |
| [docs/VIT_BASELINE.md](docs/VIT_BASELINE.md) | ViT-Small baseline results |
| [docs/RF_BASELINE.md](docs/RF_BASELINE.md) | MFCC + Random Forest baseline results |
| [docs/CNN_TRANSFORMER.md](docs/CNN_TRANSFORMER.md) | Main CNN-Transformer model, HP search, results |

---

## Repository structure

```
aml2026-group-10/
├── PROBLEMSETTING.md
├── README.md
├── train.py                   # training harness (run_training callable from hp_search)
├── evaluate.py                # val + test evaluation (--split val|test)
├── train_rf.py                # Random Forest training (separate due to sklearn API)
├── models/
│   ├── cnn_transformer.py         # main proposed model (configurable CNN backbone + Transformer)
│   ├── pretrained_transformer.py  # pretrained ViT fine-tuning experiment
│   ├── vit_baseline.py            # ViT from scratch baseline
│   ├── cnn_baseline.py            # ResNet-18 / ResNet-34 baseline
│   ├── rf_baseline.py             # MFCC + Random Forest baseline
│   └── registry.py                # auto-discovery + kwargs forwarding
├── data/
│   └── preprocessing/
│       └── data_pipeline.py   # mel-spectrogram, 3-way split, DataLoaders
├── configs/
│   ├── hp_cnn_transformer.yaml
│   ├── hp_vit_baseline.yaml
│   └── hp_cnn_baseline.yaml
├── scripts/
│   ├── hp_search.py           # random HP search runner
│   └── stash_birdclef_data.py
├── utils/
│   ├── inference.py
│   ├── metrics.py
│   ├── notify.py
│   └── soundscape_eval.py
└── docs/
    ├── TRAINING.md
    ├── DATA_PIPELINE.md
    ├── CNN_TRANSFORMER.md
    ├── CNN_BASELINE.md
    ├── VIT_BASELINE.md
    └── RF_BASELINE.md
```
