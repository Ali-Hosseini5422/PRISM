# PRISM

PyTorch implementation of the PRISM model for mobile crowdsensing.

The code includes the main components (selective GSSM, allocation-aware MoE, residual cross-attention, anticipation module, and a simple TTA adapter) plus training scripts that work with both synthetic data and real mobility traces when available.

## Files

```
prism_model.py     # main model
train_eval.py      # basic synthetic training loop
train_real.py      # training / eval with dataset loaders
datasets/          # loaders for synthetic, Roma Taxi, T-Drive
data/              # put real trace files here
requirements.txt
```

## Data

By default everything runs on synthetic non-stationary traces (with controlled drift).

If you have the real datasets, drop them in the expected places and the loaders will pick them up:

- Roma Taxi: `data/taxi_february.txt` (or .csv)
- T-Drive: files under `data/tdrive/`

Links:
- Roma: https://ieee-dataport.org/open-access/crawdad-romataxi
- T-Drive: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

If the files are missing it just falls back to synthetic data with similar spatial statistics. CrowdSenSim traces can also be used if you generate them yourself.

## Running

```bash
pip install -r requirements.txt
python train_real.py
```

`train_real.py` trains for a few epochs on synthetic data and then tries to evaluate on Roma / T-Drive if the files are present. You can change the number of epochs, steps, and dataset name inside the script.

## Notes

- The selective scan is currently a pure PyTorch loop (fine for the scales we used in experiments, but not the fastest possible).
- Some parts (especially the diffusion-style TTA adapter and a few projections) are simplified compared to the full description in the paper. They were sufficient for the results we report.
- Default loss weights match the ones used in the experiments.
- Tested with PyTorch 2.1+. GPU is optional but recommended if you increase the number of workers/tasks.

If something breaks or you want the exact hyperparameters from a specific figure, open an issue or check the training scripts — most of the knobs are near the top of `train_real.py`.
