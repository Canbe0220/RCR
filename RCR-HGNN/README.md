# RCR-HGNN

This folder is the implementation of **RCR-HGNN**, built on HGNN for FJSP.

## Changes from HGNN

RCR augments policy action representations with a Balanced-Capacity Descriptor (BCD) and adds a Completion-Loss Certificate (CLC) penalty to the original reward.

| File | Change |
| --- | --- |
| `RCR.py` | Added: resource-capacity reasoning for BCD and CLC. |
| `PPO_model.py` | Modified: concatenates BCD features with eligible operation-machine pair representations. |
| `env/fjsp_env.py` | Modified: provides the scheduling-state information required by RCR and incorporates the CLC auxiliary penalty. |

The heterogeneous graph encoder and PPO optimizer are unchanged. Training and testing follow the original HGNN workflow.

## Dependencies

- Python 3.7.11; PyTorch 1.11.0 (CUDA 11.3); NumPy, pandas, and tqdm.
- Optional: OR-Tools 9.3.10497 for computing reference solutions.

## Training and Testing

Check the problem settings, dataset paths, and model parameters in `config.json`. Train an RCR-HGNN model using the original entry point:

```bash
python train.py
```

The corresponding validation instances should be placed under `data_dev/`. Evaluate a trained model using:

```bash
python test.py
```

Place the trained `.pt` model under `model/` and configure the test settings in `config.json`. Datasets and outputs follow the original layout: generated instances are stored under `data/`, test instances under `data_test/`, trained models under `results/`, and evaluation results under `save/`.

## References

The implementation of this work refers to the following excellent work:

- https://github.com/songwenas12/fjsp-drl
- https://github.com/zcaicaros/L2D
- https://github.com/yd-kwon/MatNet
- https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
