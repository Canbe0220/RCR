# RCR-RESCHED

REINFORCE implementation of **Learning with Resource-Capacity Reasoning for Flexible Job-Shop Scheduling**, built on RESCHED for FJSP and JSSP.

## Changes from RESCHED

RCR augments policy action representations with a Balanced-Capacity Descriptor (BCD) and adds a Completion-Loss Certificate (CLC) penalty to the original reward.

| File | Change |
| --- | --- |
| `RCR.py` | Added: resource-capacity reasoning for BCD and CLC. |
| `SchedulingModel.py` | Modified: concatenates BCD features with candidate-action representations. |
| `REINFORCETrainer.py` | Modified: incorporates the CLC auxiliary penalty during training. |

The backbone encoder and environment are unchanged. Training and testing follow the original RESCHED workflow.

## Dependencies

- PyTorch 2.3.1 (CUDA 12.1), NumPy, pandas.
- Optional: OR-Tools 9.11.4210 for computing reference solutions.

## Training and Testing

Set the problem in `REINFORCE/SchedulingMain.py`:

```python
PROBLEM = 'fjsp'  # 'fjsp' or 'jssp'
```

In the corresponding `REINFORCE/configs/*.py` file, check dataset paths and update these `runner_params` fields:

| Parameter | Train a new model | Test a trained model |
| --- | --- | --- |
| `test_only` | `False` | `True` |
| `checkpoint` | `None` | `None` |
| `model_path` | `None` | Path to an RCR-RESCHED `.pth` file |

Run either mode using the same entry point:

```bash
cd REINFORCE
python SchedulingMain.py
```

Use `checkpoint` to resume an existing experiment. Datasets follow the original layout: `data/FJSP/TNNLS/` and `data/JSSP/L2D/`. Logs and model outputs are saved under `result/`.

## Acknowledgments

Built on RESCHED. We also acknowledge [POMO](https://github.com/yd-kwon/POMO), [L2D](https://github.com/zcaicaros/L2D), [fjsp-drl](https://github.com/songwenas12/fjsp-drl/), [FJSP-DRL](https://github.com/wrqccc/FJSP-DRL), and [MatNet](https://github.com/yd-kwon/MatNet), referenced by the original codebase.
