# RCR-DANIEL

This folder is the implementation of **RCR-DANIEL**, built on DANIEL for FJSP.

## Changes from DANIEL

RCR augments policy action representations with a Balanced-Capacity Descriptor (BCD) and adds a Completion-Loss Certificate (CLC) penalty to the original reward.

| File | Change |
| --- | --- |
| `RCR.py` | Added: resource-capacity reasoning for BCD and CLC. |
| `model/main_model.py` | Modified: concatenates BCD features with candidate operation-machine pair representations. |
| `fjsp_env_same_op_nums.py` | Modified: provides the scheduling-state information required by RCR and incorporates the CLC auxiliary penalty. |
| `fjsp_env_various_op_nums.py` | Modified: provides the corresponding RCR support for instances with varying numbers of operations. |

The dual-attention backbone and PPO optimizer are unchanged. Training and testing follow the original DANIEL workflow.

## Dependencies

- Python 3.7.11; PyTorch 1.11.0 (CUDA 11.3); NumPy, pandas, and tqdm.
- Optional: OR-Tools 9.3.10497 for computing reference solutions.

## Training and Testing

Check the dataset paths and train an RCR-DANIEL model using the original entry point:

```bash
python train.py \
  --n_j 10 \
  --n_m 5 \
  --data_source SD1 \
  --model_suffix demo
```

Evaluate a trained model using greedy decoding:

```bash
python test_trained_model.py \
  --data_source SD1 \
  --model_source SD1 \
  --test_data 10x5 \
  --test_model 10x5 \
  --test_mode False \
  --sample_times 100
```

Set `--test_mode True` to use sampling. Datasets and outputs follow the original layout: test instances are stored under `data/BenchData/`, `data/SD1/`, and `data/SD2/`; validation instances under `data/data_train_vali/`; trained models under `trained_network/`; and evaluation results under `test_results/`.

## References

The implementation of this work refers to the following excellent work:

- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
- https://github.com/Diego999/pyGAT
