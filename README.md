# Learning with Resource-Capacity Reasoning for Flexible Job-Shop Scheduling

This repository contains the anonymous implementation of **Resource-Capacity Reasoning (RCR)**, an encoder-agnostic framework for deep reinforcement learning-based scheduling.

> **RCR explicitly evaluates whether each scheduling decision preserves sufficient future resource capacity—without replacing the representation encoder of the underlying scheduler.**

## Overview

Existing DRL schedulers primarily evaluate candidate decisions through learned state representations. RCR complements these representations with analytical resource-capacity information at two levels: candidate-action evaluation and reward-based policy learning. The same framework is integrated into three architecturally distinct backbones and supports both PPO and REINFORCE.

## Core Components

### Balanced-Capacity Descriptor (BCD)

BCD constructs a four-dimensional descriptor for each feasible operation-machine assignment. It characterizes how the candidate decision changes residual capacity pressure and redistributes bottlenecks across jobs, machines, local time windows, and resource groups. The normalized descriptor is concatenated with the original action representation before policy scoring:

```text
RCR action representation = backbone action representation || BCD
```

### Completion-Loss Certificate (CLC)

CLC evaluates the realized successor state against necessary capacity conditions for completing the remaining operations within a reference horizon. Its violation value is converted into an auxiliary penalty and added to the original makespan-oriented reward:

```text
RCR reward = original backbone reward + CLC auxiliary penalty
```

BCD and CLC are derived from the same resource-capacity foundation: BCD provides action-level guidance before a decision, while CLC provides completion-oriented feedback after the transition.

## Implementations

| Implementation | Backbone | Optimizer | Problems | Details |
| --- | --- | --- | --- | --- |
| RCR-HGNN | Heterogeneous graph neural network | PPO | FJSP | [`RCR-HGNN/`](./RCR-HGNN/) |
| RCR-DANIEL | Dual-attention network | PPO | FJSP | [`RCR-DANIEL/`](./RCR-DANIEL/) |
| RCR-RESCHED | Transformer-based scheduler | REINFORCE | FJSP and JSSP | [`RCR-ReSched/`](./RCR-ReSched/) |

Each implementation preserves the original backbone encoder and training workflow. Its subdirectory provides the modified-file summary, dependencies, dataset layout, and training and evaluation instructions.

## Reproduction

Select a backbone above and follow the corresponding README. Training and testing use the original entry points of each codebase, allowing the RCR-enhanced models to be evaluated under the same experimental workflow as their backbones.

## Acknowledgments

This repository builds on the following open-source implementations:

- https://github.com/XiangjieXiao/ReSched
- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
