<div style="text-align: left;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-brown" alt="rsl-rl v3.3.0" style="margin-bottom: 1px;">
</div>

## Overview

morphosymm-rl is a reinforcement learning library for IsaacLab that extends the Proximal Policy Optimization (PPO) implementation of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) with [Morphological Symmetries](https://arxiv.org/abs/2402.15552). 

Features:

- data-augmentation
- explicit equivariant neural networks via [symmetric_learning](https://github.com/Danfoa/symmetric_learning)


## Installation

Install this package with:

```bash
pip install -e .
```

## How to use

See [here](https://github.com/iit-DLSLab/morphosymm-rl/blob/main/README_how_to.md).

## Citing this work

If you find the work useful, please consider citing one of our works:

#### [Leveraging Symmetry in RL-based Legged Locomotion Control (IROS-2024)](https://arxiv.org/pdf/2403.17320)

```
@inproceedings{suhuang2024leveraging,
  author={Su, Zhi and Huang, Xiaoyu and Ordoñez-Apraez, Daniel and Li, Yunfei and Li, Zhongyu and Liao, Qiayuan and Turrisi, Giulio and Pontil, Massimiliano and Semini, Claudio and Wu, Yi and Sreenath, Koushil},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Leveraging Symmetry in RL-based Legged Locomotion Control}, 
  year={2024},
  pages={6899-6906},
  doi={10.1109/IROS58592.2024.10802439}
}
```

#### [Morphological symmetries in robotics (IJRR-2025)](https://arxiv.org/pdf/2402.15552):

```
@article{ordonez2025morphosymm,
  author = {Daniel Ordoñez-Apraez and Giulio Turrisi and Vladimir Kostic and Mario Martin and Antonio Agudo and Francesc Moreno-Noguer and Massimiliano Pontil and Claudio Semini and Carlos Mastalli},
  title ={Morphological symmetries in robotics},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  volume = {44},
  number = {10-11},
  pages = {1743-1766},
  doi = {10.1177/02783649241282422}
}
```

## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi) and [Daniel Ordonez](https://github.com/Danfoa).
