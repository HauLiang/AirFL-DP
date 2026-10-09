# AirFL-DP

Python code for the paper "Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station".

If you use the code, please cite our paper:
> [[1] Liang, Hao and Wen, Haifeng and Wu, Kaishun and Letaief, Khaled B. and Xing, Hong, "Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station", *IEEE Journal on Selected Areas in Communications (JSAC)*, 2026.](https://arxiv.org/abs/2510.23463 "https://arxiv.org/abs/2510.23463")



## A Fast Reproduction Guide

If you just want to reproduce the figures in our paper, navigate to the `Plot_figures` folder and run the corresponding plotting script (e.g., `plot_figure_x.py`). This will generate the x-th figure demonstrated in the paper.



## Preparation

- Environment:

  - Python==3.11.13, numpy==1.26.4, matplotlib==3.10.3
  - torch==2.0.1,  torchvision==0.15.2, cvxpy==1.6.6

- Dataset setup:

  From the repository root, navigate to the `dataset` directory:

  ```shell
  cd ./dataset
  ```

  - i. i. d. dataset

    ```shell
    python generate_FashionMNIST.py iid - - # for iid and balanced scenario
    ```

  - non-i.i.d. dataset

    ```shell
    python generate_FashionMNIST.py noniid - pat # for pathological noniid and unbalanced scenario
    ```

  The generated dataset files will be saved in the `dataset/FashionMNIST` folder.

  

## Running Experiments

After preparing the dataset, run experiments from the `system` directory:

```shell
cd ./system
python main.py -data FashionMNIST -algo AirFL_DP -did 0 -jr 0.9 -epsilon 0.1 -pow 2e-11
```

The command above runs the `AirFL_DP` method on Fashion-MNIST. To evaluate other supported methods or configurations, modify the corresponding command-line arguments.

  

## Main Arguments

| Argument   | Description                | Example        |
| ---------- | -------------------------- | -------------- |
| `-data`    | Dataset name               | `FashionMNIST` |
| `-algo`    | Algorithm to run           | `AirFL_DP`     |
| `-did`     | Device ID                  | `0`            |
| `-jr`      | Client participation ratio | `0.9`          |
| `-epsilon` | Privacy budget parameter   | `0.1`          |
| `-pow`     | Transmit power parameter   | `2e-11`        |

Supported algorithm names in the experimental code include `FedAvg`, `FedAvg_Clip`, `AirFL_ZF`, `AirFL_DP`, and `AirFL_DP_online`. See the code for algorithm-specific settings.



## Experimental Outputs

After an experiment finishes, results are saved in the `results` and `result_for_monte` directories. These outputs contain information used for performance evaluation and Monte Carlo result aggregation.




##

This code is based on the code available from
https://github.com/TsingZ0/PFLlib from the following paper:

> [[2] Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian. "PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark". *Journal of Machine Learning Research*, 2025.](https://www.jmlr.org/papers/v26/23-1634.html "https://www.jmlr.org/papers/v26/23-1634.html")

Please check the accompanying license and the license of [2] before using. 


@ All rights are reserved by the authors.
