# Efficient Nonparametric Tensor Decomposition for Binary and Count Data

Implementation for the paper [[arXiv]](http://arxiv.org/abs/2401.07711)

**Efficient Nonparametric Tensor Decomposition for Binary and Count Data (AAAI-24)**

by Zerui Tao, Toshihisa Tanaka and Qibin Zhao

**TL;DR:** A Gaussian process tensor completion algorithm for binary and count data.

## Usage

The code is based on `PyTorch`.

Core required packages are listed in `./requirements.txt`.

To conduct the experiments, run
```
CUDA_VISIBLE_DEVICES='0' python main.py --dataset $DATASET 
```

Configuration files are stored in `./config`.

## Citation

```
@inproceedings{tao2024efficient,
	author={Tao, Zerui and Tanaka, Toshihisa and Zhao, Qibin},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	title={Efficient Nonparametric Tensor Decomposition for Binary and Count Data},
	year={2024},
}
```