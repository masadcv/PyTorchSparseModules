# Simple Sparse Convolutions
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
<!-- [![PyPI version](https://badge.fury.io/py/torchhaarfeatures.svg)](https://badge.fury.io/py/torchhaarfeatures) -->
<!-- <img src="https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8%20|%203.9-3776ab.svg"/> -->
<!-- <img src="https://img.shields.io/badge/PyTorch-%3E%3D%201.6-brightgreen.svg"/> -->

This repository implements simplest forms of sparse convolutions in PyTorch. 

Within the repository, implementation is provided for the following:

## Installation
This package can be installed as: 

`pip install torchsparsemodules`

or 

`pip install git+https://github.com/masadcv/PyTorchSparseModules`

## Examples
Usage examples are provided in example python files within the repository.

A simple example (`example.py`) usage following a PyTorch usage format:

```
import torchsparsemodules
import torch

sparseconv3d = torchsimplesparseconv.Conv3d(kernel_size=(9, 9, 9), stride=1)
output_sparse_conv3d = sparseconv3d(torch.rand(size=(1, 1, 128, 128, 128)))

print(output_sparse_conv3d.shape)

sparseconv3d = torchsimplesparseconv.Conv2d(kernel_size=(9, 9), stride=1)
output_sparse_conv2d = sparseconv3d(torch.rand(size=(1, 1, 128, 128)))
print(output_sparse_conv2d.shape)
```

## Citation
If you use our code, please consider citing our paper:

```
```
