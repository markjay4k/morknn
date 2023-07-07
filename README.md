# Morknn

A deep learning library written _mostly_ from scratch with the goal of using
only the Python standard library, numpy, and CUDA.

## Current Status (_incomplete_)

`matmul.so` is a shared object that can do matrix multiplication using a single
GPU

## Install

1. Install Nvidia driver
2. Install a compatible version of CUDA toolkit
3. Install CUDNN
4. clone repo and run `make` to build the shared objects

```shell
        >git clone https://github.com/markjay4k/morknn
        >cd morknn
        >make
```

5. Create a python virtual environment

```shell
        >python3 -m venv venv
        >source venv/bin/activate
 (venv) >pip install -r requirements.txt
```

## examples

```python
    import PyInit_matmul as mm
    import numpy as np


    n = 2 ** 12
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    print(f'multiplying {A.shape} by {B.shape} matrices')
    result = mm.matmulcuda(A, B)
```

## Requirements

you will need an Nvidia GPU and the CUDA toolkit installed

