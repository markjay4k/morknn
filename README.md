# Morknn

A deep learning library written _mostly_ from scratch.

## Current Status

`matmul.so` is a shared object that can do matrix multiplication using a single
GPU

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

