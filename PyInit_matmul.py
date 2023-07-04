import ctypes
import numpy as np


def matmulcuda(A, B):
    matmul = ctypes.CDLL('./matmul.so')
    matmul.matrixMulCuda.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    matmul.matrixMulCuda.restype = None
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape

    if a_cols == b_rows:
        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
            if A.dtype == np.float32 and B.dtype == np.float32:
                C = np.zeros((a_rows, b_rows), dtype=np.float32)
                matmul.matrixMulCuda(A, B, C, a_rows, a_cols, b_cols)
                return C
            else: 
                raise TypeError(f"""
                    wrong dtype for {A} or {B}.\n
                    dtype should be np.float32
                """)
        else:
            raise TypeError(f"""
                {A} and {B} should by np.ndarrays
            """)
    else:
        raise TypeError(f"""
            wrong shape for {A} and {B}\n
        """)


if __name__ == '__main__':
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]).astype(np.float32)

    B = np.array([
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ]).astype(np.float32)
    
    print(f'multiplying {A.shape} by {B.shape} matrices')
    result = matmulcuda(A, B)
    print(result)

    n = 1024
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    print(f'multiplying {A.shape} by {B.shape} matrices')
    result = matmulcuda(A, B)
    print(result)
