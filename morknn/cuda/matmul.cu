//#include <Python.h>
#include <iostream>

extern "C" {
    __global__ void matrixMultiplication(float* A, float* B, float* C, int N, int O, int M)
    {
        /*
           Matrix A is shape (NxO)
           Matrix B is shape (OxM)
           Matrix C will be shape (NxM)
        */
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (row < N && col < M)
        {
            float sum = 0.0;
            for (int i = 0; i < O; i++)
            {
                sum += A[row * O + i] * B[i * M + col];
            }
            C[row * M + col] = sum;
        }
    }
    
    void matrixMulCuda(float* h_A, float* h_B, float* h_C, int N, int O, int M)
    {
        // Pointers for GPU memory
        float *d_A, *d_B, *d_C;
    
        // Allocate GPU memory
        cudaMalloc((void**)&d_A, N * O * sizeof(float));
        cudaMalloc((void**)&d_B, O * M * sizeof(float));
        cudaMalloc((void**)&d_C, N * M * sizeof(float));
    
        // Copy input matrices from host to GPU memory
        cudaMemcpy(d_A, h_A, N * O * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, O * M * sizeof(float), cudaMemcpyHostToDevice);
    
        // Define grid and block dimensions
        dim3 blockSize(32, 32);
        dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
        // Launch kernel for matrix multiplication
        matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, O, M);
    
        // Copy the result matrix from GPU to host memory
        cudaMemcpy(h_C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    
        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

