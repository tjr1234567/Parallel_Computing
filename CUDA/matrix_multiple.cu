#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024  // size of matrix
#define TILE_SIZE 16  // size of tile

// kernel function of matrix multiply
__global__ void matrixMul(int* A, int* B, int* C, int n)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * TILE_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = TILE_SIZE;

    int bBegin = TILE_SIZE * bx;
    int bStep = TILE_SIZE * n;

    int cRow = TILE_SIZE * by;
    // utilize the share memory to accelerate the access of data
    __shared__ int sA[TILE_SIZE][TILE_SIZE];
    __shared__ int sB[TILE_SIZE][TILE_SIZE];

    int cValue = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        sA[ty][tx] = A[a + n * ty + tx];
        sB[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            cValue += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    C[cRow + n * ty + tx] = cValue;
}

int main()
{
    int *h_A, *h_B, *h_C;  // pointer on host
    int *d_A, *d_B, *d_C;  // pointer on device
    int size = N * N * sizeof(int);

    // allocate memory on CPU
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // allocate the global memory for A , B and C on GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // using rand() to initialize the A and B matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = rand() % 100;
            h_B[i * N + j] = rand() % 100;
        }
    }

    // copy the A and B matrix from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // define the size of grid and blocks 
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    // CALL the CUDA kernel function to calculate the multiple of matrix
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // copy the result from GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print part of the result matrix
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // free the memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}