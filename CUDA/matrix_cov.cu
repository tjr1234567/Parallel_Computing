#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024  // matrix size
#define M 3     // kernel size

// calculate the convolution of the target matrix
__global__ void convolution(float* input, float* output, float* kernel, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        float value = 0.0f;

        for (int k = 0; k < m; k++)
        {
            for (int l = 0; l < m; l++)
            {
                int x = i + k - m / 2;
                int y = j + l - m / 2;

                if (x >= 0 && x < n && y >= 0 && y < n)
                {
                    value += input[x * n + y] * kernel[k * m + l];
                }
            }
        }

        output[i * n + j] = value;
    }
}

int main()
{
    float *h_input, *h_output, *h_kernel;  // host pointer
    float *d_input, *d_output, *d_kernel;  // device pointer
    int size = N * N * sizeof(float);

    // allocate memory on host
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_kernel = (float*)malloc(M * M * sizeof(float));

    // allocate memory on device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_kernel, M * M * sizeof(float));

    // initial the target matrix and kernel matrix by rand()
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_input[i * N + j] = rand() % 100;
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            h_kernel[i * M + j] = rand() % 10;
        }
    }

    // copy the matrix and covolution kernel to GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, M * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    convolution<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, N, M);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%f ", h_output[i * N + j]);
        }
        printf("\n");
    }

    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}