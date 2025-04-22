#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// 定义矩阵大小和卷积核大小
#define N 1024  // 矩阵大小（N x N）
#define M 3     // 卷积核大小（M x M）

// 计算目标矩阵的卷积
__global__ void convolution(float* input, float* output, float* kernel, int n, int m)
{
    // 计算当前线程在矩阵中的位置
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的行索引
    int j = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程的列索引

    // 确保线程索引在矩阵范围内
    if (i < n && j < n)
    {
        float value = 0.0f; // 用于存储卷积结果

        // 遍历卷积核
        for (int k = 0; k < m; k++)
        {
            for (int l = 0; l < m; l++)
            {
                // 计算卷积核对应的输入矩阵位置
                int x = i + k - m / 2; // 输入矩阵的行索引
                int y = j + l - m / 2; // 输入矩阵的列索引

                // 检查输入矩阵索引是否在范围内
                if (x >= 0 && x < n && y >= 0 && y < n)
                {
                    // 累加卷积结果
                    value += input[x * n + y] * kernel[k * m + l];
                }
            }
        }

        // 将卷积结果写入输出矩阵
        output[i * n + j] = value;
    }
}

int main()
{
    // 定义主机端和设备端的指针
    float *h_input, *h_output, *h_kernel;  // 主机端指针
    float *d_input, *d_output, *d_kernel; // 设备端指针
    int size = N * N * sizeof(float);     // 矩阵所需的内存大小

    // 在主机端分配内存
    h_input = (float*)malloc(size);       // 输入矩阵
    h_output = (float*)malloc(size);      // 输出矩阵
    h_kernel = (float*)malloc(M * M * sizeof(float)); // 卷积核

    // 在设备端分配内存
    cudaMalloc((void**)&d_input, size);   // 输入矩阵
    cudaMalloc((void**)&d_output, size);  // 输出矩阵
    cudaMalloc((void**)&d_kernel, M * M * sizeof(float)); // 卷积核

    // 初始化输入矩阵和卷积核矩阵
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_input[i * N + j] = rand() % 100; // 随机生成输入矩阵元素
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            h_kernel[i * M + j] = rand() % 10; // 随机生成卷积核元素
        }
    }

    // 将输入矩阵和卷积核从主机端复制到设备端
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, M * M * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 blockSize(16, 16); // 每个线程块包含 16x16 个线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y); // 网格大小

    // 启动卷积核函数
    convolution<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, N, M);

    // 将结果从设备端复制回主机端
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // 打印输出矩阵的前 10x10 部分
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%f ", h_output[i * N + j]);
        }
        printf("\n");
    }

    // 释放主机端和设备端的内存
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}