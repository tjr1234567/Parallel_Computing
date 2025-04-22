#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// 定义矩阵大小和分块大小
#define N 1024       // 矩阵大小（N x N）
#define TILE_SIZE 16 // 每个块的大小（TILE_SIZE x TILE_SIZE）

// 矩阵乘法的 CUDA 核函数
__global__ void matrixMul(int* A, int* B, int* C, int n)
{
    // 获取当前线程块的索引
    int bx = blockIdx.x; // 当前线程块的 x 索引
    int by = blockIdx.y; // 当前线程块的 y 索引

    // 获取当前线程在块内的索引
    int tx = threadIdx.x; // 当前线程的 x 索引
    int ty = threadIdx.y; // 当前线程的 y 索引

    // 计算当前块在矩阵 A 和 B 中的起始位置
    int aBegin = n * TILE_SIZE * by; // 矩阵 A 的起始位置
    int aEnd = aBegin + n - 1;       // 矩阵 A 的结束位置
    int aStep = TILE_SIZE;           // 矩阵 A 的步长（每次移动 TILE_SIZE 列）

    int bBegin = TILE_SIZE * bx;     // 矩阵 B 的起始位置
    int bStep = TILE_SIZE * n;       // 矩阵 B 的步长（每次移动 TILE_SIZE 行）

    // 计算当前块在矩阵 C 中的起始位置
    int cRow = TILE_SIZE * by;

    // 使用共享内存加速数据访问
    __shared__ int sA[TILE_SIZE][TILE_SIZE]; // 用于存储矩阵 A 的共享内存
    __shared__ int sB[TILE_SIZE][TILE_SIZE]; // 用于存储矩阵 B 的共享内存

    int cValue = 0; // 用于存储当前线程计算的结果

    // 遍历矩阵 A 和 B 的分块
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // 将矩阵 A 和 B 的当前块加载到共享内存
        sA[ty][tx] = A[a + n * ty + tx];
        sB[ty][tx] = B[b + n * ty + tx];

        // 同步线程，确保共享内存加载完成
        __syncthreads();

        // 计算当前块的部分结果
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            cValue += sA[ty][k] * sB[k][tx];
        }

        // 同步线程，确保所有线程完成当前块的计算
        __syncthreads();
    }

    // 将计算结果写入矩阵 C
    C[cRow + n * ty + tx] = cValue;
}

int main()
{
    // 定义主机端和设备端的指针
    int *h_A, *h_B, *h_C;  // 主机端指针
    int *d_A, *d_B, *d_C;  // 设备端指针
    int size = N * N * sizeof(int); // 矩阵所需的内存大小

    // 在主机端分配内存
    h_A = (int*)malloc(size); // 矩阵 A
    h_B = (int*)malloc(size); // 矩阵 B
    h_C = (int*)malloc(size); // 矩阵 C

    // 在设备端分配全局内存
    cudaMalloc((void**)&d_A, size); // 矩阵 A
    cudaMalloc((void**)&d_B, size); // 矩阵 B
    cudaMalloc((void**)&d_C, size); // 矩阵 C

    // 使用随机数初始化矩阵 A 和 B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = rand() % 100; // 随机生成矩阵 A 的元素
            h_B[i * N + j] = rand() % 100; // 随机生成矩阵 B 的元素
        }
    }

    // 将矩阵 A 和 B 从主机端复制到设备端
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 blockSize(TILE_SIZE, TILE_SIZE); // 每个线程块包含 TILE_SIZE x TILE_SIZE 个线程
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE); // 网格大小

    // 调用 CUDA 核函数计算矩阵乘法
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 将结果矩阵 C 从设备端复制回主机端
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印结果矩阵的前 10x10 部分
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // 释放主机端和设备端的内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}