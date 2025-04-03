#include <mpi.h> // 包含 MPI 库头文件，用于并行计算
#include <stdio.h> // 包含标准输入输出库头文件
#include <stdlib.h> // 包含标准库头文件，用于 atoi 等函数

/* 宏定义：返回 'n' 的第 'i' 位是否为 1。如果是 1，返回 1；否则返回 0 */
#define EXTRACT_BIT(n, i) ((n & (1 << i)) ? 1 : 0)

int count = 0;    // 局部计数器，用于统计当前进程找到的解的数量
int global_count; // 全局计数器，用于统计所有进程找到的解的总数
double elapsed_time; // 记录程序运行的时间
int logic_circuit(int id, int z); // 函数声明，用于检查逻辑电路是否满足条件

int main(int argc, char *argv[])
{
    int i;  // 循环变量
    int id; // 当前进程的 ID（rank）
    int p;  // 总进程数
    int n;  // 需要检查的总组合数

    // 检查命令行参数是否正确
    if (argc != 2)
    {
        printf("input error"); // 如果参数数量不正确，打印错误信息
        return 1; // 返回错误代码
    }
    else
    {
        n = 1 << atoi(argv[1]); // 将命令行参数转换为整数，并计算 2 的该整数次幂
    }

    MPI_Init(&argc, &argv); // 初始化 MPI 环境
    MPI_Barrier(MPI_COMM_WORLD); // 同步所有进程，确保计时准确
    elapsed_time += MPI_Wtime(); // 开始计时

    MPI_Comm_size(MPI_COMM_WORLD, &p); // 获取总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // 获取当前进程的 ID（rank）

    // 每个进程处理一部分组合
    for (i = id; i < n; i += p)
    {
        count += logic_circuit(id, i); // 检查逻辑电路是否满足条件，并累加满足条件的解
    }

    // 将所有进程的局部计数器汇总到全局计数器
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime(); // 结束计时

    // 如果是主进程（ID 为 0），打印结果
    if (!id)
        printf("There are %d different solutions\nThe program has run %f seconds\nTimer resolution is %f\n",
               global_count, elapsed_time, MPI_Wtick());

    // 每个进程打印完成信息
    printf("Process %d is done\n", id);
    fflush(stdout); // 强制刷新输出缓冲区

    MPI_Finalize(); // 结束 MPI 环境
    return 0; // 程序正常结束
}

int logic_circuit(int id, int z)
{
    int v[16]; // 用于存储 z 的每一位（共 16 位）
    int i; // 循环变量

    // 提取 z 的每一位，并存储到数组 v 中
    for (i = 0; i < 16; i++)
        v[i] = EXTRACT_BIT(z, i);

    // 检查逻辑电路是否满足目标逻辑函数
    if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3]) && (!v[3] || !v[4]) && 
        (v[4] || !v[5]) && (v[5] || !v[6]) && (v[5] || v[6]) && (v[6] || !v[15]) && 
        (v[7] || !v[8]) && (!v[7] || !v[13]) && (v[8] || v[9]) && (v[8] || !v[9]) && 
        (!v[9] || !v[10]) && (v[9] || v[11]) && (v[10] || v[11]) && (v[12] || v[13]) && 
        (v[13] || !v[14]) && (v[14] || v[15]))
    {
        // 如果满足条件，打印当前进程 ID 和满足条件的位组合
        printf("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
               v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
               v[10], v[11], v[12], v[13], v[14], v[15]);

        fflush(stdout); // 强制刷新输出缓冲区
        return 1; // 返回 1，表示找到一个满足条件的解
    }

    return 0; // 返回 0，表示未找到满足条件的解
}