#include <stdio.h> // 包含标准输入输出库头文件，用于 printf 函数
#include <mpi.h>   // 包含 MPI 库头文件，用于并行计算

int main(int argc, char *argv[])
{
    int rank, nproc; // 定义两个变量：rank 表示当前进程的 ID，nproc 表示总进程数

    MPI_Init(&argc, &argv); // 初始化 MPI 环境，必须在其他 MPI 函数调用之前执行
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的 ID（rank），存储到 rank 中
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // 获取总进程数，存储到 nproc 中

    int data = 0; // 定义一个整型变量 data，用于存储广播的数据，初始值为 0
    MPI_Status status; // 定义一个 MPI_Status 变量，用于存储通信状态
    int status_data =0;
    int tag = 0;
    
    // 如果当前进程是主进程（rank 为 0）
    if (rank == 0)
    {
        data = 99; // 主进程将 data 设置为 99
        status_data = 1; // 主进程将 status_data 设置为 1
        //在此处不能更改tag，否则会导致双方的 tag 不一致，出现阻塞
        //tag = 100; // 定义一个整型变量 tag，表示消息的标签
        // 使用 MPI_Send 函数将 status_data 发送到其他进程
        // 参数说明：
        // &status_data：指向要发送的数据的指针
        // 1：发送的数据个数
        // MPI_INT：数据类型（整型）
        // MPI_ANY_SOURCE：接收进程的 ID（可以是任意进程）
        // tag：消息的标签
        // MPI_COMM_WORLD：通信域，表示所有进程
        MPI_Send(&status_data, 1, MPI_INT, 1, tag, MPI_COMM_WORLD); // 主进程发送消息
    }
    else if(rank == 1){
        printf("Process %d is waiting for data from process %d\n", rank, 0);
        MPI_Recv(&status_data, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); // 其他进程接收消息
        // 使用 MPI_Recv 函数接收来自主进程的数据
        // 参数说明：
        // &status_data：指向接收数据的指针
        // 1：接收的数据个数    
        // MPI_INT：数据类型（整型）
        // 0：发送进程的 ID（主进程的 rank 为 0）
        // tag：消息的标签
        // MPI_COMM_WORLD：通信域，表示所有进程
        // &status：存储通信状态的变量
        printf("Process %d received data %d from process %d with tag %d\n", rank, status_data, status.MPI_SOURCE, tag);
    }

    // 确保所有进程在广播之前完成点对点通信
    MPI_Barrier(MPI_COMM_WORLD);

    // 使用 MPI_Bcast 函数将主进程的 data 值广播到所有进程
    // 参数说明：
    // &data：指向要广播的数据的指针
    // 1：广播的数据个数
    // MPI_INT：数据类型（整型）
    // 0：广播的源进程（主进程的 rank 为 0）
    // MPI_COMM_WORLD：通信域，表示所有进程
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);



    // 每个进程打印 data 的值以及其进程 ID（rank）
    // 由于使用了 MPI_Bcast，所有进程的 data 值都应该是 99
    printf("data = %d in %d process.\n", data, rank);

    MPI_Finalize(); // 结束 MPI 环境，释放资源
    return 0; // 程序正常结束
}