#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *local_data = new int[size];
    int *alltoall_data = new int[size];

    // initial local array
    for (int i = 0; i < size; ++i)
    {
        local_data[i] = rank * 10 + i;
    }
    // local_data[0] = rank * 10 + 1;

    // call AlltoAll function
    //解释一下各个参数含义
    // local_data: 发送数据的起始地址
    // 1: 每个进程发送的元素个数
    // MPI_INT: 发送数据的类型
    // alltoall_data: 接收数据的起始地址
    // 1: 每个进程接收的元素个数
    // MPI_INT: 接收数据的类型
    // MPI_COMM_WORLD: 通信域
    // 这里的意思是每个进程发送1个int类型的数据，接收1个int类型的数据
    MPI_Alltoall(local_data, 1, MPI_INT, alltoall_data, 1, MPI_INT, MPI_COMM_WORLD);

    // output the result
    for (int i = 0; i < size; ++i)
    {
        std::cout << "Process " << rank << ": Received " << alltoall_data[i] << " from Process " << i << std::endl;
    }

    delete[] local_data;
    delete[] alltoall_data;

    MPI_Finalize();

    return 0;
}