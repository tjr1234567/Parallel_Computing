#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, nproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int data = 0;
    int tag = 100;
    MPI_Status status;
    // main process data=99
    if (rank == 0)
    {
        data = 99;
    }
    // bcast the value of data，如果没有这个函数的话0号进程中的data就会和其他进程中的data不一样。
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("data = %d in %d process.\n", data, rank);
    MPI_Finalize();
    return 0;
}