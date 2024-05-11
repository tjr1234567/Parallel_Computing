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
        local_data[i] = rank;
    }

    // call AlltoAll function
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