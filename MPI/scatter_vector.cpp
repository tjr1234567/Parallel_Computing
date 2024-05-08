#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *send_buffer = NULL;
    int send_count = 0;
    int *recv_buffer = (int *)malloc(sizeof(int));

    if (rank == 0)
    {
        // process create a whole array
        int array_size = size * 2; // size is the double of the size of process
        send_buffer = (int *)malloc(array_size * sizeof(int));

        // filling the array
        for (int i = 0; i < array_size; ++i)
        {
            send_buffer[i] = i;
        }

        // calculate the number and displs of each process
        int *send_counts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; ++i)
        {
            send_counts[i] = 2; // because the array's size is double of the process
            displs[i] = i * 2;
        }

        // scatter the array to other process
        MPI_Scatterv(send_buffer, send_counts, displs, MPI_INT,
                     recv_buffer, 2, MPI_INT,
                     0, MPI_COMM_WORLD);

        free(send_buffer);
        free(send_counts);
        free(displs);
    }
    else
    {
        // other process receive element from process 0
        MPI_Scatterv(send_buffer, NULL, NULL, MPI_INT,
                     recv_buffer, 2, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    // every process print the element it received
    printf("Rank %d received: %d\n", rank, recv_buffer[0]);

    free(recv_buffer);

    MPI_Finalize();
    return 0;
}