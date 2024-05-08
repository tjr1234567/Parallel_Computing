#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // partial matrix of each process
    int local_rows = rank + 1; // row number of local matrix
    int local_cols = 3;        // column number of local matrix
    int *local_matrix = (int *)malloc(local_rows * local_cols * sizeof(int));
    // fill the local matrix
    for (int i = 0; i < local_rows; ++i)
    {
        for (int j = 0; j < local_cols; ++j)
        {
            local_matrix[i * local_cols + j] = (rank + 1) * 10 + i * local_cols + j;
        }
    }

    // calculate the counts and displs in every process
    int *recv_counts = (int *)malloc(size * sizeof(int)); // element number each process received
    int *displs = (int *)malloc(size * sizeof(int));      // displs in every process
    for (int i = 0; i < size; ++i)
    {
        recv_counts[i] = (i + 1) * local_cols;
        displs[i] = (i > 0) ? (displs[i - 1] + recv_counts[i - 1]) : 0;
    }

    // caculate the total number of the matrix
    int total_rows = (size * (size + 1)) / 2; // total row number of the whole matrix
    int *recv_buffer = NULL;
    if (rank == 0)
    {
        recv_buffer = (int *)malloc(total_rows * local_cols * sizeof(int)); // buffer to save the receive data
    }

    // gather the local matrix
    MPI_Gatherv(local_matrix, local_rows * local_cols, MPI_INT,
                recv_buffer, recv_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // process 0 output the result
    if (rank == 0)
    {
        printf("Received Matrix:\n");
        for (int i = 0; i < total_rows; ++i)
        {
            for (int j = 0; j < local_cols; ++j)
            {
                printf("%d ", recv_buffer[i * local_cols + j]);
            }
            printf("\n");
        }
    }

    // free the resource
    free(local_matrix);
    free(recv_counts);
    free(displs);
    if (rank == 0)
    {
        free(recv_buffer);
    }

    MPI_Finalize();
    return 0;
}