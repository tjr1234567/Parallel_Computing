#include <iostream>
#include <mpi.h>

#define matrix_row 3
#define vector_size 3

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // split the data by row
    int rows_per_process = matrix_row / size;
    int leftover_rows = matrix_row % size;

    int *matrix = nullptr;
    int *vector = nullptr;
    int *result = nullptr;

    // process 0 initial and send matrix and vector
    if (rank == 0)
    {
        // location of matrix , vector and result
        matrix = new int[matrix_row * matrix_row];
        vector = new int[vector_size];
        result = new int[matrix_row];

        // initial matrix
        for (int i = 0; i < matrix_row; ++i)
        {
            for (int j = 0; j < matrix_row; ++j)
            {
                matrix[i * matrix_row + j] = i + j;
            }
        }
        // initial vector
        for (int i = 0; i < vector_size; ++i)
        {
            vector[i] = i;
        }
    }

    // send vector to all processes
    MPI_Bcast(vector, vector_size, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate the start and end of every process
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    if (rank == size - 1)
    {
        end_row += leftover_rows;
    }

    int *local_matrix = new int[rows_per_process * matrix_row];
    int *local_result = new int[rows_per_process];

    // scatter the matrix to local matrix
    MPI_Scatter(matrix, rows_per_process * matrix_row, MPI_INT, local_matrix,
                rows_per_process * matrix_row, MPI_INT, 0, MPI_COMM_WORLD);

    // execute the mutiply in every process
    for (int i = 0; i < rows_per_process; ++i)
    {
        local_result[i] = 0;
        for (int j = 0; j < matrix_row; ++j)
        {
            local_result[i] += local_matrix[i * matrix_row + j] * vector[j];
        }
    }

    // gather the local result
    MPI_Gather(local_result, rows_per_process, MPI_INT, result, rows_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    // output the result
    if (rank == 0)
    {
        std::cout << "Matrix-Vector Multiplication Result:" << std::endl;
        for (int i = 0; i < matrix_row; ++i)
        {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    }

    // Finalize
    delete[] matrix;
    delete[] vector;
    delete[] result;
    delete[] local_matrix;
    delete[] local_result;

    MPI_Finalize();

    return 0;
}