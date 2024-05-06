#include <mpi.h>
#include <stdio.h>
#include <cstdlib>

#define BLOCK_OWNER(index, p, n) ((p * (index + 1) - 1) / n)
#define BLOCK_LOW(id, p, n) (id * n / p)
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW(id + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW(id + 1, p, n) - BLOCK_LOW(id, p, n))
#define MIN(a, b) (a < b ? a : b)

void print(int *array, int n);
void printrow(int *array, int low, int high, int n);
int main(int argc, char *argv[])
{
    int n; // size of the matrix
    int i, j, k;
    int id;
    int p;
    int min, max;
    double time, max_time;
    int *a;
    int offset; // local index of broadcast row
    int root;   // process controlling row to be becast
    int *temp;  // holds the broadcast rows

    if (argc != 4)
    {
        printf("Please input two parameter :the second is the size of the matrix");
    }
    n = atoi(argv[1]);
    min = atoi(argv[2]);
    max = atoi(argv[3]);
    a = (int *)malloc(n * n * sizeof(int));
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                a[i * n + j] = 0;
            }
            else
            {
                a[i * n + j] = (rand() % (max - min + 1)) + min;
            }
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!id)
    {
        printf("the origin matrix is:\n");
        print(a, n);
    }
    time = -MPI_Wtime();
    temp = (int *)malloc(n * sizeof(int));
    for (k = 0; k < n; k++)
    {
        root = BLOCK_OWNER(k, p, n);
        if (root == id)
        {
            offset = k - BLOCK_LOW(id, p, n);
            for (j = 0; j < n; j++)
            {
                temp[j] = a[offset * n + j];
            }
        }
        MPI_Bcast(temp, n, MPI_INT, root, MPI_COMM_WORLD);
        for (i = 0; i < BLOCK_SIZE(id, p, n); i++)
        {
            for (j = 0; j < n; j++)
            {
                a[i * n + j] = MIN(a[i * n + j], a[i * n + k] + temp[j]);
            }
        }
    }
    free(temp);
    time += MPI_Wtime();
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!id)
    {
        printf("Floyd\nmatrix size: %d\n%dprocess:%6.2f seconds\n", n, p, max_time);
    }
    printrow(a, BLOCK_LOW(id, p, n), BLOCK_HIGH(id, p, n), n);
    free(a);
    MPI_Finalize();
    return 0;
}
void print(int *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d,", array[i * n + j]);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
}
void printrow(int *array, int low, int high, int n)
{
    for (int i = low; i <= high; i++)
    {
        printf("row%d:  ", i);
        for (int j = 0; j < n; j++)
        {
            printf("%d,", array[i * n + j]);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
}
