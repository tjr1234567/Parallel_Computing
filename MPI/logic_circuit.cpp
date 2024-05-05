#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n, i) ((n & (1 << i)) ? 1 : 0)
int count = 0;    // local sum
int global_count; // global sum
double elapsed_time;
int logic_circuit(int id, int z);

int main(int argc, char *argv[])
{
    int i;
    int id;
    int p;
    int n;

    if (argc != 2)
    {
        printf("input error");
        return 1;
    }
    else
    {
        n = 1 << atoi(argv[1]);
    }
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (i = id; i < n; i += p)
    {
        count += logic_circuit(id, i);
    }
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    if (!id)
        printf("There are %d different solutions\nThe program has run %f seconds\nTimer resolution is %f\n",
               global_count, elapsed_time, MPI_Wtick());
    printf("Process %d is done\n", id);
    fflush(stdout); // force to output the content in buffer
    MPI_Finalize();
    return 0;
}
int logic_circuit(int id, int z)
{
    int v[16]; // Each element is a bit of z
    int i;
    for (i = 0; i < 16; i++)
        v[i] = EXTRACT_BIT(z, i);
    // target logical function
    if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3]) && (!v[3] || !v[4]) && (v[4] || !v[5]) && (v[5] || !v[6]) && (v[5] || v[6]) && (v[6] || !v[15]) && (v[7] || !v[8]) && (!v[7] || !v[13]) && (v[8] || v[9]) && (v[8] || !v[9]) && (!v[9] || !v[10]) && (v[9] || v[11]) && (v[10] || v[11]) && (v[12] || v[13]) && (v[13] || !v[14]) && (v[14] || v[15]))
    {
        printf("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
               v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
               v[10], v[11], v[12], v[13], v[14], v[15]);

        fflush(stdout);
        return 1;
    }
    return 0;
}