#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define LL long long

int main(int argc, char *argv[])
{
    LL count;            /* Local prime count */
    double elapsed_time; /* Parallel execution time */
    LL first;            /* Index of first multiple */
    LL global_count = 0; /* Global prime count */
    LL high_value;       /* Highest value on this proc */
    LL i;
    int id;       /* Process ID number */
    LL index;     /* Index of current prime */
    LL low_value; /* Lowest value on this proc */
    char *marked; /* Portion of 2,...,'n' */
    char *marked0;
    LL n;          /* Sieving from 2, ..., 'n' */
    int p;         /* Number of processes */
    LL proc0_size; /* Size of proc 0's subarray */
    LL prime;      /* Current prime */
    LL size;       /* Elements in 'marked' */
    LL size0;
    LL low0;
    LL high0;
    LL start_n = (int)sqrt((double)atoll(argv[1])) + 1; // the start number of the processes,to remove the number in marked0
    MPI_Init(&argc, &argv);

    /* Start the timer */

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    if (argc != 2)
    {
        if (!id)
            printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    n = atoll(argv[1]);

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */

    low_value = start_n + id * (n - start_n + 1) / p; // lowest value in the process
    if (low_value % 2 == 0)
    {
        low_value += 1;
    }
    high_value = start_n - 1 + (id + 1) * (n - start_n + 1) / p; // highest value in the process
    size = (high_value - low_value) / 2 + 1;
    size0 = (start_n - 4) / 2 + 1; // the range for searching the prime

    /* Bail out if all the primes used for sieving are
       not all held by process 0 */
    // no need to detect the process0 covers all the primes needed
    /* Allocate this process's share of the array. */
    marked0 = (char *)malloc(size0);

    if (marked0 == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < size0; i++)
    {
        marked0[i] = 0;
    }
    index = 0;
    prime = 3;

    do
    {
        first = (prime * prime - 3) / 2;
        for (i = first; i < size0; i += prime)
            marked0[i] = 1;
        while (marked0[++index])
            ;
        prime = index * 2 + 3;
    } while (prime * prime < start_n);
    // each process count the prime of 1/p "marked0" array
    count = 0;
    low0 = id * size0 / p;
    high0 = (id + 1) * size0 / p - 1;
    for (i = low0; i <= high0; i++)
        if (!marked0[i])
            count++;

    int cache1_size = 65536;      // 64k
    int cache2_size = 524288;     // 512k
    int cache3_size = 33554432;   // 32768k
    int cache_size = cache3_size; //
    int cache_int = cache_size / 4;

    int B_size = cache_int / p;
    int B_num = size / B_size;
    int B_remain = size % B_size;
    int B_id = 0;
    LL B_low_value = 2 * B_id * B_size + low_value;            // lowest value in the
    LL B_high_value = 2 * (B_id + 1) * B_size + low_value - 2; // because the (2*B_id + 1)*B_N + low_value) is odd
    LL B_count;
    marked = (char *)malloc(B_size);
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    while (B_id < B_num)
    {
        index = 0;
        prime = 3;
        B_count = 0;
        for (i = 0; i < B_size; i++)
            marked[i] = 0;
        do
        {
            if (!(B_low_value % prime))
                first = 0;
            else
                first = prime - (B_low_value % prime);
            if ((B_low_value + first) % 2 != 0)
                first = first / 2;
            else
                first = (first + prime) / 2;
            for (i = first; i < B_size; i += prime)
                marked[i] = 1;

            while (marked0[++index])
                ;
            prime = index * 2 + 3;
        } while (prime * prime <= B_high_value);
        for (i = 0; i < B_size; i++)
        {
            if (!marked[i])
                B_count++;
        }
        count += B_count;
        B_id++;
        B_low_value = 2 * B_id * B_size + low_value;
        B_high_value = 2 * (B_id + 1) * B_size + low_value - 2; // because the (2*B_id + 1)*B_N + low_value) is odd
        B_size = (B_high_value - B_low_value) / 2 + 1;
    }
    if (B_remain != 0)
    {
        index = 0;
        prime = 3;
        B_count = 0;
        B_high_value = high_value;
        for (i = 0; i < B_remain; i++)
            marked[i] = 0;
        do
        {
            if (!(B_low_value % prime))
                first = 0;
            else
                first = prime - (B_low_value % prime);
            if ((B_low_value + first) % 2 != 0)
                first = first / 2;
            else
                first = (first + prime) / 2;
            for (i = first; i < B_remain; i += prime)
                marked[i] = 1;
            while (marked0[++index])
                ;
            prime = index * 2 + 3;
        } while (prime * prime <= B_high_value);
        for (i = 0; i < B_remain; i++)
        {
            if (!marked[i])
                B_count++;
        }
        count += B_count;
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    /* Stop the timer */

    elapsed_time += MPI_Wtime();

    /* Print the results */

    if (!id)
    {
        printf("There are %lld primes less than or equal to %lld\n",
               global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}