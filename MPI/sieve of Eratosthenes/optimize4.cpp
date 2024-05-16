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
    LL high_index;       /* Highest index on this proc */
    LL i;                /*a temp var for some circulate*/
    int id;              /* Process ID number */
    LL index;            /* Index of current prime */
    LL low_index;        /* Lowest index on this proc */
    char *marked;        /* Portion of 5,7,11,..,'n' */
    char *marked0;       /* Portion of 5,7,11,..,'sqrt(n)' */
    LL n;                /* Sieving from 2, ..., 'n' */
    int p;               /* Number of processes */
    LL prime;            /* Current prime */
    LL size;             /* Elements in 'marked' */
    LL size0;            /* Elements in 'marked0' */
    // LL    low0;         //lowest index for a process to count the prime in the marked0,be removed in this optimize
    // LL    high0;        //highest index for a process to count the prime in the marked0,be removed in this optimize
    LL high_value0; /* the highest value in 'marked0' */
    LL start_n;     /* sqrt(n) */
    LL r;           /* prime/3 */
    LL q;           /* prime%3 */
    LL t;           /* thr number which B_low_value greater than the multiple just smaller than it */
    // initial the MIP environment
    MPI_Init(&argc, &argv);

    /* Start the timer */

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    // check the input
    if (argc != 2)
    {
        if (!id)
            printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    // caculate basic vars about the range of the num which the process responsible to
    n = atoll(argv[1]);
    start_n = (int)sqrt((double)n);
    size0 = start_n / 3; // the range where the process find the var 'prime'
    if (((size0 - 1) * 3 + 5 - (size0 - 1) % 2) > start_n)
        size0--;
    // LL num = n/3 - size0;
    LL num = n / 3; // the length of the array in the process

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */

    // low_index = size0 + id*(num - 1)/p;                                  to remove the 'marked0' part from the main circulation
    // high_value = size0 + (id+1) * (num - 1) / p - 1;                     seem to be feckless in shrink the running time
    low_index = id * (num - 1) / p;
    high_index = (id + 1) * (num - 1) / p - 1;
    if (id == p - 1)
    { // make sure the value the high_index linked is not out of range
        while ((3 * high_index + 5 - high_index % 2) > n)
            high_index--;
    }
    size = high_index - low_index + 1;

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
    } // initial the marked0
    if ((size0 - 1) % 2 == 0)
        high_value0 = (size0 - 1) * 3 + 5; // caculate the high_value0
    else
        high_value0 = (size0 - 1) * 3 + 4;
    index = 0;
    prime = 5;
    do
    {
        r = prime / 3;
        q = prime % 3; // q=1 or 2
        if (q == 1)
        {
            // prime + 4 + 6i = (3r + 1)*5 => i = 2r            6k+1 array
            for (i = 2 * r; prime + 4 + 6 * i <= high_value0; i += prime)
            {
                marked0[(prime + 4 + 6 * i - 5) / 3] = 1;
            }
            // prime + 6i*prime = any integer => i = 1          6k-1 array
            for (i = 1; prime + prime * i * 6 <= high_value0; i++)
            {
                marked0[((prime + prime * i * 6) - 4) / 3] = 1;
            }
        }
        else
        {
            // prime + 2 + 6i = (3r + 2)*5 => i = 2r + 1        6k-1 array
            for (i = 2 * r + 1; prime + 2 + 6 * i <= high_value0; i += prime)
            {
                marked0[(prime + 2 + 6 * i - 4) / 3] = 1;
            }
            // prime + 6i*prime = any integer => i = 1          6k+1 array
            for (i = 1; prime + prime * i * 6 <= high_value0; i++)
            {
                marked0[((prime + prime * i * 6) - 5) / 3] = 1;
            }
        }

        while (marked0[++index])
            ;
        prime = index * 3 + 5 - index % 2;
    } while (prime * prime <= start_n);
    count = 0;
    // low0 = id*size0/p;                to count primes in the 1/p of 'marked0' array
    // high0 = (id+1)*size0/p - 1;       var "high0" and "low0" represent the edge of the index
    // for(i = low0;i<=high0;i++)
    //     if(!marked0[i]) count++;

    // int cache1_size = 65536;          // size of L1 cache :64k
    // int cache2_size = 524288;         // size of L2 cache :512k
    LL cache3_size = 33554432;     // size of L3 cache :32768k
    LL cache_size = cache3_size;   // identify the cache used in this program, the L3 cache seem to have the best performance
    LL cache_int = cache_size / 8; // through the test seem that when the cache_size divide 8 can have a better performance

    LL B_size = cache_int / p;   // B_size = 524288
    LL B_num = size / B_size;    // represent the number of blocks this process
    LL B_remain = size % B_size; // represent the remain unoperated number
    LL B_id = 0;                 // record the current Block id in the process
    LL B_n = B_size / 10;        // B_n used in initial the "marked" array to elimite the multiple of
    LL B_low_index;              // lowest index in the block to cauculate the relative index in "marked"
    LL B_high_index;             // highest index in the block to cauculate the relative index in "marked"
                                 // should be guarantee not above the high_index
    LL B_low_value;              // the value which B_low_index relate to
    LL B_high_value;             // the value which B_high_index relate to
    LL s;                        // temp var used in circulation
    int k;                       // temp var used in circulation
    marked = (char *)malloc(B_size);
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    while (B_id < B_num)
    {
        index = 1;
        prime = 7;
        // caculate the low and high index and their value
        B_low_index = low_index + B_id * B_size;
        B_low_value = 3 * B_low_index + 5 - B_low_index % 2;
        if (B_low_value > (3 * high_index + 5 - high_index % 2))
            break;
        B_high_index = low_index + (B_id + 1) * B_size - 1;
        B_high_value = 3 * B_high_index + 5 - B_high_index % 2;
        if (B_high_value > 3 * high_index + 5 - high_index % 2)
        {
            B_high_value = 3 * high_index + 5 - high_index % 2;
        }
        // optimize the initial circulation "for (i=0; i<B_size; i++) marked[i] = 1;"
        switch (B_low_index % 10)
        {
        case (0):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 0;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 0;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (1):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 0;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 0;
            }
            break;
        }
        case (2):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 0;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 0;
                marked[s + 9] = 1;
            }
            break;
        }
        case (3):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 0;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 0;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (4):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 0;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 0;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (5):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 0;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 0;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (6):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 0;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 0;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (7):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 0;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 0;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (8):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 0;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 0;
            }
            break;
        }
        case (9):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 0;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 0;
                marked[s + 9] = 1;
            }
            break;
        }
        }
        do
        {
            // find the first multiple of the prime in this block
            // mark all multiples of the prime in the "6k-1" array and "6k+1" array, respectively.
            // the detail will be described in the report
            r = prime / 3;
            q = prime % 3;
            if (q == 1)
            {
                first = prime + 4 + 12 * r;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 5)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - first) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j - 5)) / 3 - B_low_index] = 0;
                }
                first = prime + 6 * prime;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 4)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - first) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j) - 4) / 3 - B_low_index] = 0;
                }
            }
            else
            {
                first = prime + 2 + 6 * (2 * r + 1);
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 4)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - first) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j - 4)) / 3 - B_low_index] = 0;
                }
                first = prime + 6 * prime;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 5)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - first) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j) - 5) / 3 - B_low_index] = 0;
                }
            }
            // search the next prime
            while (++index < size0 && marked0[index])
                ;
            prime = index * 3 + 5 - index % 2;
        } while (prime * prime <= B_high_value);
        // count the prime found in this block
        for (i = 0; i < B_size; i++)
        {
            count += marked[i];
        }
        // switch to the next block
        B_id++;
    }
    // similar to the main circulation
    // seperate the remain part just to avoid some unnecessary branching statements
    if (B_remain)
    {
        index = 1;
        prime = 7;
        B_low_index = low_index + B_id * B_size;
        B_low_value = 3 * (B_low_index) + 5 - (B_low_index) % 2;
        B_high_index = high_index;
        B_high_value = 3 * high_index + 5 - high_index % 2;
        B_n = B_remain / 10;
        switch (B_low_index % 10)
        {
        case (0):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 0;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 0;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (1):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 0;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 0;
            }
            break;
        }
        case (2):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 0;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 0;
                marked[s + 9] = 1;
            }
            break;
        }
        case (3):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 0;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 0;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (4):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 0;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 0;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (5):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 0;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 0;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (6):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 0;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 0;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (7):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 0;
                marked[s + 1] = 1;
                marked[s + 2] = 1;
                marked[s + 3] = 0;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 1;
            }
            break;
        }
        case (8):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 1;
                marked[s + 2] = 0;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 1;
                marked[s + 9] = 0;
            }
            break;
        }
        case (9):
        {
            for (i = 0; i <= B_n; i++)
            {
                s = i * 10;
                marked[s] = 1;
                marked[s + 1] = 0;
                marked[s + 2] = 1;
                marked[s + 3] = 1;
                marked[s + 4] = 1;
                marked[s + 5] = 1;
                marked[s + 6] = 1;
                marked[s + 7] = 1;
                marked[s + 8] = 0;
                marked[s + 9] = 1;
            }
            break;
        }
        }
        do
        {
            r = prime / 3;
            q = prime % 3;
            if (q == 1)
            {
                first = prime + 4 + 12 * r;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 5)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - prime - 4 - 12 * r) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j - 5)) / 3 - B_low_index] = 0;
                }
                first = prime + 6 * prime;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 4)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - prime - 6 * prime) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j) - 4) / 3 - B_low_index] = 0;
                }
            }
            else
            {
                first = prime + 2 + 6 * (2 * r + 1);
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 4)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - prime - 2 - 6 * (2 * r + 1)) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j - 4)) / 3 - B_low_index] = 0;
                }
                first = prime + 6 * prime;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 5)) / 3 - B_low_index] = 0;
                    t = 6 * prime - (B_low_value - prime - 6 * prime) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j) - 5) / 3 - B_low_index] = 0;
                }
            }
            while (++index < start_n && marked0[index])
                ;
            prime = index * 3 + 5 - index % 2;
        } while (prime * prime <= B_high_value);
        for (i = 0; i < B_remain; i++)
        {
            count += marked[i];
        }
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Stop the timer */

    elapsed_time += MPI_Wtime();

    /* Print the results */

    if (!id)
    {
        printf("There are %lld primes less than or equal to %lld\n",
               global_count + 3, n);
        // the number 3 means the prime 2,3 and 5 haven't been statistic in the forward procedure
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}