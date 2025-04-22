#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define LL long long
//改进，利用缓存分块，每个块都跑完所有primer，减少缓存读取
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
    //每个进程都计算前根号n，除开之后部分平分
    low_value = start_n + id * (n - start_n) / p; // lowest value in the process
    if (low_value % 2 == 0)
    {
        low_value += 1;
    }
    high_value = start_n - 1 + (id + 1) * (n - start_n) / p; // highest value in the process
    size = (high_value - low_value) / 2 + 1;
    //每个进程都计算的前根号n，同时也是去掉偶数的索引版本
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
    //将前根号n进行处理拿到其中的素数
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
    //计算前根号n中素数个数，这项工作也分给所有进程一起计算
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
    //在L3缓存中能装多少个int
    int cache_int = cache_size / 4;

    //每个进程块有多少个int
    int B_size = cache_int / p;
    //进程负责的数需要分给多少个块
    int B_num = size / B_size;
    //剩下的零头还有多少个数
    int B_remain = size % B_size;
    //每个进程内进行一次编号，对于一个块将其所有的primer全部过一遍，这样能减少缓存的读取次数
    int B_id = 0;
    //因为是奇数索引，所以只有一半，必须得*2
    LL B_low_value = 2 * B_id * B_size + low_value;            // lowest value in the
    LL B_high_value = 2 * (B_id + 1) * B_size + low_value - 2; // because the (2*B_id + 1)*B_N + low_value) is odd
    LL B_count;
    //以一个块为单位进行处理
    marked = (char *)malloc(B_size);
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    //遍历所有块
    while (B_id < B_num)
    {
        //素数从3开始筛
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
            //注意此处是只需要在前根号n中找素数
            while (marked0[++index])
                ;
            prime = index * 2 + 3;
        //在每个块内标记素数的倍数
        } while (prime * prime <= B_high_value);
        for (i = 0; i < B_size; i++)
        {
            if (!marked[i])
                B_count++;
        }
        count += B_count;
        B_id++;
        //因为是奇数索引，所以只有一半，必须得*2，high自然也得-2，因为直接+1算的是后面一个数不包括
        B_low_value = 2 * B_id * B_size + low_value;
        B_high_value = 2 * (B_id + 1) * B_size + low_value - 2; // because the (2*B_id + 1)*B_N + low_value) is odd
    }
    //remain计算方式同理
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