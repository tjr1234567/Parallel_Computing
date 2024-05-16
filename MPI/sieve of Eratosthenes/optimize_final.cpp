#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define LL long long

#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")

int main(int argc, char *argv[])
{
    LL count;            /* Local prime count */
    double elapsed_time; /* Parallel execution time */
    int first;           /* Index of first multiple */
    LL global_count = 0; /* Global prime count */
    LL high_value;       /* Highest value on this proc */
    LL i;
    int id;       /* Process ID number */
    LL low_value; /* Lowest value on this proc */
    char *marked;
    LL n;          /* Sieving from 2, ..., 'n' */
    int p;         /* Number of processes */
    LL proc0_size; /* Size of proc 0's subarray */
    LL prime;      /* Current prime */
    LL size1;      /* Elements in 'marked' */
    LL size2;
    LL *primes;
    int pcount;
    char *st;

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

    /* Bail out if all the primes used for sieving are
       not all held by process 0 */

    proc0_size = (1 + (n - 2) / p) >> 1;

    if ((3 + proc0_size) < (int)sqrt((double)n))
    {
        if (!id)
            printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    int sq_n = (int)sqrt((double)n);
    primes = (LL *)malloc(6500);
    st = (char *)malloc(sq_n + 2);
    for (int j = 3; j <= sq_n; j += 2)
        st[j] = 0;

    if (primes == NULL || st == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    pcount = 1;
    // 埃式筛预处理sq_n以内的质数,计数位pcount,质数存到primes数组中
    for (int j = 3; j <= sq_n; j += 2)
    {
        if (st[j])
            continue;
        else
            primes[pcount++] = j;
        if (j > sq_n / j)
            continue;
        int two_j = j << 1;
        for (int t = j * j; t <= sq_n; t += two_j)
        {
            st[t] = 1;
        }
    }

    count = 0;

    low_value = 7 + id * (n - 6) / p;
    high_value = 6 + (id + 1) * (n - 6) / p;
    LL mod6 = low_value % 6;
    LL low_value_plus1 = mod6 < 2 ? low_value - mod6 + 1 : low_value - mod6 + 7; // 6n+1序列的第一个数
    LL low_value_plus5 = low_value - mod6 + 5;                                   // 6n+5序列的第一个数
    mod6 = high_value % 6;
    LL high_value_plus1 = mod6 == 0 ? high_value - 6 + 1 : high_value - mod6 + 1; // 6n+1序列的最后一个数
    LL high_value_plus5 = mod6 == 5 ? high_value : high_value - mod6 - 1;         // 6n+5序列的最后一个数

    int size_plus1 = (high_value_plus1 - low_value_plus1) / 6 + 1; // 当前进程处理的6n+1序列的大小
    int size_plus5 = (high_value_plus5 - low_value_plus5) / 6 + 1; // 当前进程处理的6n+5序列的大小

    int block_size = 120000 * 8 / p;             // 数字120000*8多次实验最优
    LL block_num1 = size_plus1 / block_size;     // 从0开始计数
    int block_remain1 = size_plus1 % block_size; // 最后一块大小

    LL block_num5 = size_plus5 / block_size; // 从0开始计数
    int block_remain5 = size_plus5 % block_size;

    marked = (char *)malloc(block_size);
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    LL block_size_of_6 = block_size * 6;
    LL block_low_value = low_value_plus1 - block_size_of_6;
    for (int block_id1 = 0; block_id1 <= block_num1; block_id1++)
    {
        int now_block_size = block_id1 == block_num1 ? block_remain1 : block_size;
        block_low_value += block_size_of_6;
        LL block_high_value = block_low_value + now_block_size * 6 - 6;
        for (i = 0; i < now_block_size; i++)
            marked[i] = 0;
        prime = 5;
        int pindex = 3;
        LL power_prime = 25;
        do
        {
            LL temp;
            if (power_prime > block_low_value)
            {
                temp = power_prime;
                while (temp % 6 != 1)
                {
                    temp += prime;
                }

                first = (temp - block_low_value) / 6;
            }

            else
            {
                // LL m0,n0;
                // exgcd(prime,6,m0,n0);
                // n0 = (n0%6+6)%6;//特解n0, 满足prime倍数的6n+1形式为 n = n0+6k;
                // cout<<
                // LL n1 = block_low_value/6;
                // while (n1%6!=n0)
                // {
                //     n1++;
                // }

                // first = (n1*6+1 - block_low_value)/6;

                // 朴素求法
                temp = (block_low_value + prime - 1) / prime * prime;
                while (temp % 6 != 1)
                {
                    temp += prime;
                }
                first = (temp - block_low_value) / 6;
            }
            for (i = first; i < now_block_size; i += prime)
                marked[i] = 1;
            prime = primes[pindex++];
            power_prime = prime * prime;
        } while (pindex <= pcount && power_prime <= block_high_value);

        // 统计当前缓存块素数的大小
        int block_count = 0;
        for (int i = 0; i < now_block_size; i++)
        {
            if (!marked[i])
                block_count++;
        }
        count += block_count;
    }

    block_low_value = low_value_plus5 - block_size_of_6;
    for (int block_id5 = 0; block_id5 <= block_num5; block_id5++)
    {
        int now_block_size = block_id5 == block_num5 ? block_remain5 : block_size;
        block_low_value += block_size_of_6;
        LL block_high_value = block_low_value + now_block_size * 6 - 6;
        for (i = 0; i < now_block_size; i++)
            marked[i] = 0;
        prime = 5;
        int pindex = 3;
        LL power_prime = 25;
        do
        {
            if (power_prime > block_low_value)
            {
                LL temp = power_prime;
                while (temp % 6 != 5)
                {
                    temp += prime;
                }

                first = (temp - block_low_value) / 6;
            }

            else
            {
                // LL m0,n0;
                // exgcd(prime,6,m0,n0);
                // n0 = (n0%6+6)%6;//特解n0, 满足prime倍数的6n+1形式为 n = n0+6k;
                // LL n1 = block_low_value/6;//当前block块第一个数6n1+1
                // while (n1%6!=n0)
                // {
                //     n1++;
                // }

                // first = (n1*6+5 - block_low_value)/6;
                LL temp = (block_low_value + prime - 1) / prime * prime;
                while (temp % 6 != 5)
                {
                    temp += prime;
                }
                first = (temp - block_low_value) / 6;
            }

            for (i = first; i < now_block_size; i += prime)
                marked[i] = 1;
            prime = primes[pindex++];
            power_prime = prime * prime;
        } while (pindex <= pcount && power_prime <= block_high_value);

        int block_count = 0;
        for (int i = 0; i < now_block_size; i++)
        {
            if (!marked[i])
                block_count++;
        }
        count += block_count;
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    /* Stop the timer */

    elapsed_time += MPI_Wtime();

    /* Print the results */

    if (!id)
    {
        printf("There are %lld primes less than or equal to %lld\n",
               global_count + 3, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
