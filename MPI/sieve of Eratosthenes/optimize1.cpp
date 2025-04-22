// In this optimize I eliminate even numbers to reduce the amout of data to be processed
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define LL long long

//对base的改进：默认跳过偶数，直接从3开始，走奇数索引
int main(int argc, char *argv[]) 
{
    LL count;            /* Local prime count */  //本地素数计数器
    double elapsed_time; /* Parallel execution time */  //并行执行时间
    LL first;            /* Index of first multiple */  //第一个素数倍数的索引
    LL global_count = 0; /* Global prime count */    //全局素数计数器
    LL high_value;       /* Highest value on this proc */   //最高值
    LL i;   // Loop counter */  //循环计数器
    int id;        /* Process ID number */  //进程ID号
    LL index;      /* Index of current prime */  //当前素数的索引
    LL low_value;  /* Lowest value on this proc */  //最低值
    char *marked;  /* Portion of 2,...,'n' */   //2到n的部分 （跳过偶数）
    LL n;          /* Sieving from 2, ..., 'n' */ //从2到n的筛选 （跳过偶数）
    int p;         /* Number of processes */  //进程数
    LL proc0_size; /* Size of proc 0's subarray */   //进程0的子数组大小
    LL prime;      /* Current prime */   //当前素数
    LL size;       /* Elements in 'marked' */   //标记数组的元素数

    MPI_Init(&argc, &argv);

    /* Start the timer */

    MPI_Comm_rank(MPI_COMM_WORLD, &id); // 获取进程ID
    MPI_Comm_size(MPI_COMM_WORLD, &p);  //获取进程数
    MPI_Barrier(MPI_COMM_WORLD);  ///同步
    elapsed_time = -MPI_Wtime();  // 开始计时

    if (argc != 2)  //参数个数不对
    {
        if (!id)
            printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    n = atoll(argv[1]); //获取参数n

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */
    //计算当前进程的最小值，最大值和数组大小
    // 3~n的范围 同时去除偶数
    low_value = 3 + id * (n - 2) / p; // lowest value, should be odd to make the size is correct
    if (low_value % 2 == 0)
    {
        low_value += 1; // 去除偶数
    }
    // 计算当前进程的最大值
    high_value = 2 + (id + 1) * (n - 2) / p; // highest value
    size = (high_value - low_value) / 2 + 1;

    /* Bail out if all the primes used for sieving are
       not all held by process 0 */

    proc0_size = ((n - 2) / p - 1) / 2 + 1;
    // 检查进程0是否包含足够的素数用于筛选
    if ((1 + 2 * proc0_size) < (int)sqrt((double)n))
    {
        if (!id)
            printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    /* Allocate this process's share of the array. */

    marked = (char *)malloc(size);

    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (i = 0; i < size; i++)
        marked[i] = 0;
    //只对进程0进行初始化
    if (!id)
        index = 0;
        //默认去除偶数，从3开始
    prime = 3;
    //first是所属进程第一个primer倍数的索引
    //找到当前进程内第一个primer奇倍数的索引
    do
    {
        // find the first multiple in the process
        //若是比较小的素数的平方大，则从该素数的平方开始，拿到其奇数索引
        if (prime * prime > low_value)
            first = (prime * prime - low_value) / 2;
            //若不是，则从自己最小的low_value开始
        else
        {
            //拿到第一个第一个primer倍数的索引的正常索引
            if (!(low_value % prime))
                first = 0;
            else
                first = prime - (low_value % prime);
            //若本身是奇数，将索引转化为奇数索引
            if ((low_value + first) % 2 != 0)
                first = first / 2;
            else
                //若本身是偶数，加primer转化成离其最近的primer奇数倍数，再转化成奇数索引
                first = (first + prime) / 2;
        }
        //拿到first后，对整个size大小的间隔primer进行标记
        // marked[i] = 1表示该索引是素数的倍数
        for (i = first; i < size; i += prime)
            marked[i] = 1;
        if (!id)
        {
            while (marked[++index])
                ;
            prime = index * 2 + 3;
        }
        if (p > 1)
            MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= n);
    count = 0;
    for (i = 0; i < size; i++)
        if (!marked[i])
            count++;
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