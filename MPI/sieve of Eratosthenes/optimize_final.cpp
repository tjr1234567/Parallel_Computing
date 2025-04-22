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


// 两个改进：1.前根号n记录素数不用标记数组而用记录数组。
//2.去除2，3方式改进（与取索引方式解耦，直接在最开始定位6n+1和6n+5的边界值），导致索引可以直接/6计算，因而可以从平方开始
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

    // n  - 2 是为了减掉1，从2开始，多减一个1除以p再加1是为了上取整，最后除以2是去除偶数
    //用二进制思考
    proc0_size = (1 + (n - 2) / p) >> 1;
    //0进程所包含的素数必须比根号n大，偏移量是3
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
    //素数表初始化
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
    //通过后面那个for循环从j^2开始每次加two_j可以抹掉奇数里面的素数，例如9，15，21
    //这种做法比之前的优化都更妙
    for (int j = 3; j <= sq_n; j += 2)
    {
        if (st[j])
            continue;
        else
            primes[pcount++] = j;
        //此步是为了省略下面的for计算
        if (j > sq_n / j)
            continue;
        //标记非素数
        int two_j = j << 1;
        for (int t = j * j; t <= sq_n; t += two_j)
        {
            st[t] = 1;
        }
    }
    //本进程的素数个数记录
    count = 0;
    //除掉1，2，3，4，5，6再分配
    low_value = 7 + id * (n - 6) / p;
    high_value = 6 + (id + 1) * (n - 6) / p;
    LL mod6 = low_value % 6;
    //通过以下方式吧2，3倍数去掉
    //若是取余数小于2，那么就要减去mod6加1，否则就是+7，确保low_value_plus1比low_value大
    //若是取余数等于5，那么就要减去mod6加1，否则就是-1，确保high_value_plus5比high_value小
    LL low_value_plus1 = mod6 < 2 ? low_value - mod6 + 1 : low_value - mod6 + 7; // 6n+1序列的第一个数
    LL low_value_plus5 = low_value - mod6 + 5;                                   // 6n+5序列的第一个数
    mod6 = high_value % 6;
    //若正好是6的倍数，那么就取离其最近的6n+1的数，即-6再+1，否则就是减掉mod6+1
    //若正好取余是5，则直接就是6n+5,否则减掉该余数再减1就是最近的数
    LL high_value_plus1 = mod6 == 0 ? high_value - 6 + 1 : high_value - mod6 + 1; // 6n+1序列的最后一个数
    LL high_value_plus5 = mod6 == 5 ? high_value : high_value - mod6 - 1;         // 6n+5序列的最后一个数

    int size_plus1 = (high_value_plus1 - low_value_plus1) / 6 + 1; // 当前进程处理的6n+1序列的大小
    int size_plus5 = (high_value_plus5 - low_value_plus5) / 6 + 1; // 当前进程处理的6n+5序列的大小

    //取一个缓存块的大小能计算多少个数
    int block_size = 120000 * 8 / p;             // 数字120000*8多次实验最优
    //6n+1序列和6n+5序列需要多少块
    LL block_num1 = size_plus1 / block_size;     // 从0开始计数
    int block_remain1 = size_plus1 % block_size; 

    LL block_num5 = size_plus5 / block_size; // 从0开始计数
    int block_remain5 = size_plus5 % block_size; // 最后一块大小

    marked = (char *)malloc(block_size);
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    //一个块内真实存储了多少个数
    LL block_size_of_6 = block_size * 6;
    //为了统一块计算和后面remain计算，先把块大小减掉，最开始那块等于没变，后面的都通过加块大小来得到最小值
    LL block_low_value = low_value_plus1 - block_size_of_6;
    for (int block_id1 = 0; block_id1 <= block_num1; block_id1++)
    {
        //若是块还没算完，继续算块大小数字，否则算剩余的块的大小数字
        int now_block_size = block_id1 == block_num1 ? block_remain1 : block_size;
        //与上面形成呼应
        block_low_value += block_size_of_6;
        //此处计算出来块中最大数字，这个是统一remain
        LL block_high_value = block_low_value + now_block_size * 6 - 6;
        //初始化标记数组
        for (i = 0; i < now_block_size; i++)
            marked[i] = 0;
        //直接从5开始
        prime = 5;
        //因为3，5都去掉计算了，找的是6之后的素数，所以pindex也从3开始
        //1 - 3， 2 - 5，3 - 7。
        int pindex = 3;
        LL power_prime = 25;
        do
        {
            LL temp;
            if (power_prime > block_low_value)
            {
                //找到第一个符合6n+1的数
                temp = power_prime;
                while (temp % 6 != 1)
                {
                    temp += prime;
                }
                //计算第一个索引
                first = (temp - block_low_value) / 6;
            }
            //若是比最小的小
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

                // 朴素求法，找到第一个是primer倍数的数同时符合6n+1的形式
                //若block_low_value是primer倍数，则下列运算后仍是自己，否则就是最近的primer倍数
                //+ prime - 1是为了向上取整到primer倍数
                temp = (block_low_value + prime - 1) / prime * prime;
                //找到符合6n+1的数
                while (temp % 6 != 1)
                {
                    temp += prime;
                }
                first = (temp - block_low_value) / 6;
            }
            for (i = first; i < now_block_size; i += prime)
                marked[i] = 1;
            //直接记录往后走，而不是标记往后走
            prime = primes[pindex++];
            power_prime = prime * prime;
            //下面退出条件前面为根号n里面的数，后一个保证在块内
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
