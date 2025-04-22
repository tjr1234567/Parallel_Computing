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
    //算出根号n
    start_n = (int)sqrt((double)n);
    //去掉偶数和3的倍数，整体/3
    size0 = start_n / 3; // the range where the process find the var 'prime'
    //通过索引反退回来数value是否大于根号n，根号n之后的数都平分计算，在那之前的数所有进程都计算
    if (((size0 - 1) * 3 + 5 - (size0 - 1) % 2) > start_n)
        size0--;
    // LL num = n/3 - size0;
    //偶数和3的倍数都去掉，只剩下三分之一的数，需要计算
    LL num = n / 3; // the length of the array in the process

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */

    // low_index = size0 + id*(num - 1)/p;                                  to remove the 'marked0' part from the main circulation
    // high_value = size0 + (id+1) * (num - 1) / p - 1;                     seem to be feckless in shrink the running time
    //索引是从0开始，即0 ~ num-1，故此处为 (num - 1) / p
    //本进程最小的索引
    low_index = id * (num - 1) / p;
    high_index = (id + 1) * (num - 1) / p - 1;
    //如果是最后一个进程，则high_index为所有数最后一个元素的下标，利用通项反推其数字防止其超过n
    if (id == p - 1)
    { // make sure the value the high_index linked is not out of range
        while ((3 * high_index + 5 - high_index % 2) > n)
            high_index--;
    }
    //计算本进程含有的数的个数
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
    //索引是从0开始，故此处也得size0 - 1而不直接是size0
    high_value0 = (size0 - 1) * 3 + 5 - (size0 - 1) % 2;
    //排除素数直接从5开始
    index = 0;
    prime = 5;
    do
    {
        r = prime / 3;
        q = prime % 3; // q=1 or 2
        //划分为6*k + 1 和 6*k - 1两个数列。
        //3后面的素数也肯定是3*p + q的形式，其中q等于1或2
        //同时每种情况都有可能对应到6*k + 1 和 6*k - 1两个数列当中去 （6*k - 1对应的是3*index + 5）
        //以q=1对应到 6k -1的情况为例，其首先得先+4进入到6k - 1队列中，再加n个6表示其为primer的倍数，倍数最小为5
        //即primer + 4 + 6*n = prime * 5 推出 n = (4*primer - 4) / 6 = (12r + 4 - 4) / 6 = 2r
        //所以最小从2r开始然后每次累加primer，这个对应的数反推回索引，就是其 减掉5再除3 的商，就是其索引，将该索引对应的mark设置为1，又因为再该数列内，故必须以6的单位加，但如果只+1又不再是primer的倍数，所以必须每次+primer
        //另外一种就是以q=1对应到6k - 1 （6*k - 1对应的是3*index + 4）
        //首先已经在其队列，那么就直接加6*n即可，也就是 primer + 6*n = primer * m,m为整数，
        //而6肯定不会是primer的倍数，故n肯定是primer的倍数，最小等于primer。
        //反推回索引的时候就是其 减掉4再除3 的商，就是其索引，将该索引对印的mark设置为1
        if (q == 1)
        {
            // 6k-1
            // prime + 4 + 6i = (3r + 1)*5 => i = 2r            6k-1 array
            for (i = 2 * r; prime + 4 + 6 * i <= high_value0; i += prime)
            {
                marked0[(prime + 4 + 6 * i - 5) / 3] = 1;
            }
            //6k+1
            // prime + 6i*prime = any integer => i = 1          6k+1 array
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
    //除去5的倍数
    LL B_n = B_size / 10;        // B_n used in initial the "marked" array to elimite the multiple of 5
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
        //下面的case把5倍数除掉了，直接从7开始删
        prime = 7;
        // caculate the low and high index and their value

        //本进程中本快最小的索引    
        B_low_index = low_index + B_id * B_size;
        B_low_value = 3 * B_low_index + 5 - B_low_index % 2;
        //当B_low_value大于high_value，则跳出循环
        if (B_low_value > (3 * high_index + 5 - high_index % 2))
            break;
        B_high_index = low_index + (B_id + 1) * B_size - 1;
        B_high_value = 3 * B_high_index + 5 - B_high_index % 2;
        //当B_high_value大于high_value，就让其等于high_value
        if (B_high_value > 3 * high_index + 5 - high_index % 2)
        {
            B_high_value = 3 * high_index + 5 - high_index % 2;
        }
        // optimize the initial circulation "for (i=0; i<B_size; i++) marked[i] = 1;"
        //除掉5的倍数的具体操作，此时0标记为非素数
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
            //参考上面，q=1时，会有两种情况，进入6*k-1和进入6*k+1
            if (q == 1)
            {
                //6*k-1，first是第一个primer的倍数，若是B_low_value比它大，就判断是否B_low_value整除primer
                //整除且不等于primer则直接标记上B_low_value对应的index为0，即为非素数
                first = prime + 4 + 12 * r;
                if (first < B_low_value)
                {
                    if (B_low_value % prime == 0 && B_low_value != prime)
                        marked[((B_low_value - 5)) / 3 - B_low_index] = 0;
                    //算到右边那个更大的first
                    t = 6 * prime - (B_low_value - first) % (6 * prime);
                    first = B_low_value + t;
                }
                for (LL j = 0; first + 6 * j <= B_high_value; j += prime)
                {
                    marked[((first + 6 * j - 5)) / 3 - B_low_index] = 0;
                }
                //6*k+1，另外一种情况，但类似
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
            //q=2同样有两种情况
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
    //remain的情况与block的情况类似
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