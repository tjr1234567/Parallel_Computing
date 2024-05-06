#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include <fstream>
#include <vector>
#define BLOCK_OWNER(index, p, n) ((p * (index + 1) - 1) / n)
#define BLOCK_LOW(id, p, n) (id * n / p)
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW(id + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW(id + 1, p, n) - BLOCK_LOW(id, p, n))
#define MIN(a, b) (a < b ? a : b)

void load_data(int **a, int *storage, std::string fname, int *row, int *col, int *correct);
void print(int **a, int row, int col, int id);
int main(int argc, char *argv[])
{
    int **a;      // Doubly-subscripted array.every element points to the head of each row
    int *storage; // Local portion of array elements
    int i, j, k;
    int id; // process rank
    int m;  // rows in matrix
    int n;  // columns in matrix. supposing the matrix is square
    int p;  // process number
    double time, max_time;
    int correct;
    std::string fname = "matrixdata";
    a = (int **)malloc((16) * sizeof(int *));
    storage = (int *)malloc((16) * (16) * sizeof(int));
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    load_data(a, storage, fname, &m, &n, &correct);
    if (m != n)
    {
        printf("The input matrix is not a square matrix!!!");
        exit(-1);
    }
    for (i = 0; i < p; i++)
    {
        if (i == id)
        {
            print(a, m, n, id);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    free(a);
    free(storage);

    return 0;
}
void print(int **a, int row, int col, int id)
{
    printf("id:%d\n", id);
    fflush(stdout);
    for (int x = 0; x < row; x++)
    {
        for (int y = 0; y < col; y++)
        {
            printf("%d,", a[x][y]);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
}
void load_data(int **a, int *storage, std::string fname, int *row, int *col, int *correct)
{
    *col = 0;
    *row = 0;
    int t_col = 0;
    int t_row = 0;
    std::fstream f(fname);
    std::string eachrow;
    int temp = 0;
    if (!f.is_open())
        std::cout << "Error opening file";
    std::vector<std::vector<int>> num;
    // Defining the loop for getting input from the file
    char buf[1024] = {0};
    while (f.getline(buf, sizeof(buf)))
    {
        std::vector<int> c;
        for (int i = 0; buf[i] != 0; i++)
        {
            if (buf[i] != -17)
            {
                temp = temp * 10 + buf[i] - '0';
            }
            else
            {
                c.push_back(temp);
                temp = 0;
                t_col++;
                i += 2;
            }
        }
        if (*col == t_col)
        {
            t_row++;
            num.push_back(c);
            c.clear();
            t_col = 0;
            continue;
        }
        else if (*col == 0)
        {
            *col = t_col;
            t_row++;
            num.push_back(c);
            c.clear();
            t_col = 0;
        }
        else
        {
            printf("input error!!!");
            *correct = 0;
            return;
        }
    }
    *row = t_row;
    int x, y = 0;
    *a = storage;
    for (x = 0; x < *row; x++) // Outer loop for rows
    {

        for (y = 0; y < *col; y++) // inner loop for columns
        {
            *(storage + x * (*col) + y) = num[x][y]; // Take input from num array and put into a matrix
        }
        *(a + x + 1) = (storage + x * (*col) + y);
    }
}
