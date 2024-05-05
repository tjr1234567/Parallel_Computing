#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include <fstream>
#include <vector>
void load_data(int **a, int *storage, std::string fname, int *row, int *col, int *correct);
void compute_shortest_paths(int id, int p, int **a, int n);
int main(int argc, char *argv[])
{
    int **a;      // Doubly-subscripted array.every element points to the head of each row
    int *storage; // Local portion of array elements
    int i, j, k;
    int id; // process rank
    int n;  // columns in matrix. supposing the matrix is square
    int p;  // process number
    int row;
    int col;
    int correct;
    std::string fname = "Floyddata";
    a = (int **)malloc((16) * sizeof(int *));
    storage = (int *)malloc((16) * (16) * sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    load_data(a, storage, fname, &row, &col, &correct);
    for (int x = 0; x < row; x++)
    {
        for (int y = 0; y < col; y++)
        {
            printf("%d,", *(storage + x * col + y));
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
    MPI_Finalize();
    free(a);
    free(storage);
    return 0;
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
