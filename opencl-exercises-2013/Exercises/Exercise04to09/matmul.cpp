//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication 
//
//  PURPOSE: This is a simple matrix multiplication program
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"

int main(void)
{

    int N;   // A[N][N], B[N][N], C[N][N]
    int sz;  // number of elements in each matrix
    float tmp;

    N = ORDER;

    sz = N * N;


    std::vector<float> A(sz); // Matrix A
    std::vector<float> B(sz); // Matrix B
    std::vector<float> C(sz); // Matrix C


    initmat(N, N, N, A, B, C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on CPU ======\n",ORDER);
 
    zero_mat(N, N, C);


    util::Timer timer;


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = tmp;
        }
    }
              
    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    results(N, N, N, C, rtime);

}
