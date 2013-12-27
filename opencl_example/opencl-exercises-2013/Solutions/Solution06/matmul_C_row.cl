//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication kernel
//
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------


__kernel void mmul(

                const unsigned int N,

                __global float* A,

                __global float* B,

                __global float* C)

{

    int k, j;

    int i = get_global_id(0);

    float tmp;

    if (i < N) 
    {

        for (j = 0; j < N; j++) {

            tmp = 0.0;

            for (k = 0; k < N; k++)

                tmp += A[i*N+k] * B[k*N+j];

            C[i*N+j] = tmp;

        }

    }

}
