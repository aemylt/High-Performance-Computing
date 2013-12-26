//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication kernel
//
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------


__kernel void mmul(
                __global const float* A,
                __global const float* B,
                __global float* C)
{

    int k, j;
    const int i = get_global_id(0);
    const int N = get_global_size(0); // assumes square matrices!

    float tmp;

    for (j = 0; j < N; j++) {
        tmp = 0.0f;

        for (k = 0; k < N; k++)
            tmp += A[i*N+k] * B[k*N+j];

        C[i*N+j] = tmp;

    }
}
