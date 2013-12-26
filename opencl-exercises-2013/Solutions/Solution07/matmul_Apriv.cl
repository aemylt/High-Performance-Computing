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
    const int N = get_global_size(0);  // assumes square matrices!

    float tmp;

    float Awrk[1024]; //Assume matrix order won't be greater than 1024

    // copy a row of A from global memory to private memory
    for (k = 0; k < N; k++)
        Awrk[k] = A[i*N+k];

    // Compute a row of the product matrix, C
    for (j = 0; j < N; j++) {
        tmp = 0.0;

        for (k = 0; k < N; k++)
            tmp += Awrk[k] * B[k*N+j];

        C[i*N+j] = tmp;
    }
}
