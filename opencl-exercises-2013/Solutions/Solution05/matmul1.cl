//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication kernel
//
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------


__kernel void mmul(
                    __global const float*A,
                    __global const float*B,
                    __global       float*C)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int N = get_global_size(0);	// assumes square matrices!
    int k;

    float tmp = 0.0f;

    for (k = 0; k < N; k++)
        tmp += A[i*N+k] * B[k*N+j];
     
    C[i*N+j] = tmp;
}
