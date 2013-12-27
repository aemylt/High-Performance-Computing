//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication kernel
//
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------


__kernel void mmul(
                    const unsigned int N,
                    __global const float*A,
                    __global const float*B,
                    __global       float*C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;

    float tmp = 0.0f;

    if (i<N && j<N){
        
       for (k = 0; k < N; k++)
           tmp += A[i*N+k] * B[k*N+j];
     
       C[i*N+j] = tmp;
    }

}
