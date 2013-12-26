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
                __global float* C,
                __local  float* Bwrk)
{
    int k, j;
    const int i = get_global_id(0);
    const int N = get_global_size(0);  // assumes square arrays!

    float tmp;

    float Awrk[1024]; //Assume matrix order won't be greater than 1024

    int iloc = get_local_id(0);
    int nloc = get_local_size(0);
  
    // copy a row of A from global memory to private memory
    for (k = 0; k < N; k++)
        Awrk[k] = A[i*N+k];

    // Compute a row of the product matrix, C
    for (j = 0; j < N; j++) {

        // copy a column of B for all work-items to share
        for (k = iloc; k < N; k += nloc)
            Bwrk[k] = B[k*N+j];

        // make sure all writes to local memory finished before proceeding
        barrier(CLK_LOCAL_MEM_FENCE); 

        tmp = 0.0f;

        for (k = 0; k < N; k++)
            tmp += Awrk[k] * Bwrk[k];

        C[i*N+j] = tmp;

        // make sure its safe to go to the next iteration of j
        // and update local memory (Bwrk)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
