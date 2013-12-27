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

    float Awrk[1024]; //Assume matrix order won't be greater than 1024

    if (i < N) 
    {

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

}
