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
//           We run a distinct dot-product for each element of the
//           product matrix, C.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, November 2013 
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

int main(void)
{

    int N;   // A[N][N], B[N][N], C[N][N]
    int sz;  // number of elements in each matrix
    float tmp;

    N = ORDER;

    sz = N * N;

    std::vector<float> h_A(sz); // Matrix A on the host
    std::vector<float> h_B(sz); // Matrix B on the host
    std::vector<float> h_C(sz); // Matrix C on the host

    cl::Buffer d_A;    // matrix A on the device
    cl::Buffer d_B;    // matrix B on the device
    cl::Buffer d_C;    // matrix C on the device



    initmat(N, N, N, h_A, h_B, h_C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on CPU ======\n",ORDER);
 
    zero_mat(N, N, h_C);


    util::Timer timer;


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += h_A[i*N+k] * h_B[k*N+j];
            }
            h_C[i*N+j] = tmp;
        }
    }
              
    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    results(N, N, N, h_C, rtime);

    printf("\n===== Parallel matrix mult (dot prod), order %d on CPU ======\n",ORDER);
 
    zero_mat(N, N, h_C);
    try
    {
   
       cl::Context context(DEVICE);

       // Load in kernel source, creating a program object for the context.
       // Build program explicitly so I can catch the error and display
       // compiler error messages (should any be generated)

       cl::Program program(context, util::loadProgram("matmul1.cl"));
       try
       {
           program.build();
       }
       catch (cl::Error error)
       {
          // If it was a build error then show the error
          if (error.err() == CL_BUILD_PROGRAM_FAILURE)
           {
               std::vector<cl::Device> devices;
               devices = context.getInfo<CL_CONTEXT_DEVICES>();
               std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
               std::cerr << built << "\n";
           }
           throw error;
       }


        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
 
        auto mmul = cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>(program, "mmul");

        util::Timer timer;


        d_A   = cl::Buffer(context, begin(h_A), end(h_A), true);
        d_B   = cl::Buffer(context, begin(h_B), end(h_B), true);
        d_C   = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * sz);

        mmul(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(N,N)),
            N, 
            d_A,
            d_B,
            d_C);

        cl::copy(queue, d_C, begin(h_C), end(h_C));

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        results(N, N, N, h_C, rtime);
          
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr 
            << "ERROR: "
            << err.what()
            << std::endl;
 
    }

}
