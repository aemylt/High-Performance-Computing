#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {
  
  int nthreads, tid;
  
  /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(tid)
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();
    printf("Hello, world from thread = %d\n", tid);

    /* Only master thread does this */
#pragma omp master
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  }  /* All threads join master thread and disband */

  return EXIT_SUCCESS;  
}
