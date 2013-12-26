//------------------------------------------------------------------------------
//
// Name:       pi.cpp
// 
// Purpose:    Numeric integration to estimate pi
//
// HISTORY:    Written by Tim Mattson, Nov 2013
//             

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

#define INSTEPS (512*512)

int main(void)
{
    float x, sum=0.0f, pi;
    int i, nsteps = INSTEPS;	
    float step_size;
    step_size = 1.0f/((float)nsteps);

    try
    {

       util::Timer timer;

       for(i=0; i<nsteps;i++){
           x = (i+0.5f)*step_size;   
           sum += 4.0f/(1.0f+x*x);  
      } 
        
      pi = sum * step_size;
	
      double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.;
      printf("\nThe calculation ran in %lf seconds\n", rtime);
      printf(" pi = %f for %d steps\n", pi, nsteps);

    }
      catch (cl::Error err) {
        std::cout << "Exception\n";
	std::cerr 
        << "ERROR: "
        << err.what()
        << std::endl;
    }
}
