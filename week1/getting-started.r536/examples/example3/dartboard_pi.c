/*
** A serial program to estimate pi using the dartboard
** (monte-carlo) algorithm.
**
** Imagine a circle inscribed inside a square.
** The area of the circle is, of course: A-circ = pi * sqr(r).
** But we're trying to find pi, so we re-arrange to get:
** pi = A-circ / sqr(r)  -- (eqn 1).
** We know that sqr(r) is the area of one quarter of the square,
** so A-sq = 4 * sqr(r).
** Re-arranging again, we get:
** sqr(r) = A-sq / 4  -- (eqn 2).
** We can subsitute (eqn 2) into (eqn 1) to get:
** pi = 4 * A-circ / A-sq. 
**
** Lastly, if we assume the darts land randomly somewhere inside
** the square, i.e. sometimes within the cicle, and sometimes
** outside the circle, then we can substitute the ratio of areas
** with a ratio of dart counts, i.e. a count of the darts which 
** fell inside the cicrle over a count of those which fell inside
** the square (all of them).
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef TWISTER
#include "mersenne.h"
#endif

#define NDARTS 5000000  /* number of throws at dartboard */
#define ROUNDS 10       /* number of times we throw NDARTS */

/* function prototypes */
double throw_darts (int nthrows);

#define sqr(x)((x)*(x))

int main(int argc, char **argv) 
{  

  double pi;              /* average of pi after "darts" are thrown */
  double avepi;           /* average pi value for all iterations */
  struct timeval timstr;  /* structure to hold elapsed time */
  struct rusage ru;       /* structure to hold CPU time--system and user */
  double tic,toc;         /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;          /* floating point number to record elapsed user CPU time */
  double systim;          /* floating point number to record elapsed system CPU time */
  int ii;

#ifdef DEBUG
  double PI25DT = 3.141592653589793238462643;
#endif

  /* start timing */
  gettimeofday(&timstr,NULL);
  tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  /* Set seed for random number generator */
#ifdef TWISTER
  sgenrand(time(NULL));  /* mersenne twister code */
#else
  srand (time(NULL));    /* c library random number generator */
#endif

  /* perform the Monte Carlo trials */
  avepi = 0;
  for (ii=0; ii<ROUNDS; ii++) {
    pi = throw_darts(NDARTS);
    avepi = ((avepi * ii) + pi)/(ii + 1);  

#ifdef DEBUG
    printf("   After %3d throws, average value of pi = %10.8f (error is %.16f)\n",
	   (NDARTS * (ii + 1)),avepi, fabs(avepi - PI25DT));
#endif

  }     

  /* finish timing */
  gettimeofday(&timstr,NULL);
  toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr=ru.ru_utime;        
  usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  timstr=ru.ru_stime;        
  systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

  /* print to stdout */
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);

  return EXIT_SUCCESS;
}

double throw_darts(int nthrows)
{
  double x_coord;       /* x coordinate, between -1 and 1  */
  double y_coord;       /* y coordinate, between -1 and 1  */
  double pi;            /* pi  */
  double r1,r2;         /* random number between 0 and 1  */
  int score = 0;        /* number of darts that hit circle */
  int ii;
  
  /* "throw darts at board" */
  for (ii=1; ii<=nthrows; ii++) {
    /* generate random numbers for x and y coordinates */
#ifdef TWISTER
    r1 = genrand();
    r2 = genrand();
#else
    r1 = (double)rand()/RAND_MAX;
    r2 = (double)rand()/RAND_MAX;
#endif
    x_coord = (2.0 * r1) - 1.0;
    y_coord = (2.0 * r2) - 1.0;
    
    /* if dart lands in circle, increment score */
    if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
      score++;
  }
  
  /* calculate pi */
  pi = 4.0 * (double)score/(double)nthrows;
  return(pi);
} 

