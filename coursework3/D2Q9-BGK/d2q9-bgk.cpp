/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp" // utility library

#include<time.h>
#include<vector>
#include<iostream>
#include<sys/time.h>
#include<sys/resource.h>
#include<cstdlib>
#include<cstdio>
#include"err_code.c"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define NGROUPS 64
#define NUNITS  16

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  int tot_cells;
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed;

enum boolean { FALSE, TRUE };

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, std::vector<t_speed> & cells_ptr,
               std::vector<int> & obstacles_ptr, float** av_vels_ptr);

int write_values(const t_param params, std::vector<t_speed> & cells, std::vector<int> & obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, std::vector<t_speed> & cells_ptr,
             std::vector<int> & obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, std::vector<t_speed> & cells);

/* compute average velocity */
float av_velocity(const t_param params, cl::Buffer cell_buf, cl::Buffer obs_buf, cl::Kernel sum_velocity, cl::Buffer loc_vel, cl::CommandQueue queue);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, cl::Buffer cell_buf, cl::Buffer obs_buf, cl::Kernel sum_velocity, cl::Buffer loc_vel, cl::CommandQueue queue);

/* utility functions */
void die(const char* message, const int line, const char *file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile;         /* name of the input parameter file */
  char*    obstaclefile;      /* name of a the input obstacle file */
  t_param  params;            /* struct to hold parameter values */
  std::vector<t_speed> cells;  /* grid containing fluid densities */
  cl::Buffer cell_buf;
  cl::Buffer tmp_buf;
  std::vector<int> obstacles;  /* grid indicating which cells are blocked */
  cl::Buffer obs_buf;
  cl::Buffer loc_vel;
  float*  av_vels   = NULL;  /* a record of the av. velocity computed for each timestep */
  int      ii;                /* generic counter */
  struct timeval timstr;      /* structure to hold elapsed time */
  struct rusage ru;           /* structure to hold CPU time--system and user */
  double tic,toc;             /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;              /* floating point number to record elapsed user CPU time */
  double systim;              /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if(argc != 3) {
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, cells, obstacles, &av_vels);
  
  try {
      // Create a context
      cl::Context context(DEVICE);

      // Load in kernel source, creating a program object for the context

      cl::Program program(context, util::loadProgram("d2q9-bgk.cl"));
      program.build(context.getInfo<CL_CONTEXT_DEVICES>(), "-cl-mad-enable");

      // Get the command queue
      cl::CommandQueue queue(context);

      // Create the kernel functor
 
      auto accelerate_flow_and_propagate = cl::make_kernel<float, float, cl::Buffer, cl::Buffer, cl::Buffer>(program, "accelerate_flow_and_propagate");
      auto rebound_or_collision = cl::make_kernel<float, cl::Buffer, cl::Buffer, cl::Buffer>(program, "rebound_or_collision");
      cl::Kernel sum_velocity(program, "sum_velocity");
      obs_buf = cl::Buffer(context, begin(obstacles), end(obstacles), true);
      tmp_buf = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(t_speed) * params.nx * params.ny);
      loc_vel = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * NGROUPS);

      /* iterate for maxIters timesteps */
      gettimeofday(&timstr,NULL);
      tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      cell_buf = cl::Buffer(context, begin(cells), end(cells), true);
    
      for (ii=0;ii<params.maxIters;ii++) {
        accelerate_flow_and_propagate(cl::EnqueueArgs(queue, cl::NDRange(params.ny, params.nx), cl::NDRange(1, params.nx)), params.density, params.accel, cell_buf, tmp_buf, obs_buf);
        rebound_or_collision(cl::EnqueueArgs(queue, cl::NDRange(params.ny * params.nx), cl::NDRange(params.nx)),params.omega,cell_buf,tmp_buf,obs_buf);
        cl::copy(queue, cell_buf, begin(cells), end(cells));
        cell_buf = cl::Buffer(context, begin(cells), end(cells), true);
        av_vels[ii] = av_velocity(params,cell_buf,obs_buf,sum_velocity,loc_vel,queue);
    #ifdef DEBUG
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",total_density(params,cells));
    #endif
      }
      gettimeofday(&timstr,NULL);
      toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      getrusage(RUSAGE_SELF, &ru);
      timstr=ru.ru_utime;        
      usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      timstr=ru.ru_stime;        
      systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
    
      /* write final values and free memory */
      printf("==done==\n");
      printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cell_buf,obs_buf,sum_velocity,loc_vel,queue));
      printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
      printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
      printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
      write_values(params,cells,obstacles,av_vels);
      finalise(&params, cells, obstacles, &av_vels);
  } catch (cl::Error err) {
		std::cout << "Exception\n";
		std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
            << ")"
            << std::endl;
  }
  
  return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, std::vector<t_speed> & cells_ptr,
               std::vector<int> & obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */

  /* open the parameter file */
  fp = fopen(paramfile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input parameter file: %s", paramfile);
    die(message,__LINE__,__FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp,"%d\n",&(params->nx));
  if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->ny));
  if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->maxIters));
  if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
  if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->density));
  if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->accel));
  if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->omega));
  if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);
  params->tot_cells = params->nx * params->ny;

  /* and close up the file */
  fclose(fp);

  /* 
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  cells_ptr.resize(params->ny*params->nx);
  if (cells_ptr.size() == 0) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);

  /* the map of obstacles */
  obstacles_ptr.resize(params->ny*params->nx);
  if (obstacles_ptr.size() == 0) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */ 
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input obstacles file: %s", obstaclefile);
    die(message,__LINE__,__FILE__);
  }

  /* read-in the blocked cells list */
  while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if ( retval != 3)
      die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
    if ( xx<0 || xx>params->nx-1 )
      die("obstacle x-coord out of range",__LINE__,__FILE__);
    if ( yy<0 || yy>params->ny-1 )
      die("obstacle y-coord out of range",__LINE__,__FILE__);
    if ( blocked != 1 ) 
      die("obstacle blocked value should be 1",__LINE__,__FILE__);
    /* assign to array */
    (obstacles_ptr)[yy*params->nx + xx] = blocked;
    params->tot_cells--;
  }
  
  /* and close the file */
  fclose(fp);

  /* 
  ** allocate space to hold a record of the avarage velocities computed 
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, std::vector<t_speed> & cells_ptr,
             std::vector<int> & obstacles_ptr, float** av_vels_ptr)
{
  /* 
  ** free up allocated memory
  */
  std::vector<t_speed>().swap(cells_ptr);

  std::vector<int>().swap(obstacles_ptr);

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, cl::Buffer cell_buf, cl::Buffer obs_buf, cl::Kernel sum_velocity, cl::Buffer loc_vel, cl::CommandQueue queue)
{
  int ii;
  std::vector<float> results(NGROUPS);
  float tot_u_x = 0;
  auto reduce = cl::make_kernel<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer>(sum_velocity);
  reduce(cl::EnqueueArgs(queue, cl::NDRange(NGROUPS * NUNITS), cl::NDRange(NUNITS)), cell_buf, obs_buf, cl::Local(sizeof(float) * NUNITS), params.nx * params.ny, loc_vel);
  cl::copy(queue, loc_vel, begin(results), end(results));
  for (int ii = 0; ii < NGROUPS; ii++) {
      tot_u_x += results[ii];
  }

  return tot_u_x / (float)params.tot_cells;
}

float calc_reynolds(const t_param params, cl::Buffer cell_buf, cl::Buffer obs_buf, cl::Kernel sum_velocity, cl::Buffer loc_vel, cl::CommandQueue queue)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return av_velocity(params,cell_buf,obs_buf,sum_velocity,loc_vel, queue) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, std::vector<t_speed> & cells)
{
  int ii,jj,kk;        /* generic counters */
  float total = 0.0;  /* accumulator */

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
        total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  
  return total;
}

int write_values(const t_param params, std::vector<t_speed> & cells, std::vector<int> & obstacles, float *av_vels)
{
  FILE* fp;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */

  fp = fopen(FINALSTATEFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[ii*params.nx + jj]) {
        u_x = u_y = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
        local_density = 0.0;
        for(kk=0;kk<NSPEEDS;kk++) {
          local_density += cells[ii*params.nx + jj].speeds[kk];
        }
        /* compute x velocity component */
        u_x = (cells[ii*params.nx + jj].speeds[1] +
               cells[ii*params.nx + jj].speeds[5] +
               cells[ii*params.nx + jj].speeds[8]
               - (cells[ii*params.nx + jj].speeds[3] +
                  cells[ii*params.nx + jj].speeds[6] +
                  cells[ii*params.nx + jj].speeds[7]))
          / local_density;
        /* compute y velocity component */
        u_y = (cells[ii*params.nx + jj].speeds[2] +
               cells[ii*params.nx + jj].speeds[5] +
               cells[ii*params.nx + jj].speeds[6]
               - (cells[ii*params.nx + jj].speeds[4] +
                  cells[ii*params.nx + jj].speeds[7] +
                  cells[ii*params.nx + jj].speeds[8]))
          / local_density;
        /* compute pressure */
        pressure = local_density * c_sq;
      }
      /* write to file */
      fprintf(fp,"%d %d %.12E %.12E %.12E %d\n",ii,jj,u_x,u_y,pressure,obstacles[ii*params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }
  for (ii=0;ii<params.maxIters;ii++) {
    fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
