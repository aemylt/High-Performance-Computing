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

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include "mpi.h"

#define MASTER 0
#define NUMPARAMS 7
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
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
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int size, int rank, int* distribution);

/* 
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int size, int rank, MPI_Datatype cells_type);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int synchronise(const t_param params, t_speed* cells, int size, int rank, MPI_Datatype cells_type, MPI_Request* req0, MPI_Request* req1, MPI_Request* req2, MPI_Request* req3);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, int size, int rank, MPI_Request* req0, MPI_Request* req1, MPI_Request* req2, MPI_Request* req3);
int rebound_or_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, int size, int rank, int distribution);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells, int size, int rank, MPI_Datatype cells_type);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles, int size, int rank, MPI_Datatype cells_type);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, int size, int rank, MPI_Datatype cells_type);

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
  t_speed* cells     = NULL;  /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;  /* scratch space */
  int*     obstacles = NULL;  /* grid indicating which cells are blocked */
  float*  av_vels   = NULL;  /* a record of the av. velocity computed for each timestep */
  int      ii;                /* generic counter */
  struct timeval timstr;      /* structure to hold elapsed time */
  struct rusage ru;           /* structure to hold CPU time--system and user */
  double tic = 0,toc = 0;             /* floating point numbers to calculate elapsed wallclock time */
  double usrtim = 0;              /* floating point number to record elapsed user CPU time */
  double systim = 0;              /* floating point number to record elapsed system CPU time */
  int size, rank;
  float tmp_av_vels;
  MPI_Datatype cells_type;
  MPI_Aint displacements_cells[1];
  MPI_Datatype types_cells[1];
  int block_length_cells[1];
  int distribution;

  /* parse the command line */
  if(argc != 3) {
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, size, rank, &distribution);

  displacements_cells[0] = 0;
  types_cells[0] = MPI_FLOAT;
  block_length_cells[0] = NSPEEDS;
  MPI_Type_create_struct(1, block_length_cells, displacements_cells, types_cells, &cells_type);
  MPI_Type_commit(&cells_type);

  if (rank == MASTER) {
      /* iterate for maxIters timesteps */
      gettimeofday(&timstr,NULL);
      tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  }

  for (ii=0;ii<params.maxIters;ii++) {
    timestep(params,cells,tmp_cells,obstacles, size, rank, cells_type);
    
    tmp_av_vels = av_velocity(params,cells,obstacles, size, rank, cells_type);
    if (rank == MASTER) av_vels[ii] = tmp_av_vels;
#ifdef DEBUG
    float density = total_density(params,cells, size, rank, cells_type);
    if (rank == MASTER) {
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",density);
    }
#endif
  }
  if (rank == MASTER) {
      gettimeofday(&timstr,NULL);
      toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      getrusage(RUSAGE_SELF, &ru);
      timstr=ru.ru_utime;        
      usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
      timstr=ru.ru_stime;        
      systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
  }
  float reynolds = calc_reynolds(params,cells,obstacles, size, rank, cells_type);
  if (rank == MASTER) {
      /* write final values and free memory */
      printf("==done==\n");
      printf("Reynolds number:\t\t%.12E\n",reynolds);
      printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
      printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
      printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  }
  write_values(params,cells,obstacles,av_vels,size,rank,distribution);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Type_free(&cells_type);
  
  MPI_Finalize();
  
  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int size, int rank, MPI_Datatype cells_type)
{
  MPI_Request* req0, req1, req2, req3;
  accelerate_flow(params,cells,obstacles);
  synchronise(params, cells, size, rank, cells_type, req0, req1, req2, req3);
  propagate(params,cells,tmp_cells, size, rank, req0, req1, req2, req3);
  rebound_or_collision(params,cells,tmp_cells,obstacles);
  return EXIT_SUCCESS; 
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  int ii,jj;     /* generic counters */
  float w1,w2;  /* weighting factors */
  
  /* compute weighting factors */
  w1 = params.density * params.accel / 9.0;
  w2 = params.density * params.accel / 36.0;

  /* modify the first column of the grid */
  jj=0;
  for(ii=1;ii<=params.ny;ii++) {
    /* if the cell is not occupied and
    ** we don't send a density negative */
    if( !obstacles[(ii - 1)*params.nx + jj] && 
        (cells[ii*params.nx + jj].speeds[3] - w1) > 0.0 &&
        (cells[ii*params.nx + jj].speeds[6] - w2) > 0.0 &&
        (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 ) {
      /* increase 'east-side' densities */
      cells[ii*params.nx + jj].speeds[1] += w1;
      cells[ii*params.nx + jj].speeds[5] += w2;
      cells[ii*params.nx + jj].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii*params.nx + jj].speeds[3] -= w1;
      cells[ii*params.nx + jj].speeds[6] -= w2;
      cells[ii*params.nx + jj].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int synchronise(const t_param params, t_speed* cells, int size, int rank, MPI_Datatype cells_type, MPI_Request* req0, MPI_Request* req1, MPI_Request* req2, MPI_Request* req3)
{
    int right = (rank + 1) % size;
    int left = (rank == MASTER) ? size - 1 : rank - 1;
    if (rank % 2 == 0) {
        MPI_Isend(&(cells[params.ny*params.nx]), params.nx, cells_type, right, 0, MPI_COMM_WORLD, req0);
        MPI_Isend(&(cells[params.nx]), params.nx, cells_type, left, 0, MPI_COMM_WORLD, req1);
        MPI_Irecv(cells, params.nx, cells_type, left, 0, MPI_COMM_WORLD, req2);
        MPI_Irecv(&(cells[(params.ny + 1)*params.nx]), params.nx, cells_type, right, 0, MPI_COMM_WORLD, req3);
    } else {
        MPI_Irecv(cells, params.nx, cells_type, left, 0, MPI_COMM_WORLD, req2);
        MPI_Irecv(&(cells[(params.ny + 1)*params.nx]), params.nx, cells_type, right, 0, MPI_COMM_WORLD, req3);
        MPI_Isend(&(cells[params.ny*params.nx]), params.nx, cells_type, right, 0, MPI_COMM_WORLD, req0);
        MPI_Isend(&(cells[params.nx]), params.nx, cells_type, left, 0, MPI_COMM_WORLD, req1);
    }
    return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, int size, int rank, MPI_Request* req0, MPI_Request* req1, MPI_Request* req2, MPI_Request* req3)
{
  int ii,jj;            /* generic counters */
  int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
  MPI_Status status;

  /* loop over _all_ cells */
  for(ii=1;ii<=params.ny;ii++) {
    if (ii == 0) {
        MPI_Wait(&req1, &status);
        MPI_Wait(&req2, &status);
    } else if (ii == params.ny) {
        MPI_Wait(&req0, &status);
        MPI_Wait(&req3, &status);
    }
    for(jj=0;jj<params.nx;jj++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      y_n = ii + 1;
      x_e = (jj + 1) % params.nx;
      y_s = ii - 1;
      x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii *params.nx + jj].speeds[0]  = cells[ii*params.nx + jj].speeds[0]; /* central cell, */
                                                                                     /* no movement   */
      tmp_cells[ii *params.nx + jj].speeds[1] = cells[ii*params.nx + x_w].speeds[1]; /* east */
      tmp_cells[ii*params.nx + jj].speeds[2]  = cells[y_s*params.nx + jj].speeds[2]; /* north */
      tmp_cells[ii *params.nx + jj].speeds[3] = cells[ii*params.nx + x_e].speeds[3]; /* west */
      tmp_cells[ii*params.nx + jj].speeds[4]  = cells[y_n*params.nx + jj].speeds[4]; /* south */
      tmp_cells[ii*params.nx + jj].speeds[5] = cells[y_s*params.nx + x_w].speeds[5]; /* north-east */
      tmp_cells[ii*params.nx + jj].speeds[6] = cells[y_s*params.nx + x_e].speeds[6]; /* north-west */
      tmp_cells[ii*params.nx + jj].speeds[7] = cells[y_n*params.nx + x_e].speeds[7]; /* south-west */      
      tmp_cells[ii*params.nx + jj].speeds[8] = cells[y_n*params.nx + x_w].speeds[8]; /* south-east */      
    }
  }

  return EXIT_SUCCESS;
}

int rebound_or_collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  int ii,jj,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* square of speed of sound */
  const float w0 = 4.0/9.0;    /* weighting factor */
  const float w1 = 1.0/9.0;    /* weighting factor */
  const float w2 = 1.0/36.0;   /* weighting factor */
  float u_x,u_y;               /* av. velocities in x and y directions */
  float u[NSPEEDS];            /* directional velocities */
  float d_equ[NSPEEDS];        /* equilibrium densities */
  float u_sq;                  /* squared velocity */
  float local_density;         /* sum of densities in a particular cell */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for(ii=1;ii<=params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* if the cell contains an obstacle */
      if(obstacles[(ii - 1)*params.nx + jj]) {
          /* called after propagate, so taking values from scratch space
          ** mirroring, and writing into main grid */
          cells[ii*params.nx + jj].speeds[1] = tmp_cells[ii*params.nx + jj].speeds[3];
          cells[ii*params.nx + jj].speeds[2] = tmp_cells[ii*params.nx + jj].speeds[4];
          cells[ii*params.nx + jj].speeds[3] = tmp_cells[ii*params.nx + jj].speeds[1];
          cells[ii*params.nx + jj].speeds[4] = tmp_cells[ii*params.nx + jj].speeds[2];
          cells[ii*params.nx + jj].speeds[5] = tmp_cells[ii*params.nx + jj].speeds[7];
          cells[ii*params.nx + jj].speeds[6] = tmp_cells[ii*params.nx + jj].speeds[8];
          cells[ii*params.nx + jj].speeds[7] = tmp_cells[ii*params.nx + jj].speeds[5];
          cells[ii*params.nx + jj].speeds[8] = tmp_cells[ii*params.nx + jj].speeds[6];
      } else {
          /* compute local density total */
          local_density = 0.0;
          for(kk=0;kk<NSPEEDS;kk++) {
            local_density += tmp_cells[ii*params.nx + jj].speeds[kk];
          }
          /* compute x velocity component */
          u_x = (tmp_cells[ii*params.nx + jj].speeds[1] +
                 tmp_cells[ii*params.nx + jj].speeds[5] +
                 tmp_cells[ii*params.nx + jj].speeds[8]
                 - (tmp_cells[ii*params.nx + jj].speeds[3] +
                    tmp_cells[ii*params.nx + jj].speeds[6] +
                    tmp_cells[ii*params.nx + jj].speeds[7]))
            / local_density;
          /* compute y velocity component */
          u_y = (tmp_cells[ii*params.nx + jj].speeds[2] +
                 tmp_cells[ii*params.nx + jj].speeds[5] +
                 tmp_cells[ii*params.nx + jj].speeds[6]
                 - (tmp_cells[ii*params.nx + jj].speeds[4] +
                    tmp_cells[ii*params.nx + jj].speeds[7] +
                    tmp_cells[ii*params.nx + jj].speeds[8]))
            / local_density;
          /* velocity squared */
          u_sq = u_x * u_x + u_y * u_y;
          /* directional velocity components */
          u[1] =   u_x;        /* east */
          u[2] =         u_y;  /* north */
          u[3] = - u_x;        /* west */
          u[4] =       - u_y;  /* south */
          u[5] =   u_x + u_y;  /* north-east */
          u[6] = - u_x + u_y;  /* north-west */
          u[7] = - u_x - u_y;  /* south-west */
          u[8] =   u_x - u_y;  /* south-east */
          /* equilibrium densities */
          /* zero velocity density: weight w0 */
          d_equ[0] = w0 * local_density * (1.0 - u_sq * (1.0 / (2.0 * c_sq)));
          /* axis speeds: weight w1 */
          d_equ[1] = w1 * local_density * (1.0 + u[1] * (1.0 / c_sq)
                           + (u[1] * u[1]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[2] = w1 * local_density * (1.0 + u[2] * (1.0 / c_sq)
                           + (u[2] * u[2]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[3] = w1 * local_density * (1.0 + u[3] * (1.0 / c_sq)
                           + (u[3] * u[3]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[4] = w1 * local_density * (1.0 + u[4] * (1.0 / c_sq)
                           + (u[4] * u[4]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          /* diagonal speeds: weight w2 */
          d_equ[5] = w2 * local_density * (1.0 + u[5] * (1.0 / c_sq)
                           + (u[5] * u[5]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[6] = w2 * local_density * (1.0 + u[6] * (1.0 / c_sq)
                           + (u[6] * u[6]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[7] = w2 * local_density * (1.0 + u[7] * (1.0 / c_sq)
                           + (u[7] * u[7]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          d_equ[8] = w2 * local_density * (1.0 + u[8] * (1.0 / c_sq)
                           + (u[8] * u[8]) * (1.0 / (2.0 * c_sq * c_sq))
                           - u_sq * (1.0 / (2.0 * c_sq)));
          /* relaxation step */
          for(kk=0;kk<NSPEEDS;kk++) {
            cells[ii*params.nx + jj].speeds[kk] = (tmp_cells[ii*params.nx + jj].speeds[kk]
                               + params.omega *
                               (d_equ[kk] - tmp_cells[ii*params.nx + jj].speeds[kk]));
          }
      }
    }
  }

  return EXIT_SUCCESS; 
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int size, int rank, int* distribution)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */
  MPI_Aint base_addr, addr;
  int remainder = 0;

  if (rank == MASTER) {
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
    
      /* and close up the file */
      fclose(fp);
  }
  
  if (size > 1) {
      MPI_Aint displacements_params[NUMPARAMS];
      MPI_Datatype types_params[NUMPARAMS];
      MPI_Datatype params_type;
      int block_lengths_params[NUMPARAMS];
      t_param send_params;
      if (rank == MASTER) {
          send_params.nx = params->nx;
          send_params.ny = params->ny / (size);
          *distribution = send_params.ny;
          send_params.maxIters = params->maxIters;
          send_params.reynolds_dim = params->reynolds_dim;
          send_params.density = params->density;
          send_params.accel = params->accel;
          send_params.omega = params->omega;
      }
      types_params[0] = MPI_INT;
      block_lengths_params[0] = 1;
      MPI_Address(&(send_params.nx), &base_addr);
      displacements_params[0] = 0;
      types_params[1] = MPI_INT;
      block_lengths_params[1] = 1;
      MPI_Address(&(send_params.ny), &addr);
      displacements_params[1] = addr - base_addr;
      types_params[2] = MPI_INT;
      block_lengths_params[2] = 1;
      MPI_Address(&(send_params.maxIters), &addr);
      displacements_params[2] = addr - base_addr;
      types_params[3] = MPI_FLOAT;
      block_lengths_params[3] = 1;
      MPI_Address(&(send_params.reynolds_dim), &addr);
      displacements_params[3] = addr - base_addr;
      types_params[4] = MPI_FLOAT;
      block_lengths_params[4] = 1;
      MPI_Address(&(send_params.density), &addr);
      displacements_params[4] = addr - base_addr;
      types_params[5] = MPI_FLOAT;
      block_lengths_params[5] = 1;
      MPI_Address(&(send_params.accel), &addr);
      displacements_params[5] = addr - base_addr;
      types_params[6] = MPI_FLOAT;
      block_lengths_params[6] = 1;
      MPI_Address(&(send_params.omega), &addr);
      displacements_params[6] = addr - base_addr;
      
      MPI_Type_create_struct(NUMPARAMS, block_lengths_params, displacements_params, types_params, &params_type);
      MPI_Type_commit(&params_type);
      MPI_Bcast(&send_params, 1, params_type, MASTER, MPI_COMM_WORLD);
      
      if (rank == MASTER) {
          remainder = params->ny % size;
          params->ny = send_params.ny + remainder;
      } else {
          params->nx = send_params.nx;
          params->ny = send_params.ny;
          params->maxIters = send_params.maxIters;
          params->reynolds_dim = send_params.reynolds_dim;
          params->density = send_params.density;
          params->accel = send_params.accel;
          params->omega = send_params.omega;
          *distribution = params->ny;
      }
      MPI_Type_free(&params_type);
  }

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
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed)*((params->ny + 2)*params->nx));
  if (*cells_ptr == NULL) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed)*((params->ny + 2)*params->nx));
  if (*tmp_cells_ptr == NULL) 
    die("cannot allocate memory for tmp_cells",__LINE__,__FILE__);
  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
  if (*obstacles_ptr == NULL) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=1;ii<=params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */ 
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }
  
  MPI_Aint displacements_obstacles[3];
  MPI_Datatype types_obstacles[3];
  MPI_Datatype obstacles_type;
  int block_lengths_obstacles[3];

  MPI_Address(&xx, &base_addr);
  displacements_obstacles[0] = 0;
  types_obstacles[0] = MPI_INT;
  block_lengths_obstacles[0] = 1;
  MPI_Address(&yy, &addr);
  displacements_obstacles[1] = addr - base_addr;
  types_obstacles[1] = MPI_INT;
  block_lengths_obstacles[1] = 1;
  MPI_Address(&blocked, &addr);
  displacements_obstacles[2] = addr - base_addr;
  types_obstacles[2] = MPI_INT;
  block_lengths_obstacles[2] = 1;
  MPI_Type_create_struct(3, block_lengths_obstacles, displacements_obstacles, types_obstacles, &obstacles_type);
  MPI_Type_commit(&obstacles_type);

  if (rank == MASTER) {
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
          if ( blocked != 1 ) 
              die("obstacle blocked value should be 1",__LINE__,__FILE__);
          if (yy > params->ny - 1) {
              int dest = (yy - remainder) / (*distribution);
              yy = (yy - remainder) % (*distribution);
              MPI_Send(&xx, 1, obstacles_type, dest, 0, MPI_COMM_WORLD);
          } else {
              if ( yy<0 )
                  die("obstacle y-coord out of range",__LINE__,__FILE__);
              /* assign to array */
              (*obstacles_ptr)[yy*params->nx + xx] = blocked;
          }
      }

      /* and close the file */
      fclose(fp);
      xx = -1;
      for (ii = 1; ii < size; ii++) {
          MPI_Send(&xx, 1, obstacles_type, ii, 0, MPI_COMM_WORLD);
      }

      /* 
      ** allocate space to hold a record of the avarage velocities computed 
      ** at each timestep
      */
      *av_vels_ptr = (float*)malloc(sizeof(float)*params->maxIters);
  } else {
      MPI_Status status;
      MPI_Recv(&xx, 1, obstacles_type, MASTER, 0, MPI_COMM_WORLD, &status);
      while (xx != -1) {
          if ( yy<0 || yy>params->ny-1 )
              die("obstacle y-coord out of range",__LINE__,__FILE__);
          /* assign to array */
          (*obstacles_ptr)[yy*params->nx + xx] = blocked;
          MPI_Recv(&xx, 1, obstacles_type, MASTER, 0, MPI_COMM_WORLD, &status);
      }
  }
  MPI_Type_free(&obstacles_type);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /* 
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles, int size, int rank, MPI_Datatype cells_type)
{
  int    ii,jj,kk;       /* generic counters */
  int    tot_cells, tmp_cells = 0;  /* no. of cells used in calculation */
  float local_density;  /* total density in cell */
  float tot_u_x, tmp_u_x;        /* accumulated x-components of velocity */

  /* initialise */
  tmp_u_x = 0.0;

  /* loop over all non-blocked cells */
  for(ii=1;ii<=params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!obstacles[(ii - 1)*params.nx + jj]) {
        /* local density total */
        local_density = 0.0;
        for(kk=0;kk<NSPEEDS;kk++) {
          local_density += cells[ii*params.nx + jj].speeds[kk];
        }
        /* x-component of velocity */
        tmp_u_x += (cells[ii*params.nx + jj].speeds[1] +
                    cells[ii*params.nx + jj].speeds[5] +
                    cells[ii*params.nx + jj].speeds[8]
                    - (cells[ii*params.nx + jj].speeds[3] +
                       cells[ii*params.nx + jj].speeds[6] +
                       cells[ii*params.nx + jj].speeds[7])) /
          local_density;
        /* increase counter of inspected cells */
        ++tmp_cells;
      }
    }
  }
  MPI_Reduce(&tmp_u_x, &tot_u_x, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&tmp_cells, &tot_cells, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  if (rank == MASTER) {
      return tot_u_x / (float)tot_cells;
  } else {
      return 0;
  }
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, int size, int rank, MPI_Datatype cells_type)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return av_velocity(params,cells,obstacles, size, rank, cells_type) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells, int size, int rank, MPI_Datatype cells_type)
{
  int ii,jj,kk;        /* generic counters */
  float total, tmp_total = 0.0;  /* accumulator */

  for(ii=1;ii<=params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
        tmp_total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  MPI_Reduce(&tmp_total, &total, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  
  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, int size, int rank, int distribution)
{
  FILE* fp = NULL;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  int send_cells = distribution*params.nx;;
  int recv_cells = send_cells * (size - 1) + params.ny*params.nx;
  const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float* send_pressure = NULL;              /* fluid pressure in grid cell */
  float* send_u_x = NULL;                   /* x-component of velocity in grid cell */
  float* send_u_y = NULL;                   /* y-component of velocity in grid cell */
  float* recv_pressure = NULL;              /* fluid pressure in grid cell */
  float* recv_u_x = NULL;                   /* x-component of velocity in grid cell */
  float* recv_u_y = NULL;                   /* y-component of velocity in grid cell */
  int* recv_obstacles = NULL;
  int* recv_cnts = NULL;
  int* recv_disp = NULL;

  if (rank == MASTER) {
      send_pressure = (float*) malloc(sizeof(float) * params.nx * params.ny);
      send_u_x = (float*) malloc(sizeof(float) * params.nx * params.ny);
      send_u_y = (float*) malloc(sizeof(float) * params.nx * params.ny);
      recv_u_x = (float*) malloc(recv_cells * sizeof(float));
      recv_u_y = (float*) malloc(recv_cells * sizeof(float));
      recv_pressure = (float*) malloc(recv_cells * sizeof(float));
      recv_obstacles = (int*) malloc(recv_cells * sizeof(int));
      recv_cnts = (int*) malloc(size * sizeof(int));
      recv_disp = (int*) malloc(size * sizeof(int));
      recv_cnts[0] = params.ny*params.nx;
      recv_disp[0] = 0;
      for (ii = 1; ii < size; ii++) {
          recv_cnts[ii] = send_cells;
      }
      if (size > 1) {
          recv_disp[1] = params.nx * params.ny;
          for (ii = 2; ii < size; ii++) {
              recv_disp[ii] = recv_disp[ii - 1] + send_cells;
          }
      }
      send_cells = params.nx * params.ny;
  } else {
      send_pressure = (float*) malloc(sizeof(float) * send_cells);
      send_u_x = (float*) malloc(sizeof(float) * send_cells);
      send_u_y = (float*) malloc(sizeof(float) * send_cells);
  }

  for(ii=1;ii<=params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[(ii - 1)*params.nx + jj]) {
        send_u_x[(ii - 1)*params.nx + jj] = send_u_y[(ii - 1)*params.nx + jj] = 0.0;
        send_pressure[(ii - 1)*params.nx + jj] = params.density * c_sq;
      }
      /* no obstacle */
      else {
        local_density = 0.0;
        for(kk=0;kk<NSPEEDS;kk++) {
          local_density += cells[ii*params.nx + jj].speeds[kk];
        }
        /* compute x velocity component */
        send_u_x[(ii - 1)*params.nx + jj] = (cells[ii*params.nx + jj].speeds[1] +
               cells[ii*params.nx + jj].speeds[5] +
               cells[ii*params.nx + jj].speeds[8]
               - (cells[ii*params.nx + jj].speeds[3] +
                  cells[ii*params.nx + jj].speeds[6] +
                  cells[ii*params.nx + jj].speeds[7]))
          / local_density;
        /* compute y velocity component */
        send_u_y[(ii - 1)*params.nx + jj] = (cells[ii*params.nx + jj].speeds[2] +
               cells[ii*params.nx + jj].speeds[5] +
               cells[ii*params.nx + jj].speeds[6]
               - (cells[ii*params.nx + jj].speeds[4] +
                  cells[ii*params.nx + jj].speeds[7] +
                  cells[ii*params.nx + jj].speeds[8]))
          / local_density;
        /* compute pressure */
        send_pressure[(ii - 1)*params.nx + jj] = local_density * c_sq;
      }
    }
  }
  MPI_Gatherv(send_u_x, send_cells, MPI_FLOAT, recv_u_x, recv_cnts, recv_disp, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gatherv(send_u_y, send_cells, MPI_FLOAT, recv_u_y, recv_cnts, recv_disp, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gatherv(send_pressure, send_cells, MPI_FLOAT, recv_pressure, recv_cnts, recv_disp, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Gatherv(obstacles, send_cells, MPI_INT, recv_obstacles, recv_cnts, recv_disp, MPI_INT, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER) {
      fp = fopen(FINALSTATEFILE, "w");
      for (ii = 0; ii < recv_cells; ii++) {
          fprintf(fp,"%d %d %.12E %.12E %.12E %d\n",ii / params.nx,ii % params.nx,recv_u_x[ii],recv_u_y[ii],recv_pressure[ii],recv_obstacles[ii]);
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
  }

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
