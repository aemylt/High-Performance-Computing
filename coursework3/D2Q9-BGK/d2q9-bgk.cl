#define NSPEEDS         9

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

__kernel void accelerate_flow(const t_param params, __global t_speed *cells, __global int *obstacles)
{
  int ii,jj;     /* generic counters */
  float w1,w2;  /* weighting factors */
  
  /* compute weighting factors */
  w1 = params.density * params.accel / 9.0;
  w2 = params.density * params.accel / 36.0;

  /* modify the first column of the grid */
  jj=0;
  ii = get_global_size(0);
  /* if the cell is not occupied and
  ** we don't send a density negative */
  if( !obstacles[ii*params.nx + jj] && 
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
