#define NSPEEDS         9

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed;

__kernel void accelerate_flow(const int nx, const float density, const float accel, __global t_speed *cells, __global int *obstacles)
{
  int ii,jj;     /* generic counters */
  float w1,w2;  /* weighting factors */
  
  /* compute weighting factors */
  w1 = density * accel / 9.0;
  w2 = density * accel / 36.0;

  /* modify the first column of the grid */
  jj=0;
  ii = get_global_id(0);
  /* if the cell is not occupied and
  ** we don't send a density negative */
  if( !obstacles[ii*nx + jj] && 
      (cells[ii*nx + jj].speeds[3] - w1) > 0.0 &&
      (cells[ii*nx + jj].speeds[6] - w2) > 0.0 &&
      (cells[ii*nx + jj].speeds[7] - w2) > 0.0 ) {
    /* increase 'east-side' densities */
    cells[ii*nx + jj].speeds[1] += w1;
    cells[ii*nx + jj].speeds[5] += w2;
    cells[ii*nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii*nx + jj].speeds[3] -= w1;
    cells[ii*nx + jj].speeds[6] -= w2;
    cells[ii*nx + jj].speeds[7] -= w2;
  }
}
