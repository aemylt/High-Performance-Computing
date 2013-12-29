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

__kernel void propagate(__global t_speed *cells, __global t_speed *tmp_cells)
{
  int ii,jj;            /* generic counters */
  int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  ii = get_global_id(0);
  jj = get_global_id(1);
  ny = get_global_size(0);
  nx = get_global_size(1);
  y_n = (ii + 1) % ny;
  x_e = (jj + 1) % nx;
  y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii *nx + jj].speeds[0]  = cells[ii*nx + jj].speeds[0]; /* central cell, */
                                                                                 /* no movement   */
  tmp_cells[ii *nx + jj].speeds[1] = cells[ii*nx + x_w].speeds[1]; /* east */
  tmp_cells[ii*nx + jj].speeds[2]  = cells[y_s*nx + jj].speeds[2]; /* north */
  tmp_cells[ii *nx + jj].speeds[3] = cells[ii*nx + x_e].speeds[3]; /* west */
  tmp_cells[ii*nx + jj].speeds[4]  = cells[y_n*nx + jj].speeds[4]; /* south */
  tmp_cells[ii*nx + jj].speeds[5] = cells[y_s*nx + x_w].speeds[5]; /* north-east */
  tmp_cells[ii*nx + jj].speeds[6] = cells[y_s*nx + x_e].speeds[6]; /* north-west */
  tmp_cells[ii*nx + jj].speeds[7] = cells[y_n*nx + x_e].speeds[7]; /* south-west */      
  tmp_cells[ii*nx + jj].speeds[8] = cells[y_n*nx + x_w].speeds[8]; /* south-east */   
}
