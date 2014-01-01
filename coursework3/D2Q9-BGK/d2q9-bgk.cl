#define NSPEEDS         9

__kernel void accelerate_flow_and_propagate(const float density, const float accel, __global float *cells, __global float *tmp_cells, __global int *obstacles)
{
  int ii,jj,kk,nx,ny;            /* generic counters */
  int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
  float w1,w2;  /* weighting factors */
  float cell[NSPEEDS];
  int size;
  ii = get_global_id(0);
  jj = get_global_id(1);
  ny = get_global_size(0);
  nx = get_global_size(1);
  size = nx * ny;
  
  /* compute weighting factors */
  w1 = density * accel / 9.0;
  w2 = density * accel / 36.0;

  for (kk = 0; kk < NSPEEDS; kk++) {
    cell[kk] = cells[kk * size + ii * nx + jj];
  }

  /* if the cell is not occupied and
  ** we don't send a density negative */
  if( jj == 0 &&
      !obstacles[ii*nx + jj] && 
      (cell[3] - w1) > 0.0 &&
      (cell[6] - w2) > 0.0 &&
      (cell[7] - w2) > 0.0 ) {
    /* increase 'east-side' densities */
    cell[1] += w1;
    cell[5] += w2;
    cell[8] += w2;
    /* decrease 'west-side' densities */
    cell[3] -= w1;
    cell[6] -= w2;
    cell[7] -= w2;
  }

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  y_n = (ii + 1) % ny;
  x_e = (jj + 1) % nx;
  y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii *nx + jj]  = cell[0]; /* central cell, */
                                                      /* no movement   */
  tmp_cells[size + ii *nx + x_e] = cell[1]; /* east */
  tmp_cells[size * 2 + y_n*nx + jj] = cell[2]; /* north */
  tmp_cells[size * 3 + ii *nx + x_w] = cell[3]; /* west */
  tmp_cells[size * 4 + y_s*nx + jj]  = cell[4]; /* south */
  tmp_cells[size * 5 + y_n*nx + x_e] = cell[5]; /* north-east */
  tmp_cells[size * 6 + y_n*nx + x_w] = cell[6]; /* north-west */
  tmp_cells[size * 7 + y_s*nx + x_w] = cell[7]; /* south-west */      
  tmp_cells[size * 8 + y_s*nx + x_e] = cell[8]; /* south-east */   
}

__kernel void rebound_or_collision(const float omega, __global float *cells, __global float *tmp_cells, __global int *obstacles)
{
  int ii,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* square of speed of sound */
  const float w0 = 4.0/9.0;    /* weighting factor */
  const float w1 = 1.0/9.0;    /* weighting factor */
  const float w2 = 1.0/36.0;   /* weighting factor */
  float u_x,u_y;               /* av. velocities in x and y directions */
  float u[NSPEEDS];            /* directional velocities */
  float d_equ[NSPEEDS];        /* equilibrium densities */
  float u_sq;                  /* squared velocity */
  float local_density;         /* sum of densities in a particular cell */
  int size = get_global_size(0);

  ii = get_global_id(0);

  float tmp[NSPEEDS];
  float cell[NSPEEDS];
  for (kk = 0; kk < NSPEEDS; kk++) {
      tmp[kk] = tmp_cells[size * kk + ii];
  }

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  /* if the cell contains an obstacle */
  if(obstacles[ii]) {
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      cell[1] = tmp[3];
      cell[2] = tmp[4];
      cell[3] = tmp[1];
      cell[4] = tmp[2];
      cell[5] = tmp[7];
      cell[6] = tmp[8];
      cell[7] = tmp[5];
      cell[8] = tmp[6];
  } else {
      /* compute local density total */
      local_density = 0.0;
      for(kk=0;kk<NSPEEDS;kk++) {
        local_density += tmp[kk];
      }
      /* compute x velocity component */
      u_x = (tmp[1] +
             tmp[5] +
             tmp[8]
             - (tmp[3] +
                tmp[6] +
                tmp[7]))
        / local_density;
      /* compute y velocity component */
      u_y = (tmp[2] +
             tmp[5] +
             tmp[6]
             - (tmp[4] +
                tmp[7] +
                tmp[8]))
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
        cell[kk] = (tmp[kk]
                           + omega *
                           (d_equ[kk] - tmp[kk]));
      }
   }
   for (kk = 0; kk < NSPEEDS; kk++) {
       cells[kk * size + ii] = cell[kk];
   }
}

__kernel void sum_velocity(__global t_speed *cells, global int *obstacles, __local float* scratch, __const int length, __global float* result) {
  int global_index = get_global_id(0);
  int kk;
  float local_density;
  float accumulator = 0;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    if (!obstacles[global_index]) {
       /* local density total */
      local_density = 0.0;
      for(kk=0;kk<NSPEEDS;kk++) {
        local_density += cells[length * kk + global_index];
      }
      /* x-component of velocity */
      accumulator += (cells[length + global_index] +
                  cells[length * 5 + global_index] +
                  cells[length * 8 + global_index]
                  - (cells[length * 3 + global_index] +
                     cells[length * 6 + global_index] +
                     cells[length * 7 + global_index])) /
        local_density;
    }
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      scratch[local_index] += scratch[local_index + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}
