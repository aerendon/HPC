#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <limits.h>
#include <math.h>

#define col 10000000

void initVec(int *a, int *b, int *sol) {
  for (int i = 0; i < col; i++)
    a[i] = b[i] = i * 2;

  for (int i = 0; i < col; i++)
    sol[i] = 0;
}

void showVector(int *a) {
  for (int i = 0; i < col; i++)
    printf("%d  ", a[i]);
  printf("\n");
}

void vecAdd(int *a, int *b, int *sol) {
  for (int i = 0; i < col; i++)
    sol[i] = a[i] + b[i];
}

__global__
void vecAddKernel(int *d_a, int *d_b, int *d_sol) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < col) d_sol[i] = d_a[i] + d_b[i];
}

void vecAddCuda(int *h_a, int *h_b, int *h_sol) {
  int *d_a, *d_b, *d_sol;
  
  size_t size = col * sizeof(int);
  
  cudaMalloc((void **) &d_a, size);
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_b, size);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  
  cudaMalloc((void **) &d_sol, size);
  
  vecAddKernel<<<ceil(col/256.0), 256>>>(d_a, d_b, d_sol);
  
  cudaMemcpy(h_sol, d_sol, size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_a); cudaFree(d_sol); cudaFree(d_sol);
}

int check(int *seq, int *cuda) {
  for(int i = 0; i < col; i++) {
    if(seq[i] != cuda[i]) return 0;
  }
  return 1;
}


int main() {
  size_t size = col * sizeof(int);
  int *h_a = (int *) malloc(size);
  int *h_b = (int *) malloc(size);
  int *h_sol = (int *) malloc(size);
  int *h_cuda = (int *) malloc(size);
  //double start = 0, time_seq = 0, time_omp = 0;
  clock_t begin, end;
  double time_spent = 0;

  initVec(h_a, h_b, h_sol);

  begin = clock();
  vecAdd(h_a, h_b, h_sol);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%f\n", time_spent);

  
  begin = clock();
  vecAddCuda(h_a, h_b, h_cuda);
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  
  if(check(h_sol, h_cuda))
    printf(":)\n");
  else printf(":(\n");
  
  //showVector(h_sol);
  //showVector(h_cuda);
  
  free(h_a); free(h_b); free(h_sol); free(h_cuda);

  //showVector(h_a);
  //showVector(h_b);
  //showVector(h_sol);
  printf("%f\n", time_spent);

  return 0;
}
