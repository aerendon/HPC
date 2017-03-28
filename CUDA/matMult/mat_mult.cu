#include <bits/stdc++.h>
#include <cuda.h>

#define H 500
#define W 500

using namespace std;

void checkErr(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err), __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
}

void init(float* v) {
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      v[i * W + j] = i;
    }
  }
}

void mult(float *A, float *B,float *C) {
  float aux = 0;
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      aux = 0;
      for(int k=0; k < W; k++)
        aux += A[i * W + k] * B[k * W + j];
     C[i * W + j] = aux;
    }
  }
}

void display(float *v) {
  for(int i = 0; i < 5; i++){
    for(int j = 0; j < 5; j++) {
      cout << v[i * W + j] << " ";
    }
    cout << endl;
  }
}

__global__
void multMat(float *d_A, float *d_B, float *d_C ) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < H && j < W){
    int Pvalue = 0;
    for(int k = 0; k < W; k++) {
       Pvalue += d_A[i * W + k] * d_B[k * W + j];
    }
    d_C[i * W + j] = Pvalue;
  }
}

void multCuda(float *h_A, float* h_B, float* h_D) {
  float *d_A, *d_B, *d_D;
  size_t size = (H * W) * sizeof(float);
  float blockSize = 32;
  dim3 dimBlock(blockSize, blockSize);
  dim3 dimGrid(ceil(W / float(blockSize)), ceil(H / float(blockSize)), 1);

  cudaError_t err = cudaMalloc((void **) &d_A, size); checkErr(err);
  err = cudaMalloc((void **) &d_B, size); checkErr(err);
  err = cudaMalloc((void **) &d_D, size); checkErr(err);

  cudaMemcpy(d_A, h_A, sizeof(float) * H * W, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(float) * H * W, cudaMemcpyHostToDevice);

  multMat<<<dimGrid, dimBlock>>>(d_A, d_B, d_D);
  cudaMemcpy(h_D, d_D, sizeof(float) * H * W, cudaMemcpyDeviceToHost);

  err = cudaFree(d_A); checkErr(err);
  err = cudaFree(d_B); checkErr(err);
  err = cudaFree(d_D); checkErr(err);
}

int check(float *matCPU, float *matGPU) {
  for (int i = 0; i < H * W; i++) {
    if (matCPU[i] != matGPU[i]) return 0;
  }
  return 1;
}

int main() {
  size_t size = (H * W) * sizeof(float);
  float* h_A = (float *) malloc(size);
  float* h_B = (float *) malloc(size);
  float* h_C = (float *) malloc(size);
  float* h_D = (float *) malloc(size);
  //display(h_D);

  init(h_A); init(h_B);
  /*
  clock_t start = clock();
  mult(h_A, h_B, h_C);

  clock_t end = clock();
  double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%lf\n", cpu_time_used);
*/
  clock_t start = clock();
  multCuda(h_A, h_B, h_D);
  clock_t end = clock();
  double gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  std::cout << gpu_time_used << "\n";

  //display(h_C); display(h_D);

  //if (check(h_C, h_D)) printf(":)");
  //else printf(":(");

  free(h_A); free(h_B); free(h_C); free(h_D);
}
