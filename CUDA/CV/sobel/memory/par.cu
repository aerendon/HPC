#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdio>

#define TILE_SIZE 

using namespace cv;
using namespace std;

void checkError(cudaError_t err) {
  if ((err) != cudaSuccess) {
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(EXIT_FAILURE);
  }
}

__device__
bool inside_image(int row, int col, int width, int height) {
  return row >= 0 && row < height && col >= 0 && col < width;
}

__global__
void convolutionKernel(unsigned char* image, float* kernel, float* out_image, int kernel_n, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float tile[];

  if (row < height && col < width) {
    int n = kernel_n / 2;
    float accumulation = 0;
    for (int i = -n; i <= n; i++) {
      for (int j = -n; j <= n; j++) {
        if (inside_image(row + i, col + j, width, height)) {
          int image_idx = (row + i) * width + (col + j);
          int kernel_idx = (n + i) * kernel_n + (n + j);
          accumulation += image[image_idx] * kernel[kernel_idx];
        }
      }
    }
    out_image[row * width + col] = accumulation;
  }
}

__global__
void magnitudeKernel(float* x, float* y, unsigned char* r, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    r[idx] = (unsigned char) hypot(x[idx], y[idx]);
  }
}

void sobel(unsigned char *h_img, unsigned char *h_img_sobel, int width, int height) {
  unsigned char *d_img, *d_img_sobel;
  float *d_img_sobel_x, *d_img_sobel_y;
  float *d_sobel_x, *d_sobel_y;
  long long size = width * height;
  cudaError_t err;
  cudaEvent_t start, stop;

  err = cudaMalloc((void**) &d_img, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel_x, size * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel_y, size * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_sobel_x, 9 * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_sobel_y, 9 * sizeof(float)); checkError(err);

  err = cudaMemcpy(d_img, h_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice); checkError(err);

  float h_sobel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  err = cudaMemcpy(d_sobel_x, h_sobel_x, 9 * sizeof(float), cudaMemcpyHostToDevice); checkError(err);
  err = cudaMemcpy(d_sobel_y, h_sobel_y, 9 * sizeof(float), cudaMemcpyHostToDevice); checkError(err);

  int block_size = 32;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);

  convolutionKernel<<<dim_grid, dim_block>>>(d_img, d_sobel_x, d_img_sobel_x, 3, width, height);
  cudaDeviceSynchronize();

  convolutionKernel<<<dim_grid, dim_block>>>(d_img, d_sobel_y, d_img_sobel_y, 3, width, height);
  cudaDeviceSynchronize();

  magnitudeKernel<<<dim_grid, dim_block>>>(d_img_sobel_x, d_img_sobel_y, d_img_sobel, width, height);
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_img_sobel, d_img_sobel, size * sizeof(unsigned char), cudaMemcpyDeviceToHost); checkError(err);

  err = cudaFree(d_img); checkError(err);
  err = cudaFree(d_img_sobel_x); checkError(err);
  err = cudaFree(d_img_sobel_y); checkError(err);
  err = cudaFree(d_img_sobel); checkError(err);
}

void create(Mat& image) {
  int height = image.rows;
  int width = image.cols;

  unsigned char *img_sobel = (unsigned char*) malloc(width * height * sizeof(unsigned char));
  unsigned char *img = (unsigned char*) image.data;

  sobel(img, img_sobel, width, height);

  //imshow("Input", Mat(height, width, CV_8UC1, img));
  //waitKey(0);
  //imshow("Sobel operator", Mat(height, width, CV_8UC1, img_sobel));
  //waitKey(0);

  free(img_sobel);
}

int main(int argc, char** argv) {
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

  if(!image.data) {
    cout <<  "Could not open or find the image" << endl ;
    return -1;
  }

  clock_t start = clock();
  create(image);
  clock_t end = clock();
  double gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%lf\n", gpu_time_used);

  return 0;
}
