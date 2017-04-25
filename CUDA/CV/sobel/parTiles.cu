#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>

using namespace cv;
using namespace std;

// Sobel Operator in constant memory
__constant__ float sobel_x[9];
__constant__ float sobel_y[9];


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
void magnitudeKernel(float* x, float* y, unsigned char* r, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    r[idx] = (unsigned char) hypot(x[idx], y[idx]);
  }
}

__device__
unsigned char bound_to_image(unsigned char* image, int row, int col, int width, int height) {
  if (inside_image(row, col, width, height))
    return image[row * width + col];
  else
    return 0;
}


__global__
void sobel(unsigned char* image, unsigned char* out_image, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float tile[32 + 2 * 1][32 + 2 * 1];
  
  if (row < height && col < width) {
    int y_mid = threadIdx.y + 1;
    int x_mid = threadIdx.x + 1;
    tile[y_mid][x_mid] = image[row * width + col];

    if (threadIdx.y == 0 && threadIdx.x == 0) {
      tile[y_mid - 1][x_mid - 1] = bound_to_image(image, row - 1, col - 1, width, height);
    } else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
      tile[y_mid - 1][x_mid + 1] = bound_to_image(image, row - 1, col + 1, width, height);
    } else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
      tile[y_mid + 1][x_mid - 1] = bound_to_image(image, row + 1, col - 1, width, height);
    } else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
      tile[y_mid + 1][x_mid + 1] = bound_to_image(image, row + 1, col + 1, width, height);
    }
    
    if (threadIdx.y == 0) {
      tile[y_mid - 1][x_mid] = bound_to_image(image, row - 1, col, width, height);
    } else if (threadIdx.y == blockDim.y - 1) {
      tile[y_mid + 1][x_mid] = bound_to_image(image, row + 1, col, width, height);
    }
    if (threadIdx.x == 0) {
      tile[y_mid][x_mid - 1] = bound_to_image(image, row, col - 1, width, height);
    } else if (threadIdx.x == blockDim.x - 1) {
      tile[y_mid][x_mid + 1] = bound_to_image(image, row, col + 1, width, height);
    }
    
    __syncthreads();
    
    // Calculate gradient in x-direction
    float grad_x = 0;
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        grad_x += tile[y_mid + i][x_mid + j] * sobel_x[(1 + i) * 3 + (1 + j)];
      }
    }
    
    float grad_y = 0;
    for (int i = -1; i <= 1; i++) {
    	for (int j = -1; j <= 1; j++) {
        grad_y += tile[y_mid + i][x_mid + j] * sobel_y[(1 + i) * 3 + (1 + j)];
      }
    }
    
    out_image[row * width + col] = (unsigned char) hypot(grad_x, grad_y);
  }
}

void sobel(unsigned char *h_img, unsigned char *h_img_sobel, int width, int height) {
  unsigned char *d_img, *d_img_sobel;
  long long size = width * height;
  cudaError_t err;

  err = cudaMalloc((void**) &d_img, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel, size * sizeof(unsigned char)); checkError(err);
  
  err = cudaMemcpy(d_img, h_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice); checkError(err);

  float h_sobel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  err = cudaMemcpyToSymbol(sobel_x, h_sobel_x, 9 * sizeof(float)); checkError(err);
  err = cudaMemcpyToSymbol(sobel_y, h_sobel_y, 9 * sizeof(float)); checkError(err);

  dim3 dim_grid(ceil((double) width / 32), ceil((double) height / 32), 1);
  dim3 dim_block(32, 32, 1);

  sobel<<<dim_grid, dim_block>>>(d_img, d_img_sobel, width, height);
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_img_sobel, d_img_sobel, size * sizeof(unsigned char), cudaMemcpyDeviceToHost); 
	checkError(err);
  err = cudaFree(d_img); 
	checkError(err);
  err = cudaFree(d_img_sobel); 
	checkError(err);
}

void create(Mat& image) {
  int height = image.rows;
  int width = image.cols;
 
  unsigned char *img_sobel = (unsigned char*) malloc(width * height * sizeof(unsigned char));
  unsigned char *img = (unsigned char*) image.data;

  sobel(img, img_sobel, width, height);

	//imshow("Sobel", Mat(height, width, CV_8UC1, img_sobel));
	//waitKey(0)

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

