#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void checkErr(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err), __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
}

void showVector(unsigned char *mat, int row, int column) {
  for (int i = 0; i < 6; i++)
    printf("%u  ", mat[i]);
  printf("\n");
}

__global__
void pictureKernel(unsigned char *d_pin, unsigned char *d_pout, int row, int column) {
  int Column = blockDim.x * blockIdx.x + threadIdx.x;
  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  if ((Row < row) && (Column < column)) d_pout[Row * column + Column] = 2 * d_pin[Row * column + Column];
}

void pictureCuda(unsigned char *h_pin, unsigned char *h_pout, int row, int column) {
  unsigned char *d_pin, *d_pout;
  
  size_t size = row * column * sizeof(char);
  
  cudaError_t err = cudaMalloc((void **) &d_pin, size);
  checkErr(err); 
  cudaMemcpy(d_pin, h_pin, size, cudaMemcpyHostToDevice);
  err = cudaMalloc((void **) &d_pout, size);
  checkErr(err);
  cudaMemcpy(d_pout, h_pout, size, cudaMemcpyHostToDevice);
  
  int block = 32;
  dim3 dim_grid(ceil((double) column / block), ceil((double) row / block), 1);
  dim3 dim_block(block, block, 1);
  pictureKernel<<<dim_grid, dim_block>>>(d_pin, d_pout, row, column);
  
  //showVector(d_pout, column, row);
  err = cudaMemcpy(h_pout, d_pout, size, cudaMemcpyDeviceToHost); checkErr(err);

  err = cudaFree(d_pin); checkErr(err);
  err = cudaFree(d_pout); checkErr(err);
}

void pictureSeq(unsigned char *h_pin, unsigned char *h_pout, int row, int column) {

  int pos = 0;
  for (int j = 0; j < row * column * 3; j += 3) {
    //intensity = 0.2989s*red + 0.5870*green + 0.1140*blue
    //OpenCV -> color BGR
    h_pout[pos] = (0.1140 * h_pin[j]) + (0.5870 * h_pin[j + 1]) + (0.2989 * h_pin[j + 2]);
    pos++;
  }
}

void arrToImg(unsigned char *pout, int column, int row) {
  Mat img(row, column, CV_8UC1, pout);

  namedWindow("Output", WINDOW_AUTOSIZE );
  imshow("Output", img);
  waitKey(0);
}


int main(int argc, char** argv) {
  int column, row;
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

  if(!image.data) {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
  imshow("Display window", image);                   // Show our image inside it.
  waitKey(0);

  row = image.rows;
  column = image.cols;
  size_t size = (column * row) * sizeof(unsigned char);

  unsigned char *imgSec = (unsigned char*) malloc(size);
  unsigned char *imgPar = (unsigned char*) malloc(size);
  unsigned char *img = image.data;

  //showVector(img, column, row);
  //showVector(img2, column, row);
  pictureSeq(img, imgSec, row, column);
  //pictureCuda(img, img2, row, column);
  //showVector(img2, column, row);
  arrToImg(imgSec, column, row);

  free(imgSec);
  free(imgPar);

  return 0;
}
