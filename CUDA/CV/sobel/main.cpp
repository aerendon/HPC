#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void showVector(unsigned char *mat, int row, int column) {
  for (int i = 0; i < 6; i++)
    printf("%u  ", mat[i]);
  printf("\n");
}

Mat sobelOpen(Mat image) {
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat grad;

  int ddepth = CV_16S;
  int scale = 1;
  int delta = 0;
  /// Gradient X
  Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  convertScaleAbs( grad_x, abs_grad_x );
  convertScaleAbs( grad_y, abs_grad_y );

  namedWindow("sobel", WINDOW_AUTOSIZE);
  imshow("Sobel Gx", abs_grad_x);
  waitKey(0);
  imshow("Sobel Gy", abs_grad_y);
  waitKey(0);

  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  return grad;
}

void mult(int *mat1, int *mat2, int row, int column) {
  int sol = 0;

  for(int k = 0; k < column; k++) {
    for(int i = k * row; i < row * column; i++) {
      for(int j = 0; j < row; j++) {
        sol = (mat1[i] * mat2[j]) + (mat1[i + 1] * mat2[row + j]) + (mat1[i + 2] * mat2[(row * 2) + j]);
      }
      i += 2;
      cout << sol << endl;
    }
  }
}

void sobelSeq(unsigned char *image, int *Gx, int *Gy, int row, int column) {
  for(int i = 0; i < 3; i++) {
    // cout << " " << int(Gx[i]);
  }
}

Mat arrToMat(unsigned char *pout, int column, int row) {
  Mat img(row, column, CV_8UC1, pout);
  return img;
}

int main(int argc, char** argv) {
  int column, row;
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

  if(!image.data) {
      cout <<  "Could not open or find the image" << endl ;
      return -1;
  }

  namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
  imshow("Display window", image);                   // Show our image inside it.
  waitKey(0);

  row = image.rows;
  column = image.cols;
  size_t size = column * row * sizeof(unsigned char);

  unsigned char *img = image.data;
  int Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1},
      Gy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  // imshow("Sobel", sobelOpen(image));
  // waitKey(0);

  sobelOpen(image);

  // sobelSeq(img, Gx, Gy, row, column);
  // mult(Gx, Gy, 3, 3);

  return 0;
}
