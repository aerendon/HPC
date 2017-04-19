#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace std;

void showImage(gpu::GpuMat& image, Mat h_img_sobel) {
  namedWindow("Display window", WINDOW_AUTOSIZE);
  imshow("Input", Mat(image));
  waitKey(0);
  imshow("Sobel", h_img_sobel);
  waitKey(0);
}

void sobel(gpu::GpuMat& image) {
  gpu::GpuMat img_sobel_x, img_sobel_y, img_sobel;

  gpu::Sobel(image, img_sobel_x, CV_32F, 1, 0);
  gpu::Sobel(image, img_sobel_y, CV_32F, 0, 1);

  gpu::magnitude(img_sobel_x, img_sobel_y, img_sobel);

  Mat h_img_sobel(img_sobel);
  convertScaleAbs(h_img_sobel, h_img_sobel);
  showImage(image, h_img_sobel);
}

int main(int argc, char** argv) {
  Mat h_image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if (!h_image.data) {
    printf("Could not open or find the image");
    return -1;
  }

  gpu::GpuMat d_image(h_image);
  clock_t start = clock();
  sobel(d_image);
  clock_t end = clock();
  double gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("%lf\n", gpu_time_used);

  return 0;
}
