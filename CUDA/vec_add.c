#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define col 200000

void initVec(int a[col], int b[col], int sol[col], int sol_mpi[col]) {
  for (int i = 0; i < col; i++)
    a[i] = b[i] = i * 2;

  for (int i = 0; i < col; i++)
    sol[i] = sol_mpi[i] = 0;
}

void showVector(int a[col]) {
  for (int i = 0; i < col; i++)
    printf("%d  ", a[i]);
  printf("\n");
}

void vecAdd(int a[col], int b[col], int sol[col]) {
  for (int i = 0; i < col; i++)
    sol[i] = a[i] + b[i];
}


int main() {
  int a[col], b[col], sol[col], sol_mpi[col];
  double start = 0, time_seq = 0, time_omp = 0;

  initVec(a, b, sol, sol_mpi);

  clock_t begin = clock();
  vecAdd(a, b, sol);
  clock_t end = clock();

  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  // if(check(sol, sol_Open)) {
  //   printf("%s %f\n", "Time Sequential: ", time_seq);
  //   printf("%s %f\n", "Time OMP: ", time_omp);
  // }
  // else printf("%s\n", "Wrong answer");
  // showVector(a);
  // showVector(b);
  // showVector(sol);
  printf("%f\n", time_spent);

  return 0;
}
