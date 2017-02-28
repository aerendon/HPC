#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define col 500000

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

void vecAdd(int a[col], int b[col], int sol[col], int init, int size) {
  for (int i = init; i < size; i++)
    sol[i] = a[i] + b[i];
}

int check(int a[col], int b[col]) {
  for (int i = 0; i < col; i++)
    if (a[i] != b[i])
      return 0;
  return 1;
}


int main() {
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int a[col], b[col], sol[col], sol_mpi[col];
  double start = 0, time_seq = 0, time_omp = 0;
  double t1, t2;

  initVec(a, b, sol, sol_mpi);

  if (world_rank == 0) {
    t1 = MPI_Wtime();
    vecAdd(a, b, sol, 0, col);
    t2 = MPI_Wtime();
    printf( "Secuential time is %f\n", t2 - t1 );
    
    t1 = MPI_Wtime();
    vecAdd(a, b, sol_mpi, 0, col / 2); 
    MPI_Recv(sol_mpi + col/2, col - (col / 2), MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = MPI_Wtime();
    printf( "MPI time is %f\n", t2 - t1 );
    //showVector(sol_mpi);
    if(check(sol, sol_mpi)) printf("Correct\n");
    else printf(":( \n");
    
  } else if (world_rank == 1) {
    vecAdd(a, b, sol_mpi, col / 2, col);
    MPI_Send(sol_mpi + col/2, col - (col / 2), MPI_INT, 0, 1, MPI_COMM_WORLD);
    //showVector(sol_mpi);
  } 

  MPI_Finalize();
  return 0;
}

