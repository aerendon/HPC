#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define rowA 10
#define colA 5
#define colB 5

void initMatrix(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB], int sol_Open[rowA][colB]) {
	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colA; j++)
			a[i][j] = (i + 1) * 2;

	for (int i = 0; i < colA; i++)
		for (int j = 0; j < colB; j++)
			b[i][j] = (i + 1) * 2;

	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colB; j++)
			sol[i][j] = sol_Open[i][j] = 0;
}

void showMatrix(int a[rowA][colA], int b[colA][colB]) {
	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colA; j++)
			printf("%d  ", a[i][j]);
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < colA; i++) {
		for (int j = 0; j < colB; j++)
			printf("%d  ", b[i][j]);
		printf("\n");
	}
}

void multMatrix(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB], int initRowA, int finRowA) {
	for (int i = 0; i < rowA; i++)
		for (int j = initRowA; j < finRowA; j++)
			for (int k = 0; k < colA; k++)
				sol[i][j] += a[i][k] *  b[k][j];
}

int check(int a[rowA][colB], int b[rowA][colB]) {
	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colB; j++) {
			if (a[i][j] != b[i][j])
				return 0;
		}
	return 1;
}

void showSol(int sol[rowA][colB]) {
	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colB; j++)
			printf("%d  ",sol[i][j]);
		printf("\n");
	}
	printf("\n");
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

	int a[rowA][colA], b[colA][colB], sol[rowA][colB], sol_mpi[rowA][colB];
	double start = 0, time_seq = 0, time_omp = 0;
  double t1, t2;

	initMatrix(a, b, sol, sol_mpi);

  if (world_rank == 0) {
    t1 = MPI_Wtime();
    multMatrix(a, b, sol, 0, colB);
    // MPI_Recv(sol_mpi + col/2, col - (col / 2), MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = MPI_Wtime();
    // showSol(sol);
    printf( "Secuential time is %f\n", t2 - t1 );

    t1 = MPI_Wtime();
    multMatrix(a, b, sol_mpi, 0, colB / 2);
    showSol(sol_mpi);
    // MPI_Recv(sol_mpi + col/2, col - (col / 2), MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < colB; i++) {
      MPI_Recv(sol_mpi[i] + (rowA / 2), rowA - (rowA / 2), MPI_INT, 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    t2 = MPI_Wtime();

    printf( "MPI time is %f\n", t2 - t1 );
    showSol(sol_mpi);

    if(check(sol, sol_mpi)) printf("Correct\n");
    else printf(":( \n");

  } else if (world_rank == 1) {
    multMatrix(a, b, sol_mpi, colB / 2, colB);
    showSol(sol_mpi);

    for (int i = 0; i < colB; i++) {
      MPI_Send(sol_mpi[i] + (rowA / 2), rowA - (rowA / 2), MPI_INT, 0, i, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
	return 0;
}
