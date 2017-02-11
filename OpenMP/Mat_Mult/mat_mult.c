#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define rowA 700
#define colA 700
#define colB 700

void initMatrix(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB], int sol_Open[rowA][colB]) {
	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colA; j++)
			a[i][j] = i * 2;

	for (int i = 0; i < colA; i++)
		for (int j = 0; j < colB; j++)
			b[i][j] = i * 2;

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

void multMatrix(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB]) {
	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colA; j++)
			for (int k = 0; k < colA; k++)
				sol[i][j] += a[i][k] *  b[k][j];
}

void multMatrixOpen(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB]) {
	int i, j, k, rA = rowA, cA = colA, cB = colB, aux;
	#pragma omp parallel shared(a, b, rA, cA, cB, sol), private(i, j, k, aux)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < rowA; i++) {
			for (int j = 0; j < colA; j++) {
				aux = 0;
				for (int k = 0; k < colA; k++) {
					aux += a[i][k] *  b[k][j];
				}
				sol[i][j] = aux;
			}
		}
	}
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
	int a[rowA][colA], b[colA][colB], sol[rowA][colB], sol_Open[rowA][colB];
	double start = 0, time_seq = 0, time_omp = 0;

	initMatrix(a, b, sol, sol_Open);

	start = omp_get_wtime();
	multMatrix(a, b, sol);
	time_seq = omp_get_wtime() - start;
	start = omp_get_wtime();
	multMatrixOpen(a, b, sol_Open);
	time_omp = omp_get_wtime() - start;

	if(check(sol, sol_Open)) {
		printf("%s %f\n", "Time Sequential: ", time_seq);
		printf("%s %f\n", "Time OMP: ", time_omp);
	}
	else printf("%s\n", "Wrong answer");
	// showMatrix(a, b);
	// showSol(sol);
	// showSol(sol_Open);

	return 0;
}
