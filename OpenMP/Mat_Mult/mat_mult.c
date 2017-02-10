#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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
			sol[i][j] = sol_Open = 0;
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
	int i, j, k, rA = rowA, cA = colA, cB = colB;
	#pragma omp parallel shared(a, b, rA, cA, cB), private(i, j, k, sol)
	{
		#pragma omp for scheduled(static)
		for (int i = 0; i < rowA; i++)
			for (int j = 0; j < colA; j++)
				for (int k = 0; k < colA; k++)
					sol[i][j] += a[i][k] *  b[k][j];
	}
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

	initMatrix(a, b, sol, sol_Open);
	multMatrix(a, b, sol);
	multMatrixOpen(a, b, sol_Open);
	// showMatrix(a, b);
	// showSol(sol_Open);

	return 0;
}
