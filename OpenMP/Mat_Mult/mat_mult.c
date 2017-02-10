#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define rowA 10
#define colA 10
#define colB 10

void initMatrix(int a[rowA][colA], int b[colA][colB], int sol[rowA][colB], int sol_Open[rowA][colB]) {
	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colA; j++)
			a[i][j] = i * 2;

	for (int i = 0; i < colA; i++)
		for (int j = 0; j < colB; j++)
			b[i][j] = i * 2;

	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colB; j++)
			sol[i][j] = 0;

	for (int i = 0; i < rowA; i++)
		for (int j = 0; j < colB; j++)
			sol_Open[i][j] = 0;
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
	// showMatrix(a, b);
	showSol(sol);

	return 0;
}
