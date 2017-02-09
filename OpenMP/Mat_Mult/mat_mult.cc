#include <bits/stdc++.h>
#include <omp.h>

#define N 10

using namespace std;

int main() {
	int a[N][N], b[N][N], sol[N][N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = b[i][j] = i * 2;
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << b[i][j] << "  ";
		}
		cout << endl;
	}

	return 0;
}