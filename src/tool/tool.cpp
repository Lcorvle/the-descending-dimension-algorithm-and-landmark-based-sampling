#include "tool.h"

static const int QT_NO_DIMS = 2;

// Compute gradient of the cost function (using Barnes-Hut algorithm)
void Tool::computeGradient(int* inpRowP, int* inpColP, double* inpValP, double* Y, int N, int D, double* dC, double theta)
{
	timeInitTree -= clock();

	// Construct quadtree on current map
	QuadTree* tree = new QuadTree(Y, N);
	timeInitTree += clock();

	// Compute all terms required for t-SNE gradient
	double sumQ = .0;
	double* posF = (double*)calloc(N * D, sizeof(double));
	double* negF = (double*)calloc(N * D, sizeof(double));
	if (posF == NULL || negF == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	timeEdge -= clock();
	tree->computeEdgeForces(inpRowP, inpColP, inpValP, N, posF, oldNum);
	timeEdge += clock();

	timeNonEdge -= clock();
#pragma omp parallel for reduction(+:sumQ)
	for (int n = 0; n < N; n++) {
		double buff[QT_NO_DIMS];
		double thisQ = .0;
		tree->computeNonEdgeForces(n, theta, negF + n * D, &thisQ, &buff[0]);
		sumQ += thisQ;
	}
	timeNonEdge += clock();

	timeTotal -= clock();

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = posF[i] - (negF[i] / sumQ);
	}
	timeTotal += clock();
	free(posF);
	free(negF);
	delete tree;
}

// Evaluate cost function (approximately)
double Tool::evaluateError(int* rowP, int* colP, double* valP, double* Y, int N, double theta)
{

	// Get estimate of normalization term
	//const int QT_NO_DIMS = 2;
	QuadTree* tree = new QuadTree(Y, N);
	double buff[QT_NO_DIMS] = { .0, .0 };
	double sumQ = .0;
	for (int n = 0; n < N; n++) {
		double buff1[QT_NO_DIMS];
		tree->computeNonEdgeForces(n, theta, buff, &sumQ, &buff1[0]);
	}

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * QT_NO_DIMS;
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {
			Q = .0;
			ind2 = colP[i] * QT_NO_DIMS;
			for (int d = 0; d < QT_NO_DIMS; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < QT_NO_DIMS; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < QT_NO_DIMS; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sumQ;
			C += valP[i] * log((valP[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}
	return C;
}

// Makes data zero-mean
void Tool::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[n * D + d];
		}
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[n * D + d] -= mean[d];
		}
	}
	free(mean); mean = NULL;
}

// Makes Y zero-mean, and keep relative oldY at the same time
void Tool::incrementalZeroMeanY(double* Y, int N, int D, double* oldY) {
    // Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += Y[n * D + d];
		}
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			Y[n * D + d] -= mean[d];
		}
	}
	for (int n = 0; n < oldNum; n++) {
		for (int d = 0; d < D; d++) {
			oldY[n * D + d] -= mean[d];
		}
	}
	free(mean); mean = NULL;
}

// Generates a Gaussian random number
double Tool::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

int compare(const void *a, const void *b) {
	int *pa = (int*)a;
	int *pb = (int*)b;
	return (*pa) - (*pb);  //´ÓÐ¡µ½´óÅÅÐò
}

double sign(double x) { 
	return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); 
}

void QSort(int A[], int B[], int l, int r)
{
	if (l >= r) return;
	int i = l, j = r, x = mid(A[l], A[(l + r) >> 1], A[r]);
	while (true)
	{
		while (A[i] < x) ++i;
		while (A[j] > x) --j;
		if (i > j) break;
		swap(A[i], A[j]);
		swap(B[i], B[j]);
		++i;
		--j;
	}
	QSort(A, B, l, j);
	QSort(A, B, i, r);
}