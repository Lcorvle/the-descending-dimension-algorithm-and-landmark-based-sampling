#pragma once
#include <ctime>
#include <stdlib.h>

#include "quadtree.h"

class Tool
{

public:
	Tool(int oldNumber) {
		timeTotal = 0;
		timeEdge = 0;
		timeNonEdge = 0;
		timeInitTree = 0;
		oldNum = oldNumber;
	};
	int oldNum;
	clock_t timeInitTree, timeEdge, timeNonEdge, timeTotal;
	void computeGradient(int* inpRowP, int* inpColP, double* inpValP, double* Y, int N, int D, double* dC, double theta);
	double evaluateError(int* rowP, int* colP, double* valP, double* Y, int N, double theta);
	void zeroMean(double* X, int N, int D);
	void incrementalZeroMeanY(double* Y, int N, int D, double* oldY);
	double randn();
};

int compare(const void *a, const void *b);
double sign(double x);
inline void swap(int &a, int &b) {
	int t = a;
	a = b;
	b = t;
}
inline int mid(int a, int b, int c) {
	if (a > b) {
		swap(a, b);
	}
	if (b > c) {
		swap(b, c);
	}
	if (a > b) {
		swap(a, b);
	}
	return b;
}

void QSort(int A[], int B[], int l, int r);