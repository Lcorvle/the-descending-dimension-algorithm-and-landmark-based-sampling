/*
*  tsne.h
*  Header file for t-SNE.
*
*  Created by Laurens van der Maaten.
*  Copyright 2012, Delft University of Technology. All rights reserved.
*
*  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
*  Incremental version by Shouxing Xiang, 2017. xsx1996@163.com
*/


#pragma once

#include <omp.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "..\tool\vptree.h"
#include "..\tool\datapoint.h"
#include "..\tool\tool.h"


class TSNE
{
public:
	void run(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int numThreads, int maxIter, int randomState, int oldNum, int _neighborFlag,
		int* _neighborFlags, int* _neighborRows, int* _neighborOldRows, int* _neighborCols, int* _neighborOldCols, double* _neighborValues, double* _neighborOldValues);
	
private:
	int numThreads, oldNum, neighborFlag;
	int *neighborFlags, *neighborRows, *neighborOldRows, *neighborCols, *neighborOldCols;
	double *neighborValues, *neighborOldValues;
	Tool* tool;
	void computeGaussianPerplexity(double* X, int N, int D, int** _rowP, int** _colP, double** _valP, double perplexity, int K);
	void symmetrizeMatrix(int** rowP, int** colP, double** valP, int N);
};