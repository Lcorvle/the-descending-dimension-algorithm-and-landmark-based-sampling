/*
*	hsne.h
*	Header file for h-SNE
*
*	Created by Shouxing Xiang. 2017. xsx1996@163.com
*/

#pragma once
#include <omp.h>
#include <iostream>
#include <fstream>

#include "..\tool\vptree.h"
#include "..\tool\datapoint.h"
#include "..\tool\tool.h"

class HSNE
{
public:
	void run(double* Y, int N, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int _oldNum, int* rowP, int* colP, double* valP);

private:
	int numThreads, oldNum;
	Tool* tool;
	void symmetrizeMatrix(int** rowP, int** colP, double** valP, int N);
};