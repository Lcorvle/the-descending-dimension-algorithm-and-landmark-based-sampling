/*
*  tsne.cpp
*  Implementation of both standard and Barnes-Hut-SNE, and also incremental version.
*
*  Created by Laurens van der Maaten.
*  Copyright 2012, Delft University of Technology. All rights reserved.
*
*  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
*
*  Incremental version by Shouxing Xiang, 2017. xsx1996@163.com
*/

#include "tsne.h"

using namespace std;


// Perform t-SNE
// X -- double matrix of size [N, D]
// D -- input dimentionality
// Y -- array to fill with the result of size [N, noDims]
// noDims -- target dimentionality
void TSNE::run(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int _oldNum, int _neighborFlag,
	int* _neighborFlags, int* _neighborRows, int* _neighborOldRows, int* _neighborCols, int* _neighborOldCols, double* _neighborValues, double* _neighborOldValues) {
    if (N - 1 < 3 * perplexity) {
		printf("Perplexity too large for the number of data points!\n");
		exit(1);
	}
	tool = new Tool(_oldNum);
	oldNum = _oldNum;
	numThreads = _numThreads;
	neighborFlag = _neighborFlag;
	neighborFlags = _neighborFlags;
	neighborRows = _neighborRows;
	neighborOldRows = _neighborOldRows;
	neighborCols = _neighborCols;
	neighborOldCols = _neighborOldCols;
	neighborValues = _neighborValues;
	neighborOldValues = _neighborOldValues;

	omp_set_num_threads(numThreads);

	printf("Using noDims = %d, perplexity = %f, and theta = %f\n", noDims, perplexity, theta);

	// Set learning parameters
	float totalTime = .0;
	time_t start, end;
	clock_t timeStart, timeEnd, timeComputeGradient, timeUpdateY, timePrintProcess;
	int stopLyingIter = 250, momSwitchIter = 250;
	double momentum = .5, finalMomentum = .8;
	double eta = 200.0;

	// Allocate some memory
	double* dY = (double*)calloc(N * noDims, sizeof(double));
	double* uY = (double*)calloc(N * noDims, sizeof(double));
	double* gains = (double*)malloc(N * noDims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * noDims; i++) {
		gains[i] = 1.0;
	}

	// Normalize input data (to prevent numerical problems)
	printf("Computing input similarities...\n");
	start = time(0);
	tool->zeroMean(X, N, D);
	double maxX = .0;
	for (int i = 0; i < N * D; i++) {
		if (X[i] > maxX) maxX = X[i];
	}
	for (int i = 0; i < N * D; i++) {
		X[i] /= maxX;
	}

	// Compute input similarities
	int* rowP; int* colP; double* valP;

	// Compute asymmetric pairwise input similarities
	timeStart = clock();
	computeGaussianPerplexity(X, N, D, &rowP, &colP, &valP, perplexity, (int)(3 * perplexity));
	timeEnd = clock();
	cout << "computing input similarities: " << (timeEnd - timeStart) << endl;

	// Symmetrize input similarities
	symmetrizeMatrix(&rowP, &colP, &valP, N);
	double sumP = .0;
	for (int i = 0; i < rowP[N]; i++) {
		sumP += valP[i];
	}
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] /= sumP;
	}

	end = time(0);
	printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start), (double)rowP[N] / ((double)N * (double)N));

	// Step 2
	// Lie about the P-values
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] *= 12.0;
	}



	// Initialize solution
	// Build ball tree on old data set and update new points position

	if (oldNum > 0) {
		VpTree<DataPoint, euclideanDistance>* oldTree = new VpTree<DataPoint, euclideanDistance>();
		vector<DataPoint> oldObjX(oldNum, DataPoint(D, -1, X));
		if (neighborFlag == 0 || neighborFlags[1] < N - oldNum) {
			timeStart = clock();
#pragma omp parallel for
			for (int n = 0; n < oldNum; n++) {
				oldObjX[n] = DataPoint(D, n, X + n * D);
			}
			oldTree->create(oldObjX);
			timeEnd = clock();
			cout << "Build ball tree on old data set: " << (timeEnd - timeStart) << endl;
		}

		timeStart = clock();
		//ofstream out("outOld.txt", ios::out);
		//out << oldNum << "=" << N << endl;
#pragma omp parallel for
		for (int i = oldNum * noDims; i < N * noDims; i += noDims) {
			// Find nearest neighbors
			vector<int> neighborIndexes;
			vector<double> distances;
			if (neighborFlag > 0) {
				int left = neighborOldRows[i / noDims - oldNum],
					right = neighborOldRows[i / noDims - oldNum + 1];
				if (left < right) {
					for (int j = left; j < right; j++) {
					    if (neighborIndexes.size() == 5) {
					        break;
					    }
						neighborIndexes.push_back(neighborOldCols[j]);
						distances.push_back(neighborOldValues[j]);
					}
				}
				else {
					vector<DataPoint> indices;
					int n = i / noDims;
					oldTree->search(DataPoint(D, n, X + n * D), 1, &indices, &distances);
					neighborIndexes.push_back(indices[0].index());
				}
			}
			else {
				vector<DataPoint> indices;
				int n = i / noDims, K = oldNum < 300 ? 1 : min(5, (oldNum / 300));
				oldTree->search(DataPoint(D, n, X + n * D), K, &indices, &distances);
				for (int j = 0; j < K; j++) {
					neighborIndexes.push_back(indices[j].index());
				}
			}
			int len = neighborIndexes.size();
			//out << i / noDims << " " << len << " ";
			// for (int j = 0;j < len;j++){
			//     out << neighborIndexes[j] << " ";
			// }
			// out << endl;
			for (int j = 0; j < noDims; j++) {
				Y[i + j] = .0;
			}
			double totalWeight = .0;
			for (int k = 0; k < len; k++) {
				for (int j = 0; j < noDims; j++) {
					Y[i + j] += Y[neighborIndexes[k] * noDims + j] / distances[k];
				}
				totalWeight += 1 / distances[k];
			}
			for (int j = 0; j < noDims; j++) {
				Y[i + j] /= totalWeight;
			}
		}
		//out.close();
		delete oldTree;
		oldObjX.clear();
		timeEnd = clock();
		cout << "Init new data points' position: " << (timeEnd - timeStart) << endl;
	}
	else {
		timeStart = clock();
        if (randomState != -1) {
            srand(randomState);
        }
        for (int i = oldNum * noDims; i < N * noDims; i++) {
            Y[i] = tool->randn() * .0001;
        }
		timeEnd = clock();
		cout << "Init all data points' position: " << (timeEnd - timeStart) << endl;
	}

    double* oldY = (double*)malloc((oldNum * noDims) * sizeof(double));
    for (int i = 0; i < oldNum * noDims;i++) {
        oldY[i] = Y[i];
    }

    if (randomState != -1) {
        srand(randomState);
    }
    for (int i = oldNum * noDims; i < N * noDims; i++) {
        Y[i] = tool->randn() * .0001;
    }


	// Perform main training loop
	start = time(0);
	timeStart = clock();
	timeComputeGradient = 0;
	timeUpdateY = 0;
	timePrintProcess = 0;
	ofstream file("iter.txt");
	file << N << " " << maxIter / 4 + 1 << endl;
	for (int iter = 0; iter < maxIter; iter++) {
        if (iter % 4 == 0){
            for (int i = 0;i < N;i++){
                for(int j = 0;j < noDims;j++){
                    file << Y[i * noDims + j];
                    if (j != noDims - 1){
                        file << " ";
                    }
                }
                file << endl;
            }
        }


		// Compute approximate gradient
		timeComputeGradient = timeComputeGradient - clock();
		tool->computeGradient(rowP, colP, valP, Y, N, noDims, dY, theta);
		timeComputeGradient = timeComputeGradient + clock();
		timeUpdateY = timeUpdateY - clock();
#pragma omp parallel for
		for (int n = 0; n < N; n++) {
			double sumF = .0, sumD = .0;
			for (int j = 0;j < noDims; j++) {
		        // Update gains
		        int i = n * noDims + j;
                gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
                if (gains[i] < .01) {
                    gains[i] = .01;
                }

                // Perform gradient update (with momentum and gains)
                uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
                if (n < oldNum) {
                	sumF += uY[i] * uY[i];
					sumD += (oldY[i] - Y[i]) * (oldY[i] - Y[i]);
                }
		    }
			if (n < oldNum && sumD > .1) {
				sumF = sqrt(sumF);
				sumD = sqrt(sumD);
				double scale = sumF / sumD;
				if (iter < momSwitchIter) {
					scale *= 0;
				}
				else {
					scale *= 1.3;
				}
				for (int j = 0;j < noDims; j++) {
					int i = n * noDims + j;
					Y[i] += (oldY[i] - Y[i]) * scale;
				}
			}
			for (int j = 0;j < noDims; j++) {
				int i = n * noDims + j;
                Y[i] += uY[i];
		    }
			    
		}

		// Make solution zero-mean
		if (oldNum > 0) {
		    tool->incrementalZeroMeanY(Y, N, noDims, oldY);
		}
		else {
		    tool->zeroMean(Y, N, noDims);
		}

		timeUpdateY = timeUpdateY + clock();

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stopLyingIter) {
			for (int i = 0; i < rowP[N]; i++) {
				valP[i] /= 12.0;
			}
		}
		if (iter == momSwitchIter) {
			momentum = finalMomentum;
		}
		timePrintProcess = timePrintProcess - clock();

		// Print out progress
		if ((iter > 0 && iter % 50 == 0) || (iter == maxIter - 1)) {
			end = time(0);
			double C = .0;

			C = tool->evaluateError(rowP, colP, valP, Y, N, theta);  // doing approximate computation here!

			if (iter == 0)
				printf("Iteration %d: error is %f\n", iter + 1, C);
			else {
				totalTime += (float)(end - start);
				printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start));
			}
			start = time(0);
		}
		timePrintProcess = timePrintProcess + clock();
	}
	for (int i = 0;i < N;i++){
        for(int j = 0;j < noDims;j++){
            file << Y[i * noDims + j];
            if (j != noDims - 1){
                file << " ";
            }
        }
        file << endl;
    }
    file.close();
	timeEnd = clock();
	// Print out time cost in process
	cout << "Main train loop: " << (timeEnd - timeStart) << endl;
	cout << "Main train loop, compute gradient: " << timeComputeGradient << endl;
	cout << "Main train loop, update Y: " << timeUpdateY << endl;
	cout << "Main train loop, print process: " << timePrintProcess << endl;
	cout << "Compute gradient, init tree: " << tool->timeInitTree << endl;
	cout << "Compute gradient, edge: " << tool->timeEdge << endl;
	cout << "Compute gradient, nonedge: " << tool->timeNonEdge << endl;
	cout << "Compute gradient, total: " << tool->timeTotal << endl;
	end = time(0); totalTime += (float)(end - start);

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);
	free(rowP); rowP = NULL;
	free(colP); colP = NULL;
	free(valP); valP = NULL;
	free(oldY); oldY = NULL;
	delete tool;
	printf("Fitting performed in %4.2f seconds.\n", totalTime);
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, int** _rowP, int** _colP, double** _valP, double perplexity, int K) {

	if (perplexity > K) printf("Perplexity should be lower than K!\n");

	// Allocate the memory we need
	*_rowP = (int*)malloc((N + 1) * sizeof(int));
	if (*_rowP == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	int* rowP = *_rowP;
	rowP[0] = 0;
	for (int n = 0; n < N; n++) {
		if (neighborFlag > 0) {
			int left = neighborRows[n],
				right = neighborRows[n + 1];
			if (left < right) {
				rowP[n + 1] = rowP[n] + right - left;
			}
			else {
				rowP[n + 1] = rowP[n] + K / 2;
			}
		}
		else {
			rowP[n + 1] = K;
		}
	}

	*_colP = (int*)calloc(rowP[N], sizeof(int));
	*_valP = (double*)calloc(rowP[N], sizeof(double));
	if (*_colP == NULL || *_valP == NULL) { printf("Memory allocation failed!\n"); exit(1); }


	int* colP = *_colP;
	double* valP = *_valP;



	VpTree<DataPoint, euclideanDistance>* tree = new VpTree<DataPoint, euclideanDistance>();
	vector<DataPoint> objX(N, DataPoint(D, -1, X));
	if (neighborFlag == 0 || neighborFlags[0] < N) {
		// Build ball tree on data set
#pragma omp parallel for
		for (int n = 0; n < N; n++) {
			objX[n] = DataPoint(D, n, X + n * D);
		}
		tree->create(objX);
		printf("Building tree...\n");
	}

	// Loop over all points to find nearest neighbors
	int stepsCompleted = 0;
//	ofstream out("out.txt", ios::app);
//	out << "====" << oldNum << "====" << N << "====" << endl;
#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		vector<int> neighborIndexes;
		vector<double> distances;
		if (neighborFlag > 0) {
			int left = neighborRows[n],
				right = neighborRows[n + 1];
			if (left < right) {
				for (int j = left; j < right; j++) {
					neighborIndexes.push_back(neighborCols[j]);
					distances.push_back(neighborValues[j]);
				}
			}
			else {
				vector<DataPoint> indices;
				// Find nearest neighbors
				tree->search(objX[n], K / 2 + 1, &indices, &distances);
				for (int i = 0; i < K / 2; i++) {
					neighborIndexes.push_back(indices[i + 1].index());
				}
				distances.erase(distances.begin());
			}
		}
		else {
			vector<DataPoint> indices;
			// Find nearest neighbors
			tree->search(objX[n], K + 1, &indices, &distances);
			for (int i = 0; i < K; i++) {
				neighborIndexes.push_back(indices[i + 1].index());
			}
			distances.erase(distances.begin());
		}
		int neighborNumber = neighborIndexes.size();
//        for (int i = 0;i < neighborNumber;i++) {
//            out << int(distances[i]) << ",";
//        }
//        out << endl << endl;
		vector<double> curP(neighborNumber);
		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double minBeta = -DBL_MAX;
		double maxBeta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sumP;
		while (!found && iter < 200) {

			// Compute Gaussian kernel row
			for (int m = 0; m < neighborNumber; m++) {
				curP[m] = exp(-beta * distances[m]);
			}

			// Compute entropy of current row
			sumP = DBL_MIN;
			for (int m = 0; m < neighborNumber; m++) {
				sumP += curP[m];
			}
			double H = .0;
			for (int m = 0; m < neighborNumber; m++) {
				H += beta * (distances[m] * curP[m]);
			}
			H = (H / sumP) + log(sumP);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(rowP[N] / N / 3);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					minBeta = beta;
					if (maxBeta == DBL_MAX || maxBeta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + maxBeta) / 2.0;
				}
				else {
					maxBeta = beta;
					if (minBeta == -DBL_MAX || minBeta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + minBeta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (int m = 0; m < neighborNumber; m++) {
			curP[m] /= sumP;
		}
		for (int m = 0; m < neighborNumber; m++) {
			colP[rowP[n] + m] = neighborIndexes[m];
			valP[rowP[n] + m] = curP[m];
		}

		// Print progress
#pragma omp atomic
		++stepsCompleted;

		if (stepsCompleted % 10000 == 0)
		{
#pragma omp critical
			printf(" - point %d of %d\n", stepsCompleted, N);
		}
	}
	//out.close();
	// Clean up memory
	objX.clear();
	delete tree;
}


void TSNE::symmetrizeMatrix(int** _rowP, int** _colP, double** _valP, int N) {

	// Get sparse matrix
	int* rowP = *_rowP;
	int* colP = *_colP;
	double* valP = *_valP;

	// Count number of elements and row counts of symmetric matrix
	int* rowCounts = (int*)calloc(N, sizeof(int));
	if (rowCounts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {

			// Check whether element (colP[i], n) is present
			bool present = false;
			for (int m = rowP[colP[i]]; m < rowP[colP[i] + 1]; m++) {
				if (colP[m] == n) present = true;
			}
			if (present) rowCounts[n]++;
			else {
				rowCounts[n]++;
				rowCounts[colP[i]]++;
			}
		}
	}
	int noElem = 0;
	for (int n = 0; n < N; n++) noElem += rowCounts[n];

	// Allocate memory for symmetrized matrix
	int*    symRowP = (int*)malloc((N + 1) * sizeof(int));
	int*    symColP = (int*)malloc(noElem * sizeof(int));
	double* symValP = (double*)malloc(noElem * sizeof(double));
	if (symRowP == NULL || symColP == NULL || symValP == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	symRowP[0] = 0;
	for (int n = 0; n < N; n++) symRowP[n + 1] = symRowP[n] + rowCounts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {                                 // considering element(n, colP[i])

																					  // Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = rowP[colP[i]]; m < rowP[colP[i] + 1]; m++) {
				if (colP[m] == n) {
					present = true;
					if (n <= colP[i]) {                                                // make sure we do not add elements twice
						symColP[symRowP[n] + offset[n]] = colP[i];
						symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
						symValP[symRowP[n] + offset[n]] = valP[i] + valP[m];
						symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i] + valP[m];
					}
				}
			}

			// If (colP[i], n) is not present, there is no addition involved
			if (!present) {
				symColP[symRowP[n] + offset[n]] = colP[i];
				symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
				symValP[symRowP[n] + offset[n]] = valP[i];
				symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i];
			}

			// Update offsets
			if (!present || (present && n <= colP[i])) {
				offset[n]++;
				if (colP[i] != n) offset[colP[i]]++;
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < noElem; i++) symValP[i] /= 2.0;

	// Return symmetrized matrices
	free(*_rowP); *_rowP = symRowP;
	free(*_colP); *_colP = symColP;
	free(*_valP); *_valP = symValP;

	// Free up some memery
	free(offset); offset = NULL;
	free(rowCounts); rowCounts = NULL;
}