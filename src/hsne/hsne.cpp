/*
*	hsne.cpp
*	Implementation of hsne and incremental version.
*	Created by Shouxing Xiang. 2017. xsx1996@163.com
*/

#include "hsne.h"

using namespace std;

//Perform h-SNE
// Y -- array to fill with the result of size [N, noDims]
// noDims -- target dimentionality
void HSNE::run(double* Y, int N, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int _oldNum, int* rowP, int* colP, double* valP) {
	tool = new Tool(_oldNum);
	oldNum = _oldNum;
	numThreads = _numThreads;

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

	symmetrizeMatrix(&rowP, &colP, &valP, N);
	double sumP = .0;
	for (int i = 0; i < rowP[N]; i++) {
		sumP += valP[i];
	}
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] /= sumP;
	}






	// Step 2
	// Lie about the P-values
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] *= 12.0;
	}



	// Initialize solution

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
				double scale = sumF / sumD * 0.95;
				if (iter > stopLyingIter) {
				    for (int j = 0;j < noDims; j++) {
                        int i = n * noDims + j;
                        Y[i] += (oldY[i] - Y[i]) * scale;
                    }
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
	free(dY); dY = NULL;
	free(uY); uY = NULL;
	free(gains); gains = NULL;
	free(oldY); oldY = NULL;
	delete tool;
	printf("Fitting performed in %4.2f seconds.\n", totalTime);

}

void HSNE::symmetrizeMatrix(int** _rowP, int** _colP, double** _valP, int N) {
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