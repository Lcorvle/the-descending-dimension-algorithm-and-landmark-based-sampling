#include "tsne\tsne.h"
#include "landMarkBasedSampling\LevelList.h"
#include "hsne\hsne.h"

extern "C"
{
	extern void hsneRunDouble(double* Y, int N, int noDims, 
		double perplexity, double theta, int _numThreads, int maxIter, int randomState, 
		int* rowP, int* colP, double* valP) {
		printf("Performing h-SNE using %d cores.\n", _numThreads);
		HSNE hsne;
		hsne.run(Y, N, noDims, perplexity, theta, _numThreads, maxIter, randomState, 0, rowP, colP, valP);
		return;
	}

	extern void incrementalHsneRunDouble(double* Y, int N, int noDims, int oldNum,
		double perplexity, double theta, int _numThreads, int maxIter, int randomState,
		int* rowP, int* colP, double* valP) {
		printf("Performing incremental h-SNE using %d cores.\n", _numThreads);
		HSNE hsne;
		hsne.run(Y, N, noDims, perplexity, theta, _numThreads, maxIter, randomState, oldNum, rowP, colP, valP);
		return;
	}

	extern void getProbabilityDistribution(int threads, int* levelIDs, int* levelSizes,
		int* indexSet, int size,
		int* transitionSizes, int* transitionIndexes, double *transitionValues,
		int** _rowP, int** _colP, double** _valP) {
		printf("Computing probability distribution using %d cores.\n", threads);
		omp_set_num_threads(threads);
		getProbabilityDistribution(levelIDs, levelSizes, indexSet, size, transitionSizes, transitionIndexes, transitionValues, _rowP, _colP, _valP);
	}

	extern void tsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta,
		int _numThreads, int maxIter, int randomState, int flag,
		int* _flags, int* _rows, int* _oldRows, int* _cols, int* _oldCols, double* _values, double* _oldValues)
	{
		printf("Performing t-SNE using %d cores.\n", _numThreads);
		TSNE tsne;
		tsne.run(X, N, D, Y, noDims, perplexity, theta, _numThreads, maxIter, randomState, 0, flag, _flags, _rows, _oldRows, _cols, _oldCols, _values, _oldValues);
	}

	extern void incrementalTsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity,
		double theta, int _numThreads, int maxIter, int randomState, int oldNum, int flag,
		int* _flags, int* _rows, int* _oldRows, int* _cols, int* _oldCols, double* _values, double* _oldValues)
	{
		printf("Performing incremental t-SNE using %d cores.\n", _numThreads);
		TSNE tsne;
		tsne.run(X, N, D, Y, noDims, perplexity, theta, _numThreads, maxIter, randomState, oldNum, flag, _flags, _rows, _oldRows, _cols, _oldCols, _values, _oldValues);
	}

	extern void landMarkBasedSampling(int threads, double perplexity, int randWalksNum, int randWalksLength, int randWalksThrehold,
		int endSize, double* data, int rowNum, int dim, int** _levelSizes, int** _indexes,
		int** _levelInfluenceSizes, int** _pointInfluenceSizes, int** _influenceIndexes,
		double** _influenceValues, int** _nearestNeighborIndexes, double** _nearestNeighborDistances,
		int** _transitionSizes, int** _transitionIndexes, double **_transitionValues)
	{
		printf("Performing landmark sampling using %d cores.\n", threads);
		omp_set_num_threads(threads);
		LevelList levelList = LevelList(perplexity, randWalksNum, randWalksLength, randWalksThrehold, endSize);
		printf("Initing data...\n");
		clock_t t = clock();
		levelList.initData(data, rowNum, dim, _nearestNeighborIndexes, _nearestNeighborDistances);
		printf("Init data cost %f\n", float(clock() - t));
		printf("Computing level list...\n");
		t = clock();
		levelList.computeLevelList(_levelSizes, _indexes, _levelInfluenceSizes, _pointInfluenceSizes, _influenceIndexes, _influenceValues);
		printf("Compute level list cost %f\n", float(clock() - t));
		printf("Computing transition matrix for hsne...\n");
		t = clock();
		levelList.getTransitionMatrix(_transitionSizes, _transitionIndexes, _transitionValues);
		printf("Compute transition matrix for hsne cost %f\n", float(clock() - t));
	}

	extern void getNearestNeighbors(int threads, int rowNum, int* oldIndexes, int oldNum, int* newIndexes, int newNum,
		int* nearestNeighborIndexes, double* nearestNeighborDistances, int** _flags, int** _rows, int** _oldRows, int** _cols, int** _oldCols, double** _values, double** _oldValues)
	{
		printf("Get nearest neighbors.\n");
		omp_set_num_threads(threads);
		getNearestNeighbors(rowNum, oldIndexes, oldNum, newIndexes, newNum, nearestNeighborIndexes, nearestNeighborDistances, _flags, _rows, _oldRows, _cols, _oldCols, _values, _oldValues);
	}

	extern void getInfluenceIndexes(int threads, int* levelSizes, int* indexes, int* levelInfluenceSizes,
		int* pointInfluenceSizes, int* influenceIndexes, double* influenceValues, int *indexSet, int size,
		double minInfluenceValue, int** _resultSet)
	{
		printf("Get influenced indexes.\n");
		omp_set_num_threads(threads);
		getInfluenceIndexes(levelSizes, indexes, levelInfluenceSizes, pointInfluenceSizes, influenceIndexes, influenceValues, indexSet, size, minInfluenceValue, _resultSet);
	}
}