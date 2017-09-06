#pragma once

#include <vector>
#include <ctime>
#include <time.h>
#include "Level.h"
#include "..\tool\vptree.h"
#include "..\tool\datapoint.h"
#include "..\tool\tool.h"
#include <iostream>
#include <set>
#include <stdio.h>

class LevelList {
	Level *head, *tail;
	double perplexity;
	int walksNum, walksLength, threhold, endLevelSize, length;

public:
	LevelList(double perp, int randWalksNum, int randWalksLength, int randWalksThrehold, int endSize) {
		perplexity = perp;
		walksNum = randWalksNum;
		walksLength = randWalksLength;
		threhold = randWalksThrehold;
		endLevelSize = endSize;
		head = NULL;
		tail = NULL;
		length = 0;
		srand(unsigned(time(0)));
	};
	void initData(double* data, int rowNum, int dim, int** _nearestNeighborIndexes,
		double** _nearestNeighborDistances);
	void computeLevelList(int** _levelSizes, int** _indexes, int** _levelInfluenceSizes,
		int** _pointInfluenceSizes, int** _influenceIndexes, double** _influenceValues);
	void getTransitionMatrix(int** _transitionSizes, int** _transitionIndexes, double **_transitionValues);
private:
	vector<int> computeNextLevelIndexes();
	SparseMatrix<double> computeNextLevelInfluences(vector<int> indexes);
	int computeNextLevel();
	void standardizeData(double* data, int rowNum, int dim);
};

void getInfluenceIndexes(int* levelSizes, int* indexes, int* levelInfluenceSizes, int* pointInfluenceSizes,
	int* influenceIndexes, double* influenceValues, int *indexSet, int size, double minInfluenceValue,
	int** _resultSet);

void getNearestNeighbors(int rowNum, int* oldIndexes, int oldNum, int* newIndexes, int newNum,
	int* nearestNeighborIndexes, double* nearestNeighborDistances, 
	int** _flags, int** _rows, int** _oldRows, int** _cols, int** _oldCols, double** _values, double** _oldValues);

void getProbabilityDistribution(int* levelIDs, int* levelSizes,
	int* indexSet, int size,
	int* transitionSizes, int* transitionIndexes, double *transitionValues,
	int** _rowP, int** _colP, double** _valP);