#include "LevelList.h"
using namespace std;

void LevelList::getTransitionMatrix(int** _transitionSizes, int** _transitionIndexes, double **_transitionValues) {
	*_transitionSizes = new int[length * 3 + 2];
	int* sizes = *_transitionSizes;
	sizes[0] = length * 3 + 1;
	sizes[1] = 0;
	
	Level *level = tail;
	int levelID, neighborNumber;
	SparseMatrix<double> transition;
	vector<double> vMatrix;
	
	while (level != NULL) {
		levelID = level->getID();
		levelID = length - levelID;
		sizes[levelID * 3 - 1] = level->getSize();
		neighborNumber = levelID == length ? int(3 * perplexity) : sizes[levelID * 3 - 1];
		sizes[levelID * 3] = neighborNumber;
		sizes[levelID * 3 + 1] = sizes[levelID * 3 - 2] + sizes[levelID * 3 - 1] * sizes[levelID * 3];
		transition = level->getTransitionMatrix().adjoint();
		if (levelID == length) {
			*_transitionIndexes = new int[sizes[levelID * 3 - 1] * neighborNumber];
			int *indexes = *_transitionIndexes;
			vector<double> tempMatrix;
			for (int i = 0; i < sizes[levelID * 3 - 1]; i++) {
				tempMatrix.clear();
				int j = 0;
				for (SparseMatrix<double>::InnerIterator it(transition, i); it; ++it)
				{
					tempMatrix.push_back(it.value());
					indexes[i * neighborNumber + j] = it.row();
					j++;
				}
				int gap = neighborNumber - j;
				j = 0;
				for (j = neighborNumber - 1; j >= gap; j--) {
					indexes[i * neighborNumber + j] = indexes[i * neighborNumber + j - gap];
				}
				for (j = 0; j < gap; j++) {
					indexes[i * neighborNumber + j] = 0;
					vMatrix.push_back(0);
				}
				vMatrix.insert(vMatrix.end(), tempMatrix.begin(), tempMatrix.end());
			}
		}
		else {
			for (int i = 0; i < sizes[levelID * 3 - 1]; i++) {
				int j = 0;
				for (SparseMatrix<double>::InnerIterator it(transition, i); it; ++it)
				{
					int row = it.row();
					while (j < row) {
						vMatrix.push_back(0);
						j++;
					}
					if (row == i) {
						vMatrix.push_back(0);
					}
					else {
						vMatrix.push_back(it.value());
					}
					j++;
				}
				while (j < neighborNumber) {
					vMatrix.push_back(0);
					j++;
				}
			}
		}
		
		level = level->getPre();
	}
	int matrixSize = vMatrix.size();
	*_transitionValues = new double[matrixSize];
	double *matrix = *_transitionValues;
	for (int i = 0; i < matrixSize; i++) {
		matrix[i] = vMatrix[i];
	}
}

void LevelList::standardizeData(double* data, int rowNum, int dim) {
	//compute mean and maximum, and set them to zero and one.
	int i, j, k = 0;
	double maxVal = 0, temp;
	double* mean = (double*)calloc(dim, sizeof(double));
	for (i = 0; i < rowNum; i++) {
		for (j = 0; j < dim; j++) {
			temp = data[i * dim + j];
			if (temp > maxVal) {
				maxVal = temp;
				k = j;
			}
			mean[j] += temp;
		}
	}
	for (j = 0; j < dim; j++) {
		mean[j] /= double(rowNum);
	}
	maxVal -= mean[k];
	for (i = 0; i < rowNum; i++) {
		for (j = 0; j < dim; j++) {
			data[i * dim + j] = (data[i * dim + j] - mean[j]) / maxVal;
		}
	}
	free(mean); mean = NULL;
}

void LevelList::initData(double* data, int rowNum, int dim, int** _nearestNeighborIndexes,
	double** _nearestNeighborDistances) {
	printf("begin initData\n");
	//make value between -1 to 1, and mean zero.
	standardizeData(data, rowNum, dim);

	//define variables
	head = new Level(0, rowNum, 0);
	tail = head;
	SparseMatrix<double> transition(rowNum, rowNum), weight(1, rowNum), influence(0, 0);
	vector<Eigen::Triplet<double> > tTransition, tWeight;
	vector<int> indexes;

	//compute indexes, weight and transition matrix of first level
	// Build ball tree on data set
	VpTree<DataPoint, euclideanDistance>* tree = new VpTree<DataPoint, euclideanDistance>();
	vector<DataPoint> objX(rowNum, DataPoint(dim, -1, data));
#pragma omp parallel for
	for (int n = 0; n < rowNum; n++) {
		objX[n] = DataPoint(dim, n, data + n * dim);
	}
	tree->create(objX);
	for (int i = 0; i < rowNum; i++) {
		indexes.push_back(i);
		tWeight.push_back(Eigen::Triplet<double>(0, i, 1.0));
	}
	//compute transition, weight and index of the original level
	printf("begin compute transition\n");
	int K = int(3 * perplexity);
	int *tempTranSitionIndex = new int[rowNum * K];
	double *tempTranSitionValue = new double[rowNum * K];
	int stepsCompleted = 0;
	int neighborsNumber = int(6 * perplexity) + 1 > rowNum ? rowNum : int(6 * perplexity) + 1;
	*_nearestNeighborIndexes = new int[rowNum * (neighborsNumber - 1) + 2];
	int* nearestNeighborIndexes = *_nearestNeighborIndexes;
	nearestNeighborIndexes[0] = rowNum * (neighborsNumber - 1) + 1;
	nearestNeighborIndexes[1] = neighborsNumber - 1;
	*_nearestNeighborDistances = new double[rowNum * (neighborsNumber - 1) + 1];
	double* nearestNeighborDistances = *_nearestNeighborDistances;
	nearestNeighborDistances[0] = rowNum * (neighborsNumber - 1);
#pragma omp parallel for
	for (int i = 0; i < rowNum; i++) {
		vector<double> curP(int(perplexity * 3));
		vector<DataPoint> indices;
		vector<double> distances;

		// Find nearest neighbors
		tree->search(objX[i], neighborsNumber, &indices, &distances);
		for (int m = 0; m < neighborsNumber - 1; m++) {
			nearestNeighborIndexes[i * (neighborsNumber - 1) + 2 + m] = indices[m + 1].index();
			nearestNeighborDistances[i * (neighborsNumber - 1) + 1 + m] = distances[m + 1];
		}
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
			for (int m = 0; m < K; m++) {
				curP[m] = exp(-beta * distances[m + 1]);
			}

			// Compute entropy of current row
			sumP = DBL_MIN;
			for (int m = 0; m < K; m++) {
				sumP += curP[m];
			}
			double H = .0;
			for (int m = 0; m < K; m++) {
				H += beta * (distances[m + 1] * curP[m]);
			}
			H = (H / sumP) + log(sumP);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
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
		// Compute transition matrix
		beta = sqrt((1.0 / beta) / 2);
		for (int m = 0; m < K; m++) {
			curP[m] = exp(-distances[m + 1] / beta);
		}
		sumP = DBL_MIN;
		for (int m = 0; m < K; m++) {
			sumP += curP[m];
		}
		for (int m = 0; m < K; m++) {
			tempTranSitionIndex[i * K + m] = indices[m + 1].index();
			tempTranSitionValue[i * K + m] = curP[m] / sumP;
		}
		// Print progress
#pragma omp atomic
		++stepsCompleted;

		if (stepsCompleted % 10000 == 0)
		{
#pragma omp critical
			printf(" - point %d of %d\n", stepsCompleted, rowNum);
		}
	}
	printf("end compute transition\n");
	for (int i = 0; i < rowNum; i++) {
		for (int j = 0; j < K; j++) {
			tTransition.push_back(Eigen::Triplet<double>(i, tempTranSitionIndex[i * K + j], tempTranSitionValue[i * K + j]));
		}
	}
	transition.setFromTriplets(tTransition.begin(), tTransition.end());
	weight.setFromTriplets(tWeight.begin(), tWeight.end());
	head->initData(NULL, transition, weight, influence, indexes);
	length++;

	// clear memory
	delete[] tempTranSitionIndex;
	delete[] tempTranSitionValue;
	delete tree;
	printf("finish initData\n");
}

vector<int> LevelList::computeNextLevelIndexes() {
	vector<int> indexes, preIndexes = tail->getIndexes();
	SparseMatrix<double> preTransition = tail->getTransitionMatrix().adjoint();

	// rand walk in this level, and create next level by result
	int i, j;
	int tailSize = tail->getSize();
	int* randomWalkResult = (int*)calloc(tailSize, sizeof(int));
	clock_t t = clock();
#pragma omp for
	for (i = 0; i < tailSize; i++) {
		//#pragma omp for
		for (j = 0; j < walksNum; j++) {
			int l = i;
			for (int k = 0; k < walksLength; k++) {
				double p = 0;
				double randP = rand() / double(RAND_MAX + 1);
				for (SparseMatrix<double>::InnerIterator it(preTransition, l); it; ++it)
				{
					p += it.value();
					if (p >= randP) {
						l = it.row();
						break;
					}
				}
			}
#pragma omp critical
			{
				randomWalkResult[l]++;
			}
		}
	}

	printf("rand walk cost %f\n", float(clock() - t));
	// the next level has at least one index
	int maxWalkNum[4] = { 0, 0, 0, 0 };
	for (i = 0; i < tailSize; i++) {
		for (j = 0; j < 4; j++) {
			if (randomWalkResult[i] > maxWalkNum[j]) {
				if (j != 0) {
					maxWalkNum[j - 1] = maxWalkNum[j];
				}
				maxWalkNum[j] = randomWalkResult[i];
			}
			else {
			    break;
			}
		}
	}
	j = threhold;
	if (maxWalkNum[0] < threhold) {
		j = maxWalkNum[0];
	}
	for (i = 0; i < tailSize; i++) {
		if (randomWalkResult[i] >= j) {
			indexes.push_back(preIndexes[i]);
		}
	}
	free(randomWalkResult);
	randomWalkResult = NULL;
	return indexes;
}

SparseMatrix<double> LevelList::computeNextLevelInfluences(vector<int> indexes) {
	vector<int> preIndexes = tail->getIndexes();
	SparseMatrix<double> preTransition = tail->getTransitionMatrix().adjoint();
	int i, j, k, l;
	int tailSize = tail->getSize();
	SparseMatrix<double> influences(tailSize, indexes.size());
	vector<Eigen::Triplet<double> > tInfluences;
	double p, randP;
	int* flags = (int*)calloc(tailSize, sizeof(int));
	// compute influence from those end indexes to theirselves
	for (i = 0, j = 0, l = indexes.size(); i < tailSize; i++) {
		if (preIndexes[i] == indexes[j]) {
			tInfluences.push_back(Eigen::Triplet<double>(i, j, 1.0));
			flags[i] = j + 1;
			j++;
			if (j == l) {
				break;
			}
		}
	}

	// rand walk until arrive a end indexes or move the limit times
	vector<int> tempTargetIndexes;
	for (i = 0; i < tailSize; i++) {
		if (flags[i] == 0) {
			tempTargetIndexes.clear();
			for (j = 0; j < walksNum; j++) {
				l = i;
				for (k = 0; k < walksLength * 2; k++) {
					p = 0;
					randP = rand() / double(RAND_MAX + 1);
					for (SparseMatrix<double>::InnerIterator it(preTransition, l); it; ++it)
					{
						p += it.value();
						if (p >= randP) {
							l = it.row();
							break;
						}
					}
					if (flags[l] > 0) {
						tempTargetIndexes.push_back(flags[l] - 1);
						break;
					}
				}
			}
			l = tempTargetIndexes.size();
			double tempValue = double(l);
			for (j = 0; j < l; j++) {
				tInfluences.push_back(Eigen::Triplet<double>(i, tempTargetIndexes[j], 1.0 / tempValue));
			}
		}
	}
	influences.setFromTriplets(tInfluences.begin(), tInfluences.end());
	free(flags);
	flags = NULL;
	return influences;
}

int LevelList::computeNextLevel() {

	// compute next level indexes and influence first
	printf("computing next level indexes...\n");
	clock_t t = clock();
	vector<int> indexes = computeNextLevelIndexes();
	printf("compute index cost %f\n", float(clock() - t));
	printf("computing next level influence...\n");
	t = clock();
	SparseMatrix<double> transition, weight, preWeight = tail->getWeight(), influence = computeNextLevelInfluences(indexes), transposeInfluence;
	printf("compute influence cost %f\n", float(clock() - t));
	transposeInfluence = influence.adjoint();

	//compute weight and transition by matrix operation

	printf("computing next level weight...\n");
	t = clock();
	weight = preWeight * influence;
	printf("compute weight cost %f\n", float(clock() - t));
	printf("computing next level transition...\n");
	t = clock();
	for (int k = 0; k < transposeInfluence.outerSize(); ++k) {
		for (SparseMatrix<double>::InnerIterator it(transposeInfluence, k); it; ++it)
		{
			it.valueRef() *= preWeight.coeff(0, k);
		}
	}
	transition = transposeInfluence * influence;
	double sum;
	for (int k = 0; k < transition.outerSize(); ++k) {
		sum = 0;
		for (SparseMatrix<double>::InnerIterator it(transition, k); it; ++it)
		{
			sum += it.value();
		}
		for (SparseMatrix<double>::InnerIterator it(transition, k); it; ++it)
		{
			it.valueRef() /= sum;
		}
	}
	transition = transition.adjoint();
	printf("compute transition cost %f\n", float(clock() - t));
	Level* newLevel = new Level(tail->getID() + 1, indexes.size(), tail->getSize());
	tail->setNext(newLevel);
	newLevel->initData(tail, transition, weight, influence, indexes);
	tail = newLevel;
	length++;
	return indexes.size();
}

void LevelList::computeLevelList(int** _levelSizes, int** _indexes, int** _levelInfluenceSizes, int** _pointInfluenceSizes, int** _influenceIndexes, double** _influenceValues) {
	int i = head->getSize();
	while (i > endLevelSize) {
		i = computeNextLevel();
		printf("Finish compute level%d(size = %d)\n", tail->getID(), i);
	}

	printf("Finish compute level list, creating record..\n");
	vector<int> levelIndexes;
	SparseMatrix<double> levelInfluence;
	if (head == tail) {
		*_levelSizes = new int[3];
		int* levelSizes = *_levelSizes;
		levelSizes[0] = 2;
		levelSizes[1] = 1;
		levelSizes[2] = head->getSize() + 1;

		levelIndexes = head->getIndexes();
		*_indexes = new int[levelIndexes.size() + 1];
		int* indexes = *_indexes;
		indexes[0] = levelIndexes.size();
		for (i = 0; i < levelIndexes.size(); i++)
		{
			indexes[i + 1] = 0;
		}

		*_levelInfluenceSizes = new int[2];
		int* levelInfluenceSizes = *_levelInfluenceSizes;
		levelInfluenceSizes[0] = 1;
		levelInfluenceSizes[1] = 1;

		*_pointInfluenceSizes = new int[2];
		int* pointInfluenceSizes = *_pointInfluenceSizes;
		pointInfluenceSizes[0] = 1;
		pointInfluenceSizes[1] = 1;

		*_influenceIndexes = new int[1];
		int* influenceIndexes = *_influenceIndexes;
		influenceIndexes[0] = 0;

		*_influenceValues = new double[1];
		double* influenceValues = *_influenceValues;
		influenceValues[0] = 0;
		return;
	}
	vector<int> vLevelSizes, vLevelInfluenceSizes, vPointInfluenceSizes, vInfluenceIndexes;
	vector<double> vInfluenceValues;
	Level *p = tail;
	int levelSizeSum = 1, influnceSizesSum = 1, k;
	int headSize = head->getSize(), id;
	int* levelCount = (int*)calloc(headSize, sizeof(int));
	while (p) {
		//compute levelSizes
		levelSizeSum += p->getSize();
		vLevelSizes.push_back(levelSizeSum);

		//compute indexes
		if (p != head) {
			vector<int> tempIndexes = p->getIndexes();
			id = p->getID();
			for (vector<int>::iterator iter = tempIndexes.begin(); iter != tempIndexes.end(); ++iter) {
				if (levelCount[*iter] == 0) {
					levelCount[*iter] = id;
				}
			}
			levelIndexes.insert(levelIndexes.end(), tempIndexes.begin(), tempIndexes.end());
		}
		else {
			for (i = 0; i < headSize; i++) {
				levelIndexes.push_back(levelCount[i]);
			}
		}

		//compute levelInfluenceSizes
		levelInfluence = p->getInfluenceMatrix();
		for (k = 0; k < levelInfluence.outerSize(); ++k) {
			for (SparseMatrix<double>::InnerIterator it(levelInfluence, k); it; ++it)
			{
				vInfluenceValues.push_back(it.value());
				vInfluenceIndexes.push_back(it.row());
			}
			vPointInfluenceSizes.push_back(vInfluenceIndexes.size() + 1);
		}
		if (k > 0) {
			vLevelInfluenceSizes.push_back(vPointInfluenceSizes.back());
		}
		p = p->getPre();
	}
	*_levelSizes = new int[vLevelSizes.size() + 2];
	int* levelSizes = *_levelSizes;
	levelSizes[0] = vLevelSizes.size() + 1;
	levelSizes[1] = 1;
#pragma omp parallel for
	for (i = 0; i < vLevelSizes.size(); i++) {
		levelSizes[i + 2] = vLevelSizes[i];
	}

	*_indexes = new int[levelIndexes.size() + 1];
	int* indexes = *_indexes;
	indexes[0] = levelIndexes.size();
#pragma omp parallel for
	for (i = 0; i < levelIndexes.size(); i++) {
		indexes[i + 1] = levelIndexes[i];
	}

	*_levelInfluenceSizes = new int[vLevelInfluenceSizes.size() + 2];
	int* levelInfluenceSizes = *_levelInfluenceSizes;
	levelInfluenceSizes[0] = vLevelInfluenceSizes.size() + 1;
	levelInfluenceSizes[1] = 1;
#pragma omp parallel for
	for (i = 0; i < vLevelInfluenceSizes.size(); i++) {
		levelInfluenceSizes[i + 2] = vLevelInfluenceSizes[i];
	}

	*_pointInfluenceSizes = new int[vPointInfluenceSizes.size() + 2];
	int* pointInfluenceSizes = *_pointInfluenceSizes;
	pointInfluenceSizes[0] = vPointInfluenceSizes.size() + 1;
	pointInfluenceSizes[1] = 1;
#pragma omp parallel for
	for (i = 0; i < vPointInfluenceSizes.size(); i++) {
		pointInfluenceSizes[i + 2] = vPointInfluenceSizes[i];
	}

	*_influenceIndexes = new int[vInfluenceIndexes.size() + 1];
	int* influenceIndexes = *_influenceIndexes;
	influenceIndexes[0] = vInfluenceIndexes.size();
#pragma omp parallel for
	for (i = 0; i < vInfluenceIndexes.size(); i++) {
		influenceIndexes[i + 1] = vInfluenceIndexes[i];
	}

	*_influenceValues = new double[vInfluenceValues.size() + 1];
	double* influenceValues = *_influenceValues;
	influenceValues[0] = vInfluenceValues.size();
#pragma omp parallel for
	for (i = 0; i < vInfluenceValues.size(); i++) {
		influenceValues[i + 1] = vInfluenceValues[i];
	}
	free(levelCount);
	levelCount = NULL;
}

void getNearestNeighbors(int rowNum, int* oldIndexes, int oldNum, int* newIndexes, int newNum,
	int* nearestNeighborIndexes, double* nearestNeighborDistances,
	int** _flags, int** _rows, int** _oldRows, int** _cols, int** _oldCols, double** _values, double** _oldValues) {
	*_flags = (int*)calloc(2, sizeof(int));
	int *flags = *_flags;
	int neighborsNumber = nearestNeighborIndexes[1];
	*_rows = (int*)malloc((newNum + oldNum + 1) * sizeof(int));
	*_oldRows = (int*)malloc((newNum + 1) * sizeof(int));
	int *rows = *_rows;
	int *oldRows = *_oldRows;
	rows[0] = 0;
	oldRows[0] = 0;

	int* flag = new int[rowNum];
	for (int i = 0; i < rowNum; i++) {
		flag[i] = -1;
	}
	for (int i = 0; i < oldNum; i++) {
		flag[oldIndexes[i]] = i;
	}
	for (int i = 0; i < newNum; i++) {
		flag[newIndexes[i]] = i + oldNum;
	}
	vector<int> vCols, vOldCols, vTempOldCols, vTempCols;
	vector<double> vValues, vOldValues, vTempValues, vTempOldValues;
	for (int i = 0; i < oldNum; i++) {
		int temp = oldIndexes[i];
		vTempCols.clear();
		vTempValues.clear();
		for (int j = 0; j < neighborsNumber; j++) {
			int tempNeighborIndex = nearestNeighborIndexes[temp * neighborsNumber + 2 + j];
			double tempNeighborDistance = nearestNeighborDistances[temp * neighborsNumber + 1 + j];
			if (flag[tempNeighborIndex] > -1) {
				vTempCols.push_back(flag[tempNeighborIndex]);
				vTempValues.push_back(tempNeighborDistance);
			}
		}
		if (vTempCols.size() > 0) {
			flags[0] += 1;
			vCols.insert(vCols.end(), vTempCols.begin(), vTempCols.end());
			vValues.insert(vValues.end(), vTempValues.begin(), vTempValues.end());
		}
		rows[i + 1] = rows[i] + vTempCols.size();
	}

	for (int i = 0; i < newNum; i++) {
		int temp = newIndexes[i];
		vTempCols.clear();
		vTempValues.clear();
		vTempOldCols.clear();
		vTempOldValues.clear();
		for (int j = 0; j < neighborsNumber; j++) {
			int tempNeighborIndex = nearestNeighborIndexes[temp * neighborsNumber + 2 + j];
			double tempNeighborDistance = nearestNeighborDistances[temp * neighborsNumber + 1 + j];
			if (flag[tempNeighborIndex] > -1) {
				vTempCols.push_back(flag[tempNeighborIndex]);
				vTempValues.push_back(tempNeighborDistance);
				if (flag[tempNeighborIndex] < oldNum) {
					vTempOldCols.push_back(flag[tempNeighborIndex]);
					vTempOldValues.push_back(tempNeighborDistance);
				}
			}
		}
		if (vTempCols.size() > 0) {
			flags[0] += 1;
			vCols.insert(vCols.end(), vTempCols.begin(), vTempCols.end());
			vValues.insert(vValues.end(), vTempValues.begin(), vTempValues.end());
		}
		rows[i + 1 + oldNum] = rows[i + oldNum] + vTempCols.size();


		if (vTempOldCols.size() > 0) {
			flags[1] += 1;
			vOldCols.insert(vOldCols.end(), vTempOldCols.begin(), vTempOldCols.end());
			vOldValues.insert(vOldValues.end(), vTempOldValues.begin(), vTempOldValues.end());
		}
		oldRows[i + 1] = oldRows[i] + vTempOldCols.size();
	}
	*_cols = (int*)malloc(rows[newNum + oldNum] * sizeof(int));
	*_oldCols = (int*)malloc(oldRows[newNum] * sizeof(int));
	int *cols = *_cols;
	int *oldCols = *_oldCols;
	*_values = (double*)malloc(rows[newNum + oldNum] * sizeof(double));
	*_oldValues = (double*)malloc(oldRows[newNum] * sizeof(double));
	double *values = *_values;
	double *oldValues = *_oldValues;
#pragma omp for
	for (int i = 0; i < rows[newNum + oldNum]; i++) {
		cols[i] = vCols[i];
		values[i] = vValues[i];
	}
#pragma omp for
	for (int i = 0; i < oldRows[newNum]; i++) {
		oldCols[i] = vOldCols[i];
		oldValues[i] = vOldValues[i];
	}
	delete[] flag;
}

void getInfluenceIndexes(int* levelSizes, int* indexes, int* levelInfluenceSizes, int* pointInfluenceSizes,
	int* influenceIndexes, double* influenceValues, int *indexSet, int size, double minInfluenceValue, 
	int** _resultSet) {
	if (size < 0) {
		return;
	}
	int levelID = levelSizes[0] - 2;
	int i, j, n = levelSizes[levelSizes[0] - 1];
	vector<int> levelIndexes;

	// get the ID of the level which contain all index in indexSet
	for (i = 0; i < size; i++) {
		j = indexes[n + indexSet[i]];
		if (j < levelID) {
			levelID = j;
			levelIndexes.clear();
		}
		levelIndexes.push_back(indexSet[i]);
	}

	// check if no influence
	if (levelID == 0 || pointInfluenceSizes[0] == 1) {
		*_resultSet = new int[1];
		int* resultSet = *_resultSet;
		resultSet[0] = 0;
		return;
	}

	// get those indexes in indexSet which in the level got
	sort(levelIndexes.begin(), levelIndexes.end());
	int beginIndex = levelSizes[levelSizes[0] - levelID - 1];
	int endIndex = levelSizes[levelSizes[0] - levelID];
	j = 0;
	int levelIndexSize = levelIndexes.size();
	int* influenceSource = new int[levelIndexSize];

	// only consider the influence of those point in the lowest level
	for (i = 0; i + beginIndex < endIndex; i++) {
		if (indexes[i + beginIndex] == levelIndexes[j]) {
			influenceSource[j] = i;
			j++;
			if (j == levelIndexSize) {
				break;
			}
		}
	}

	// get index range of those indexes in pointInfluenceSizes
	n = levelInfluenceSizes[0] - levelID;
	i = levelInfluenceSizes[n];

	// binary search
	int l, r, temp;
	l = 1;
	r = pointInfluenceSizes[0];
	while (l <= r) {
		temp = pointInfluenceSizes[(l + r) / 2];
		if (temp == i) {
			i = (l + r) / 2;
			break;
		}
		else if (temp < i) {
			l = (l + r) / 2 + 1;
		}
		else {
			r = (l + r) / 2 - 1;
		}
	}
	j = levelInfluenceSizes[n + 1];
	l = 1;
	r = pointInfluenceSizes[0];
	while (l <= r) {
		temp = pointInfluenceSizes[(l + r) / 2];
		if (temp == j) {
			j = (l + r) / 2;
			break;
		}
		else if (temp < j) {
			l = (l + r) / 2 + 1;
		}
		else {
			r = (l + r) / 2 - 1;
		}
	}

	// get indexes be influenced
	l = 0;
	set<int> influenceTarget;
	int startIndex = levelSizes[levelSizes[0] - levelID];
	for (n = 0; n < j - i; n++) {
		if (influenceSource[l] == n) {
			r = pointInfluenceSizes[n + i + 1];
			for (int m = pointInfluenceSizes[n + i]; m < r; m++) {
				if (influenceValues[m] >= minInfluenceValue) {
					if (levelID == 1) {
						influenceTarget.insert(influenceIndexes[m]);
					}
					else {
						influenceTarget.insert(indexes[startIndex + influenceIndexes[m]]);
					}
				}
			}
			l++;
		}
	}

	// remove indexSet from result
	i = 0;
	for (set<int>::iterator it = influenceTarget.begin(); it != influenceTarget.end();) {
		if (*it == indexSet[i]) {
			influenceTarget.erase(it++);
			i++;
		}
		else {
			it++;
		}
	}

	// translate result to int*
	j = influenceTarget.size();
	*_resultSet = new int[j + 1];
	int* resultSet = *_resultSet;
	resultSet[0] = j;
	i = 0;
	for (set<int>::iterator it = influenceTarget.begin(); it != influenceTarget.end(); it++) {
		resultSet[i + 1] = *it;
		i++;
	}

	//free memory
	delete[] influenceSource;
}

void getProbabilityDistribution(int* levelIDs, int* levelSizes,
		int* indexSet, int size,
		int* transitionSizes, int* transitionIndexes, double *transitionValues,
		int** _rowP, int** _colP, double** _valP) {
	int *orderRecorder = new int[size];
	for (int i = 0; i < size; i++) {
		orderRecorder[i] = i;
	}
	QSort(indexSet, orderRecorder, 0, size - 1);
	// get the ID of the level which contain all index in indexSet
	int levelID = levelSizes[0] - 2;
	int j, n = levelSizes[levelSizes[0] - 1];
	for (int i = 0; i < size; i++) {
		j = levelIDs[n + indexSet[i]];
		if (j < levelID) {
			levelID = j;
		}
	}
	levelID = levelSizes[0] - 1 - levelID;
	int levelSize = levelSizes[levelID + 1] - levelSizes[levelID];
	vector<int> vRowP, vColP;
	vector<double> vValP;
	vRowP.push_back(0);
	if (levelID != levelSizes[0] - 1) {
		int j = 0;
		for (int i = 0; i < levelSize; i++) {
			if (indexSet[j] == levelIDs[levelSizes[levelID] + i]) {
				indexSet[j] = i;
				j++;
			}
			if (j == size) {
				break;
			}
		}
		int beginIndex = transitionSizes[levelID * 3 - 2];
		for (int i = 0; i < size; i++) {
			int count = 0;
			for (j = 0; j < size; j++) {
				if (transitionValues[beginIndex + indexSet[i] * levelSize + indexSet[j]] > 0) {
					vValP.push_back(transitionValues[beginIndex + indexSet[i] * levelSize + indexSet[j]]);
					vColP.push_back(j);
					count++;
				}
			}
			vRowP.push_back(count + vRowP.back());
		}
	}
	else {
		int beginIndex = transitionSizes[levelID * 3 - 2];
		int neighborNumber = transitionSizes[levelID * 3];
		for (int i = 0; i < size; i++) {
			int count = 0;
			int k = 0;
			for (j = 0; j < size; j++) {
				if (indexSet[j] == transitionIndexes[indexSet[i] * neighborNumber + k]) {
					if (transitionValues[beginIndex + indexSet[i] * neighborNumber + k] > 0) {
						vValP.push_back(transitionValues[beginIndex + indexSet[i] * neighborNumber + k]);
						vColP.push_back(j);
						k++;
						count++;
					}
				}
				else if (indexSet[j] > transitionIndexes[indexSet[i] * neighborNumber + k]) {
					k++;
					j--;
				}
				if (k == neighborNumber) {
					break;
				}
			}
			vRowP.push_back(count + vRowP.back());
		}
	}
	*_rowP = new int[vRowP.size()];
	int *rowP = *_rowP;
	*_colP = new int[vColP.size()];
	int *colP = *_colP;
	*_valP = new double[vValP.size()];
	double *valP = *_valP;
	for (int i = 0; i < vRowP.size() - 1; i++) {
		rowP[orderRecorder[i] + 1] = vRowP[i + 1] - vRowP[i];
	}
	rowP[0] = 0;
	for (int i = 1; i <= size; i++) {
		rowP[i] += rowP[i - 1];
	}
	for (int i = 0; i < size; i++) {
		int trueIndex = rowP[orderRecorder[i]];
		for (int j = vRowP[i]; j < vRowP[i + 1]; j++, trueIndex++) {
			colP[trueIndex] = orderRecorder[vColP[j]];
			valP[trueIndex] = vValP[j];
		}
	}
}