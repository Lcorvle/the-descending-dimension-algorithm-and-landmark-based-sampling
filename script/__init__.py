import numpy as np
import cffi
import psutil
import threading
import os
import sys
from ctypes import *
import gc

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    '''
        Only
            - nComponents
            - perplexity
            - angle
            - nIter
        parameters are used.
        Other are left for compatibility with sklearn TSNE.
    '''

    def __init__(self,
                 nComponents=2,
                 perplexity=30.0,
                 nIter=1000,
                 randomState=None,
                 angle=0.5,
                 nJobs=1):
        self.nComponents = nComponents
        self.angle = angle
        self.perplexity = perplexity
        self.nIter = nIter
        self.nJobs = nJobs
        self.randomState = -1 if randomState is None else randomState

        assert nComponents == 2, 'nComponents should be 2.'
        assert 0 < perplexity <= 50, 'perplexity should be 0 < perp <= 50.'

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            "void incrementalTsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, "
            "double theta, int _numThreads, int maxIter, int randomState, int oldNum, int flag, int* _flags, "
            "int* _rows, int* _oldRows, int* _cols, int* _oldCols, double* _values, double* _oldValues);"
            "void tsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, "
            "int _numThreads, int maxIter, int randomState, int flag, int* _flags, int* _rows, int* _oldRows, "
            "int* _cols, int* _oldCols, double* _values, double* _oldValues);")

        path = os.path.dirname(os.path.realpath(__file__))
        self.C = self.ffi.dlopen(path + "\\libLandmarkBasedSamplingAndDescendingDimension.dll")

    def run(self, X, oldNum = 0, oldY = []):

        assert X.ndim == 2, 'X should be 2D array.'
        assert X.dtype == np.float64, 'Only double arrays are supported for now. Use .astype(np.float64) to convert.'
        assert len(oldY) == oldNum, 'length of oldY should equal to oldNum.'

        if self.nJobs == -1:
            self.nJobs = psutil.cpu_count()

        assert self.nJobs > 0, 'Wrong nJobs parameter.'

        N, D = X.shape
        Y = np.zeros((N, self.nComponents))
        perp = 0
        if N > self.perplexity * 3:
            perp = self.perplexity
        else:
            perp = (N - 1) / 3
        cffiFlags = self.ffi.new('int*')
        cffiRows = self.ffi.new('int*')
        cffiOldRows = self.ffi.new('int*')
        cffiCols = self.ffi.new('int*')
        cffiOldCols = self.ffi.new('int*')
        cffiValues = self.ffi.new('double*')
        cffiOldValues = self.ffi.new('double*')
        neighborFlag = 0
        if oldNum == 0:
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)
            t = FuncThread(self.C.tsneRunDouble,
                           cffiX, N, D,
                           cffiY, self.nComponents,
                           perp, self.angle, self.nJobs, self.nIter, self.randomState,
                           neighborFlag, cffiFlags, cffiRows, cffiOldRows, cffiCols, cffiOldCols, cffiValues, cffiOldValues)

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        else:
            Y[0:oldNum] += oldY
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)

            t = FuncThread(self.C.incrementalTsneRunDouble,
                           cffiX, N, D,
                           cffiY, self.nComponents,
                           perp, self.angle, self.nJobs, self.nIter, self.randomState, oldNum,
                           neighborFlag, cffiFlags, cffiRows, cffiOldRows, cffiCols, cffiOldCols, cffiValues, cffiOldValues)

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()

        return Y


class LandmarkBasedSampling:
    def __init__(self,
                 perplexity=30.0,
                 nJobs=1):
        self.information = {}
        self.perplexity = perplexity
        self.nJobs = nJobs

        assert 0 < perplexity <= 50, 'perplexity should be 0 < perp <= 50.'

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            "void getProbabilityDistribution(int threads, int* levelIDs, int* levelSizes, int* indexSet, int size, int* transitionSizes, int* transitionIndexes, double *transitionValues, int** _rowP, int** _colP, double** _valP);"
            "void incrementalHsneRunDouble(double* Y, int N, int noDims, int oldNum,	double perplexity, double theta, int _numThreads, int maxIter, int randomState,	int* rowP, int* colP, double* valP);"
            "void hsneRunDouble(double* Y, int N, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int* rowP, int* colP, double* valP);"
            "void incrementalTsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int oldNum, int flag, int* _flags, int* _rows, int* _oldRows, int* _cols, int* _oldCols, double* _values, double* _oldValues);"
            "void tsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int flag, int* _flags, int* _rows, int* _oldRows, int* _cols, int* _oldCols, double* _values, double* _oldValues);"
            "void landMarkBasedSampling(int threads, double perplexity, int randWalksNum, int randWalksLength, int randWalksThrehold, int endSize, double* data, int rowNum, int dim, int** _levelSizes, int** _indexes, int** _levelInfluenceSizes, int** _pointInfluenceSizes, int** _influenceIndexes, double** _influenceValues, int** _nearestNeighborIndexes, double** _nearestNeighborDistances, int** _transitionSizes, int** _transitionIndexes, double **_transitionValues);"
            "void getNearestNeighbors(int threads, int rowNum, int* oldIndexes, int oldNum, int* newIndexes, int newNum, int* nearestNeighborIndexes, double* nearestNeighborDistances, int** _flags, int** _rows, int** _oldRows, int** _cols, int** _oldCols, double** _values, double** _oldValues);"
            "void getInfluenceIndexes(int threads, int* levelSizes, int* indexes, int* levelInfluenceSizes, int* pointInfluenceSizes, int* influenceIndexes, double* influenceValues, int *indexSet, int size, double minInfluenceValue, int** _resultSet);")

        path = os.path.dirname(os.path.realpath(__file__))
        self.C = self.ffi.dlopen(path + "\\libLandmarkBasedSamplingAndDescendingDimension.dll")

    def hsneRun(self,
                newIndexes,
                oldIndexes=[],
                oldY=[],
                nComponents=2,
                perplexity=30.0,
                nIter=1000,
                randomState=-1,
                angle=0.5,
                nJobs=1):
        assert self.information != {}, "You should run sampling first."
        oldNum = len(oldIndexes)
        if oldY != []:
            assert len(oldY) == oldNum, 'length of oldY should equal to oldNum.'

        newNum = len(newIndexes)
        if newNum == 0:
            return oldY
        rowNum = oldNum + newNum
        totalIndexes = oldIndexes + newIndexes
        aTotalIndexes = np.array(totalIndexes)
        cffiTotalIndexes = self.ffi.cast('int*', aTotalIndexes.ctypes.data)
        cffiRowP = self.ffi.new('int**')
        cffiColP = self.ffi.new('int**')
        cffiValP = self.ffi.new('double**')
        t = FuncThread(self.C.getProbabilityDistribution, nJobs, self.information['indexes'], self.information['levelSizes'],
                       cffiTotalIndexes, rowNum, self.information['transitionSizes'],
                       self.information['transitionIndexes'], self.information['transitionValues'], cffiRowP,
                       cffiColP, cffiValP)

        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        Y = np.zeros((rowNum, nComponents))
        perp = 0
        if rowNum > perplexity * 3:
            perp = perplexity
        else:
            perp = (rowNum - 1) / 3
        print('oldNum=', oldNum, 'newNum=', newNum)
        if oldNum == 0:
            cffiY = self.ffi.cast('double*', Y.ctypes.data)
            t = FuncThread(self.C.hsneRunDouble,
                           cffiY, rowNum, nComponents,
                           perp, angle, nJobs, nIter, randomState,
                           cffiRowP[0], cffiColP[0], cffiValP[0])

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        else:
            Y[0:oldNum] += oldY
            cffiY = self.ffi.cast('double*', Y.ctypes.data)
            t = FuncThread(self.C.incrementalHsneRunDouble,
                           cffiY, rowNum, nComponents, oldNum,
                           perp, angle, nJobs, nIter, randomState,
                           cffiRowP[0], cffiColP[0], cffiValP[0])

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        del cffiRowP
        del cffiColP
        del cffiValP
        return Y

    def tsneRun(self,
                newIndexes,
                oldIndexes = [],
                oldY=[],
                nComponents=2,
                perplexity=30.0,
                nIter=1000,
                randomState=-1,
                angle=0.5,
                nJobs=1):

        assert self.information != {}, "You should run sampling first."
        oldNum = len(oldIndexes)
        if oldY != []:
            assert len(oldY) == oldNum, 'length of oldY should equal to oldNum.'

        newNum = len(newIndexes)
        if newNum == 0:
            return oldY

        theOldData = []
        theNewData = []
        rowNum, dim = self.data.shape
        for i in range(newNum):
            theNewData.append(self.data[newIndexes[i]])
        for i in range(oldNum):
            theOldData.append(self.data[oldIndexes[i]])
        theOldData.extend(theNewData)
        X = np.array(theOldData)

        N, D = X.shape
        Y = np.zeros((N, nComponents))
        assert N > 3, 'number of data cannot be too small'
        perp = 0
        if N > perplexity * 3:
            perp = perplexity
        else:
            perp = (N - 1) / 3
        cffiFlags = self.ffi.new('int**')
        cffiRows = self.ffi.new('int**')
        cffiOldRows = self.ffi.new('int**')
        cffiCols = self.ffi.new('int**')
        cffiOldCols = self.ffi.new('int**')
        cffiValues = self.ffi.new('double**')
        cffiOldValues = self.ffi.new('double**')
        aOldIndexes = np.array(oldIndexes)
        aNewIndexes = np.array(newIndexes)
        cffiOldIndexes = self.ffi.cast('int*', aOldIndexes.ctypes.data)
        cffiNewIndexes = self.ffi.cast('int*', aNewIndexes.ctypes.data)

        t = FuncThread(self.C.getNearestNeighbors, nJobs, rowNum, cffiOldIndexes, oldNum, cffiNewIndexes, newNum,
                       self.information['nearestNeighborIndexes'], self.information['nearestNeighborDistances'],
                       cffiFlags, cffiRows, cffiOldRows, cffiCols, cffiOldCols, cffiValues, cffiOldValues)

        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        neighborFlag = 1
        if oldNum == 0:
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)
            t = FuncThread(self.C.tsneRunDouble,
                           cffiX, N, D,
                           cffiY, nComponents,
                           perp, angle, nJobs, nIter, randomState, 
                           neighborFlag, cffiFlags[0], cffiRows[0], cffiOldRows[0], cffiCols[0], cffiOldCols[0], cffiValues[0], cffiOldValues[0])

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        else:
            Y[0:oldNum] += oldY
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)

            t = FuncThread(self.C.incrementalTsneRunDouble,
                           cffiX, N, D,
                           cffiY, nComponents,
                           perp, angle, nJobs, nIter, randomState, oldNum, 
                           neighborFlag, cffiFlags[0], cffiRows[0], cffiOldRows[0], cffiCols[0], cffiOldCols[0], cffiValues[0], cffiOldValues[0])

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        return Y

    def getInfluenceIndexes(self, indexes, minInfluenceValue):
        assert self.information != {}, "You should run sampling first."
        array = np.array(indexes)
        indexSet = self.ffi.cast('int*', array.ctypes.data)
        resultSet = self.ffi.new('int**')
        t = FuncThread(self.C.getInfluenceIndexes, self.nJobs, self.information['levelSizes'], self.information['indexes'],
                       self.information['levelInfluenceSizes'], self.information['pointInfluenceSizes'],
                       self.information['influenceIndexes'], self.information['influenceValues'], indexSet, len(indexes), minInfluenceValue, resultSet)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        tempResult = resultSet[0]
        n = tempResult[0]
        result = []
        for i in range(n):
            result.append(tempResult[i + 1])
        return result

    def getLevelNumber(self):
        assert self.information != {}, "You should run sampling first."
        return self.information['levelSizes'][0] - 1

    def getLevelIndexes(self, levelID):
        assert self.information != {}, "You should run sampling first."
        num = self.information['levelSizes'][0]
        result = []
        if levelID == num - 1:
            temp = self.information['levelSizes'][num] - self.information['levelSizes'][num - 1]
            for i in range(temp):
                result.append(i)
            return result
        i = self.information['levelSizes'][levelID]
        while i < self.information['levelSizes'][levelID + 1]:
            result.append(self.information['indexes'][i])
            i += 1
        return result

    def sampling(self, X, endSize, randWalksNum = 100, randWalksLength = 50, randWalksThrehold = 100):
        assert endSize > 5, 'endSize connot be smaller than 6.'
        assert X.dtype == np.float64, 'Only double arrays are supported for now. Use .astype(np.float64) to convert.'

        rowNum, dim = X.shape
        cffiX = self.ffi.cast('double*', X.ctypes.data)
        levelSizes = self.ffi.new('int**')
        indexes = self.ffi.new('int**')
        levelInfluenceSizes = self.ffi.new('int**')
        pointInfluenceSizes = self.ffi.new('int**')
        influenceIndexes = self.ffi.new('int**')
        influenceValues = self.ffi.new('double**')
        nearestNeighborIndexes = self.ffi.new('int**')
        nearestNeighborDistances = self.ffi.new('double**')
        transitionSizes = self.ffi.new('int**')
        transitionIndexes = self.ffi.new('int**')
        transitionValues = self.ffi.new('double**')

        if self.information != {}:
            del self.information
            del self.data
            gc.collect()
        self.data = X

        perp = 0
        if rowNum > self.perplexity * 3:
            perp = self.perplexity
        else:
            perp = (rowNum - 1) / 3
        t = FuncThread(self.C.landMarkBasedSampling, self.nJobs, perp, randWalksNum, randWalksLength, randWalksThrehold,
                       endSize, cffiX, rowNum, dim, levelSizes, indexes, levelInfluenceSizes, pointInfluenceSizes,
                       influenceIndexes, influenceValues, nearestNeighborIndexes,
                       nearestNeighborDistances, transitionSizes, transitionIndexes, transitionValues)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        self.information['levelSizes'] = levelSizes[0]
        self.information['indexes'] = indexes[0]
        self.information['levelInfluenceSizes'] = levelInfluenceSizes[0]
        self.information['pointInfluenceSizes'] = pointInfluenceSizes[0]
        self.information['influenceIndexes'] = influenceIndexes[0]
        self.information['influenceValues'] = influenceValues[0]
        self.information['nearestNeighborIndexes'] = nearestNeighborIndexes[0]
        self.information['nearestNeighborDistances'] = nearestNeighborDistances[0]
        self.information['transitionSizes'] = transitionSizes[0]
        self.information['transitionIndexes'] = transitionIndexes[0]
        self.information['transitionValues'] = transitionValues[0]
        return self.getLevelIndexes(0)