import numpy as np
from LandmarkBasedSamplingAndDescendingDimension import LandmarkBasedSampling
import time
import pylab as Plot

color = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
    3: (1, 1, 0),
    4: (0, 1, 1),
    5: (1, 0, 1),
    6: (0, 0, 0),
    7: (0.1, 0.5, 0.9),
    8: (0.9, 0.5, 0.1),
    9: (0.1, 0.2, 0.7)
}

def showIteration(label, name, color):
    file = open('iter.txt', 'r')
    a = file.readline()
    b = a.split(' ')
    N = int(b[0])
    D = int(b[1])
    for i in range(D):
        X = []
        Y = []
        for j in range(N):
            a = file.readline()
            b = a.split(' ')
            X.append(float(b[0]))
            Y.append(float(b[1]))
        Plot.scatter(X, Y, 10, c=[color[x] for x in label])
        Plot.savefig(name + 'iter' + str(i * 4) + '.png')
        Plot.close()
    file.close()

def showOldNeighbors(label, name):
    file = open('outOld.txt', 'r')
    outFile = open(name + '.txt', 'a')
    a = file.readline()
    b = a.split('=')
    size = int(b[1]) - int(b[0])
    trueCount = 0
    totalCount = 0
    falseExisted = 0
    for i in range(size):
        a = file.readline()
        b = a.split(' ')
        c = []
        for x in b:
            if x != '\n':
                c.append(int(x))
        temp = trueCount
        for j in range(c[1]):
            if label[c[0]] == label[c[j + 2]]:
                trueCount += 1
        if trueCount == temp:
            falseExisted += 1
        totalCount += c[1]
    outFile.write("trueCount/totalCount:" + str(trueCount) + "/" + str(totalCount) + ", falseExisted/size:" + str(falseExisted) + '/' + str(size) + "\n")
    file.close()
    outFile.close()

def showOldNeighborPosition(oldY, name):
    file = open('outOld.txt', 'r')
    outFile = open(name + '.txt', 'a')
    a = file.readline()
    b = a.split('=')
    size = int(b[1]) - int(b[0])
    for i in range(size):
        a = file.readline()
        b = a.split(' ')
        c = []
        for x in b:
            if x != '\n':
                c.append(int(x))
        outFile.write(str(c[0]) + ":" + str(oldY[c[0], 0]) + "," + str(oldY[c[0], 1]) + "\n")

        for j in range(c[1]):
            outFile.write(str(oldY[c[j + 2], 0]) + "," + str(oldY[c[j + 2], 1]) + "\n")
        outFile.write("\n")
    file.close()
    outFile.close()

totalLabel = np.loadtxt('data/label.txt')
file = open('data/data.txt', 'r')
totalData = []
for i in range(10000):
    if i % 1000 == 0:
        print(i)
    temp = []
    a = file.readline()
    b = a.split('\t')
    for c in b:
        if(len(c) > 0):
            temp.append(float(c))
    totalData.append(temp)
file.close()
data = np.array(totalData, dtype=np.float64)
(rowNum, dim) = data.shape
perp = 30
nJobs = 4
maxIter = 600
sampling = LandmarkBasedSampling(nJobs=nJobs, perplexity=perp)
ti = time.time()
resultSet = sampling.sampling(data, 1000, 100, 50, 160)
ti = time.time() - ti
print("sampling cost: ", ti)
theOldSet = []
theOldLabel = []
theOldSize = 0
theOldY = []
# test sampling
levelNumber = sampling.getLevelNumber()
# if levelNumber < 3:
#     exit(1)
for k in range(levelNumber):
    theNewSet = sampling.getLevelIndexes(k + 1)
    theSize = len(theNewSet)
    tempSet = sorted(theOldSet)
    i = 0
    j = 0
    while i < theSize and j < theOldSize:
        if theNewSet[i] == tempSet[j]:
            del theNewSet[i]
            theSize -= 1
            j += 1
        elif theNewSet[i] > tempSet[j]:
            j += 1
        else:
            i += 1
    theNewLabel = []
    for x in theNewSet:
        theNewLabel.append(totalLabel[x])
    tempSet = theOldSet.copy()
    tempSet.extend(theNewSet)
    temp = sampling.tsneRun(tempSet, nJobs=nJobs, perplexity=perp, nIter=maxIter)
    # tempSet.sort()
    # temp = sampling.hsneRun(tempSet, nJobs=nJobs, perplexity=perp, nIter=maxIter)
    iterLabel = []
    for x in tempSet:
        iterLabel.append(totalLabel[x])
    showIteration(iterLabel, "1" + str(k), color)
    Plot.scatter(temp[0:theOldSize, 0], temp[0:theOldSize, 1], 10,
                 c=[color[x] for x in theOldLabel])
    Plot.scatter(temp[theOldSize:, 0], temp[theOldSize:, 1], 0.0001,
                 c=[color[x] for x in theNewLabel])
    Plot.savefig('test sampling' + str(k + 1) + ' old.png')
    Plot.close()
    Plot.scatter(temp[0:theOldSize, 0], temp[0:theOldSize, 1], 10,
                 c=[color[x] for x in theOldLabel])
    Plot.scatter(temp[theOldSize:, 0], temp[theOldSize:, 1], 1,
                 c=[color[x] for x in theNewLabel])
    Plot.savefig('test sampling' + str(k + 1) + '.png')
    Plot.close()
    theOldY = sampling.tsneRun(theNewSet, theOldSet, theOldY, nJobs=nJobs, perplexity=perp, nIter=maxIter)
    # theOldY = sampling.hsneRun(theNewSet, theOldSet, theOldY, nJobs=nJobs, perplexity=perp, nIter=maxIter)
    showIteration(iterLabel, "2" + str(k), color)

    Plot.scatter(theOldY[0:theOldSize, 0], theOldY[0:theOldSize, 1], 10,
                 c=[color[x] for x in theOldLabel])
    Plot.scatter(theOldY[theOldSize:, 0], theOldY[theOldSize:, 1], 0.0001,
                 c=[color[x] for x in theNewLabel])
    Plot.savefig('test sampling incremental' + str(k + 1) + ' old.png')
    Plot.close()
    Plot.scatter(theOldY[0:theOldSize, 0], theOldY[0:theOldSize, 1], 10,
                 c=[color[x] for x in theOldLabel])
    Plot.scatter(theOldY[theOldSize:, 0], theOldY[theOldSize:, 1], 1,
                 c=[color[x] for x in theNewLabel])
    Plot.savefig('test sampling incremental' + str(k + 1) + '.png')
    Plot.close()
    theOldSet.extend(theNewSet)
    theOldSize = len(theOldSet)
    theOldLabel.extend(theNewLabel)






##########################
# newSize = len(resultSet)
# indexSet = []
# oldSize = 0
# oldY = []
# count = 1
# while True:
#     tempTotalLabel = []
#     for x in indexSet:
#         tempTotalLabel.append(totalLabel[x])
#     for x in resultSet:
#         tempTotalLabel.append(totalLabel[x])
#     j = 0
#     k = 0
#     totalY = sampling.tsneRun(resultSet, nJobs=nJobs, perplexity=perp, nIter=maxIter)
#     showIteration(tempTotalLabel, "3" + str(count), color)
#     totalY = sampling.tsneRun(resultSet, indexSet, oldY, nJobs=nJobs, perplexity=perp, nIter=maxIter)
#     showIteration(tempTotalLabel, "4" + str(count), color)
#     if len(indexSet) > 0:
#         showOldNeighbors(tempTotalLabel, "a4" + str(count))
#         showOldNeighborPosition(totalY, "b4" + str(count))
#     #to show, and get indexSet
#     Plot.scatter(totalY[0:oldSize, 0], totalY[0:oldSize, 1], 10, c=[color[x] for x in tempTotalLabel[0:oldSize]])
#     Plot.scatter(totalY[oldSize:len(resultSet) + oldSize, 0], totalY[oldSize:len(resultSet) + oldSize, 1], 0.0001, c=[color[x] for x in tempTotalLabel[oldSize:len(resultSet) + oldSize]])
#     Plot.savefig('figure' + str(count) + ' old.png')
#     Plot.close()
#     Plot.scatter(totalY[0:oldSize, 0], totalY[0:oldSize, 1], 10, c=[color[x] for x in tempTotalLabel[0:oldSize]])
#     Plot.scatter(totalY[oldSize:len(resultSet) + oldSize, 0], totalY[oldSize:len(resultSet) + oldSize, 1], 1, c=[color[x] for x in tempTotalLabel[oldSize:len(resultSet) + oldSize]])
#     Plot.savefig('figure' + str(count) + '.png')
#     Plot.close()
#     s = input("Please input: minX maxX minY maxY\n")
#     nums = s.split(' ')
#     minX = float(nums[0])
#     maxX = float(nums[1])
#     minY = float(nums[2])
#     maxY = float(nums[3])
#     Plot.scatter(totalY[0:oldSize, 0], totalY[0:oldSize, 1], 10, c=[color[x] for x in tempTotalLabel[0:oldSize]])
#     Plot.scatter(totalY[oldSize:len(resultSet) + oldSize, 0], totalY[oldSize:len(resultSet) + oldSize, 1], 1, c=[color[x] for x in tempTotalLabel[oldSize:len(resultSet) + oldSize]])
#     Plot.plot([minX, maxX, maxX, minX, minX], [minY, minY, maxY, maxY, minY])
#     Plot.savefig('figure' + str(count) + 'chosen.png')
#     Plot.close()
#     tempIndexSet = []
#     oldY = []
#     for i in range(oldSize + newSize):
#         if minX < totalY[i, 0] < maxX and minY < totalY[i, 1] < maxY:
#             oldY.append(totalY[i])
#             if i < oldSize:
#                 tempIndexSet.append(indexSet[i])
#             else:
#                 tempIndexSet.append(resultSet[i - oldSize])
#     indexSet = sorted(tempIndexSet)
#     resultSet = sampling.getInfluenceIndexes(indexSet, 0.04)
#     newSize = len(resultSet)
#     oldSize = len(indexSet)
#     count += 1



# import time
#
# import numpy as np
# import pylab as Plot
# from MulticoreTSNE import MulticoreTSNE as TSNE
#
#
# def nNearestNeighborError(n, r, label):
#     res = r
#     length = len(res)
#     i = 0
#     result = []
#     for i in range(0, length):
#         temp = []
#         for j in range(0, length):
#             if i != j:
#                 dis = (res[i][0] - res[j][0]) * (res[i][0] - res[j][0]) + (res[i][1] - res[j][1]) * (
#                     res[i][1] - res[j][1])
#                 temp.append(dis)
#         tempVal = [100000000] * n
#         tempIndex = [0] * n
#         for k in range(0, length - 1):
#             for h in range(0, n):
#                 if temp[k] < tempVal[h]:
#                     m = n - 1
#                     while m > h:
#                         tempVal[m] = tempVal[m - 1]
#                         tempIndex[m] = tempIndex[m - 1]
#                         m -= 1
#                     tempVal[h] = temp[k]
#                     if k >= i:
#                         tempIndex[h] = k + 1
#                     else:
#                         tempIndex[h] = k
#                     break
#         result.append(tempIndex)
#     j = 0
#     for i in range(0, length):
#         count = 0
#         for y in result[i]:
#             if label[i] == label[y]:
#                 count += 1
#         if count * 2 < n:
#             j += 1
#     return float(j) / length
# log = open('log.txt', 'w')
# color = {
#     0: (49, 130, 189),
#     1: (158, 202, 225),
#     2: (230, 85, 13),
#     3: (253, 174, 107),
#     4: (49, 163, 84),
#     5: (161, 217, 155),
#     6: (117, 107, 177),
#     7: (188, 189, 220),
#     8: (99, 99, 99),
#     9: (189, 189, 189)
# }
# colorForNew = {
#     0: (107, 174, 214),
#     1: (198, 219, 239),
#     2: (253, 141, 60),
#     3: (253, 208, 162),
#     4: (116, 196, 118),
#     5: (199, 233, 192),
#     6: (117, 107, 177),
#     7: (218, 218, 235),
#     8: (150, 150, 150),
#     9: (217, 217, 217)
# }
#
# for i in range(10):
#     a, b, c = color[i]
#     a = float(a)
#     a /= 255.0
#     b = float(b)
#     b /= 255.0
#     c = float(c)
#     c /= 255.0
#     color[i] = (a, b, c)
#     a, b, c = colorForNew[i]
#     a = float(a)
#     a /= 255.0
#     b = float(b)
#     b /= 255.0
#     c = float(c)
#     c /= 255.0
#     colorForNew[i] = (a, b, c)
#
# totalLabel = np.loadtxt('mnist.pkl/label.txt')
# file = open('mnist.pkl/data.txt', 'r')
# totalData = []
# for i in range(10000):
#     if i % 1000 == 0:
#         print(i)
#     temp = []
#     a = file.readline()
#     b = a.split('\t')
#     for c in b:
#         if(len(c) > 0):
#             temp.append(float(c))
#     totalData.append(temp)
# file.close()
# data = np.array(totalData, dtype=np.float64)
# (rowNum, dim) = data.shape
# perp = 20
# nJobs = 4
# maxIter = 600
# tsne = TSNE(nJobs=nJobs, perplexity=perp, nIter=maxIter, angle=0.5)
# ti = time.time()
# resultSet = tsne.landmarkSampling(data, 1000, 100, 50, 100)
# ti = time.time() - ti
# print("sampling cost: ", ti)
# theOldSet = []
# theOldData = []
# theOldLabel = []
# theOldSize = 0
# theOldY = []
# # test sampling
# levelNumber = tsne.getLevelNumber()
# if levelNumber < 3:
#     exit(1)
# for k in range(levelNumber):
#     theNewSet = tsne.getLevelIndexes(k + 1)
#     theSize = len(theNewSet)
#     tempSet = sorted(theOldSet)
#     i = 0
#     j = 0
#     while i < theSize and j < theOldSize:
#         if theNewSet[i] == tempSet[j]:
#             del theNewSet[i]
#             theSize -= 1
#             j += 1
#         elif theNewSet[i] > tempSet[j]:
#             j += 1
#         else:
#             i += 1
#     theNewData = []
#     theNewLabel = []
#     for x in theNewSet:
#         theNewLabel.append(totalLabel[x])
#     j = 0
#     for i in range(rowNum):
#         if j < len(theNewSet):
#             if i == theNewSet[j]:
#                 theNewData.append(data[i])
#                 j += 1
#         else:
#             break
#     theOldSet.extend(theNewSet)
#     theOldData.extend(theNewData)
#     temp = tsne.fitTransform(np.array(theOldData))
#     Plot.scatter(temp[0:theOldSize, 0], temp[0:theOldSize, 1],
#                  c=[color[x] for x in theOldLabel], marker='+', s=10)
#     Plot.scatter(temp[theOldSize:, 0], temp[theOldSize:, 1], 8,
#                  c=[colorForNew[x] for x in theNewLabel])
#     Plot.savefig('test sampling' + str(k + 1) + '.png')
#     Plot.close()
#     theOldY = tsne.fitTransform(np.array(theOldData), theOldSize, theOldY)
#     Plot.scatter(theOldY[0:theOldSize, 0], theOldY[0:theOldSize, 1],
#                  c=[color[x] for x in theOldLabel], marker='+', s=10)
#     Plot.scatter(theOldY[theOldSize:, 0], theOldY[theOldSize:, 1], 8,
#                  c=[colorForNew[x] for x in theNewLabel])
#     Plot.savefig('test sampling incremental' + str(k + 1) + '.png')
#     Plot.close()
#     theOldSize = len(theOldData)
#     theOldLabel.extend(theNewLabel)
####################################################################################################
# newSize = len(resultSet)
# indexSet = []
# oldSize = 0
# oldY = []
# count = 1
# while True:
#     tempOldData = []
#     tempNewData = []
#     tempTotalLabel = []
#     for x in indexSet:
#         tempTotalLabel.append(totalLabel[x])
#     for x in resultSet:
#         tempTotalLabel.append(totalLabel[x])
#     j = 0
#     k = 0
#     for i in range(rowNum):
#         if j < newSize:
#             if i == resultSet[j]:
#                 tempNewData.append(data[i])
#                 j += 1
#         elif k < oldSize:
#             if i == indexSet[k]:
#                 tempOldData.append(data[i])
#                 k += 1
#         else:
#             break
#     tempOldData.extend(tempNewData)
#     totalY = tsne.fitTransform(np.array(tempOldData), oldSize, oldY)
#     #to show, and get indexSet
#     Plot.scatter(totalY[:, 0], totalY[:, 1], 10, c=[color[x] for x in tempTotalLabel])
#     Plot.savefig('figure' + str(count) + '.png')
#     Plot.close()
#     str = input("Please input: minX maxX minY maxY\n")
#     nums = str.split(' ')
#     minX = float(nums[0])
#     maxX = float(nums[1])
#     minY = float(nums[2])
#     maxY = float(nums[3])
#     tempIndexSet = []
#     for i in range(oldSize + newSize):
#         if minX < totalY[i, 0] < maxX and minY < totalY[i, 1] < maxY:
#             if i < oldSize:
#                 tempIndexSet.append(indexSet[i])
#             else:
#                 tempIndexSet.append(resultSet[i - oldSize])
#     indexSet = tempIndexSet
#     resultSet = tsne.getInfluenceIndexes(indexSet)
#     newSize = len(resultSet)
#     oldSize = len(indexSet)
#     count += 1


#old_result = tsne.fit_transform(old_data)
#embedding_array = old_result


#main
# t_count = 2
# while t_count < 3:
#
#     # ordinary
#     tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
#     ti = time.time()
#     embedding_array = tsne.fit_transform(total_data[0:(500 * (t_count + 2)), :])
#     ti = time.time() - ti
#     nne1 = n_nearest_neighbor_error(1, embedding_array, total_label[0:(500 * (t_count + 2))])
#     nne3 = n_nearest_neighbor_error(3, embedding_array, total_label[0:(500 * (t_count + 2))])
#     nne5 = n_nearest_neighbor_error(5, embedding_array, total_label[0:(500 * (t_count + 2))])
#     Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in total_label[0:(500 * (t_count + 2))]])
#     Plot.savefig('ord' + str(t_count) + 'perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
#     Plot.close()
#     log.write('ord' + str(t_count) + 'perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#               + str(n_jobs) + '\ntime: ' + str(ti) + '\nnne1: ' + str(nne1) + '\nnne3: ' + str(nne3) + '\nnne5: ' + str(nne5) + '\n\n')
#     # incremental
#     tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
#     ti = time.time()
#     embedding_array = tsne.fit_transform(total_data[0:(500 * (t_count + 2)), :], 1500, old_result)
#     old_result = embedding_array
#     ti = time.time() - ti
#     nne1 = n_nearest_neighbor_error(1, embedding_array, total_label[0:(500 * (t_count + 2))])
#     nne3 = n_nearest_neighbor_error(3, embedding_array, total_label[0:(500 * (t_count + 2))])
#     nne5 = n_nearest_neighbor_error(5, embedding_array, total_label[0:(500 * (t_count + 2))])
#     Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in total_label[0:(500 * (t_count + 2))]])
#     Plot.savefig('incremental' + str(t_count) + 'perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
#     Plot.close()
#     log.write('incremental' + str(t_count) + 'perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#               + str(n_jobs) + '\ntime: ' + str(ti) + '\nnne1: ' + str(nne1) + '\nnne3: ' + str(nne3) + '\nnne5: ' + str(nne5) + '\n\n')
#     t_count += 1
# log.close()



# total_label_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_labels.txt')
# total_data_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_X.txt')
# total_label = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/label.txt')
# total_data = []
# for i in range(20):
#     temp = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/data' + str(i) + '.txt')
#     total_data.append(temp)
#
# label_2500_1 = total_label_2500[0:2000]
# data_2500_1 = total_data_2500[0:2000, :]
# label_1 = total_label[0:1500]
# data_1 = [x[0:1500, :] for x in total_data]
#
# label_2500_2 = np.concatenate((total_label_2500[0:1500], total_label_2500[2000:2500]))
# data_2500_2 = np.concatenate((total_data_2500[0:1500, :], total_data_2500[2000:2500, :]))
# label_2 = np.concatenate((total_label[0:1000], total_label[1500:2000]))
# data_2 = [np.concatenate((x[0:1000, :], x[1500:2000, :])) for x in total_data]
#
#
#
# result = []
# perp = 8
# n_jobs = 4
# max_iter = 600
#
# # init run
# tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
# embedding_array = tsne.fit_transform(data_2500_1)
# result_2500 = embedding_array
# Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_1[0:1500]])
# Plot.savefig('init1 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# for i in range(20):
#     embedding_array = tsne.fit_transform(data_1[i])
#     result.append(embedding_array)
#     Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_1[0:1000]])
#     Plot.savefig('init1 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
#
#
#
# tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
# ti = time.time()
# embedding_array = tsne.fit_transform(data_2500_2)
# ti = time.time() - ti
# nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500_2)
# nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500_2)
# nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500_2)
# Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500_2])
# Plot.savefig('init2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
# Plot.savefig('init2 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# log.write('init2 data2500 perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#           + str(n_jobs) + '\ntime: ' + str(ti) + '\nnne1: ' + str(nne1) + '\nnne3: ' + str(nne3) + '\nnne5: ' + str(nne5) + '\n\n')
# print('init2 data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1, 'nne3:', nne3, 'nne5:', nne5)
# nne1 = 0
# nne3 = 0
# nne5 = 0
# ti = 0
# for i in range(20):
#     t = time.time()
#     embedding_array = tsne.fit_transform(data_2[i])
#     ti += time.time() - t
#     nne1 += n_nearest_neighbor_error(1, embedding_array, label_2)
#     nne3 += n_nearest_neighbor_error(3, embedding_array, label_2)
#     nne5 += n_nearest_neighbor_error(5, embedding_array, label_2)
#     Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2])
#     Plot.savefig('init2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
#     Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
#     Plot.savefig('init2 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
# log.write('init2 data0-19 perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#           + str(n_jobs) + '\ntime: ' + str(ti / 20) + '\nnne1: ' + str(nne1 / 20) + '\nnne3: ' + str(nne3 / 20) + '\nnne5: ' + str(nne5 / 20) + '\n\n')
# print('init2 data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:', nne1 / 20, 'nne3:',
#       nne3 / 20, 'nne5:', nne5 / 20)
#
# #保存降维结果到文件
# f = open('result_2500.txt', 'w')
# length = len(result_2500)
# i = 0
# while i < length:
#     f.write(str(result_2500[i, 0]) + '\t' + str(result_2500[i, 1]))
#     if i != length - 1:
#         f.write('\n')
#     i += 1
# for j in range(20):
#     f = open('result' + str(j) + '.txt', 'w')
#     length = len(result[j])
#     i = 0
#     while i < length:
#         f.write(str(result[j][i, 0]) + '\t' + str(result[j][i, 1]))
#         if i != length - 1:
#             f.write('\n')
#         i += 1
#
# result_2500 = np.loadtxt('result_2500.txt')
# result = []
# for i in range(20):
#     result.append(np.loadtxt('result' + str(i) + '.txt'))
#
# # incremental run old_num = 1500/2000 和 1000/1500
# tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
# ti = time.time()
# embedding_array = tsne.fit_transform(data_2500_2, 1500, result_2500[0:1500, :])
# ti = time.time() - ti
# nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500_2)
# nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500_2)
# nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500_2)
# Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500_2])
# Plot.savefig('incremental  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# Plot.scatter(result_2500[0:1500, 0], result_2500[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
# Plot.savefig('incremental true old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
# Plot.savefig('incremental old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
# Plot.close()
# log.write('incremental data2500 perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#           + str(n_jobs) + '\ntime: ' + str(ti) + '\nnne1: ' + str(nne1) + '\nnne3: ' + str(nne3) + '\nnne5: ' + str(nne5) + '\n\n')
# print('incremental data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1, 'nne3:', nne3, 'nne5:', nne5)
# nne1 = 0
# nne3 = 0
# nne5 = 0
# ti = 0
# for i in range(20):
#     t = time.time()
#     embedding_array = tsne.fit_transform(data_2[i], 1000, result[i][0:1000, :])
#     ti += time.time() - t
#     nne1 += n_nearest_neighbor_error(1, embedding_array, label_2)
#     nne3 += n_nearest_neighbor_error(3, embedding_array, label_2)
#     nne5 += n_nearest_neighbor_error(5, embedding_array, label_2)
#     Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2])
#     Plot.savefig('incremental  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
#     Plot.scatter(result[i][0:1000, 0], result[i][0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
#     Plot.savefig('incremental true old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
#     Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
#     Plot.savefig('incremental old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
#     Plot.close()
# log.write('incremental data0-19 perp: '+ str(perp) + '\nmax_iteration: ' + str(max_iter) + '\nn_jobs: '
#           + str(n_jobs) + '\ntime: ' + str(ti / 20) + '\nnne1: ' + str(nne1 / 20) + '\nnne3: ' + str(nne3 / 20) + '\nnne5: ' + str(nne5 / 20) + '\n\n')
# print('incremental data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:', nne1 / 20, 'nne3:',
#       nne3 / 20, 'nne5:', nne5 / 20)
# log.close()
