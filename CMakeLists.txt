CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)


FIND_PACKAGE(OpenMP REQUIRED)




SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -ffast-math -funroll-loops -static -static-libgcc -static-libstdc++")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


FILE(GLOB src src/*.cpp src/tsne/*.cpp src/hsne/*.cpp src/tool/*.cpp src/landmarkBasedSampling/*.cpp)

ADD_LIBRARY(LandmarkBasedSamplingAndDescendingDimension MODULE ${src})
