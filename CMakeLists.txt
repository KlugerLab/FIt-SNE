project(FIt-SNE)

set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -lfftw3 -lm")

add_executable(bin/fast_tsne src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp)