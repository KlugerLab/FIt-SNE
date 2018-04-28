project(FIt-SNE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -lfftw3 -lm")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fno-omit-frame-pointer -fsanitize=address")
#set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_executable(bin/fast_tsne src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp)