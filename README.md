# FFT-accelerated Interpolation-based t-SNE (FIt-SNE) for Windows

## Introduction
This is a small modification of https://github.com/KlugerLab/FIt-SNE to allow compilation under Windows using MS Visual Studio; the Visual Studio project and solutions were added along with all additional dependencies including the FFWT library, which is distributed under the GNU General Public License.

## Compiling

### OSX and Linux
```
g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_tsne -pthread -lfftw3 -lm
```

### Windows
Open the provided FItSNE solution (FItSNE.sln) using MS Visual Studio and rebuild it. Tested with MS Visual Studio 2015 (i.e., MS Visual Studio Version 14)

## References
If you use this software, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2017). Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding. (2017) *arXiv:1712.09005* ([link](https://arxiv.org/abs/1712.09005))

## More information
Please see https://github.com/KlugerLab/FIt-SNE
