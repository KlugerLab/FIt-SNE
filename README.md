# FFT-accelerated Interpolation-based t-SNE (FIt-SNE)
## Introduction
t-Stochastic Neighborhood Embedding ([t-SNE](https://lvdmaaten.github.io/tsne/)) is a highly successful method for dimensionality reduction and visualization of high dimensional datasets.  A popular [implementation](https://github.com/lvdmaaten/bhtsne) of t-SNE uses the Barnes-Hut algorithm to approximate the gradient at each iteration of gradient descent. We modified this implementation as follows:

* Computation of the N-body Simulation: Instead of approximating the N-body simulation using Barnes-Hut, we interpolate onto an equispaced grid and use FFT to perform the convolution, dramatically reducing the time to compute the gradient at each iteration of gradient descent. See the [this](http://gauss.math.yale.edu/~gcl22/blog/numerics/low-rank/t-sne/2018/01/11/low-rank-kernels.html) post for some intuition on how it works.
* Computation of Input Similiarities: Instead of computing nearest neighbors using vantage-point trees, we approximate nearest neighbors using the [Annoy](https://github.com/spotify/annoy) library. The neighbor lookups are multithreaded to take advantage of machines with multiple cores. Using "near" neighbors as opposed to strictly "nearest" neighbors is faster, but also has a smoothing effect, which can be useful for embedding some datasets (see [Linderman et al. (2017)](https://arxiv.org/abs/1711.04712)). If subtle detail is required (e.g. in identifying small clusters), then use vantage-point trees (which is also multithreaded in this implementation). 
* Early exaggeration: In [Linderman and Steinerberger (2017)](https://arxiv.org/abs/1706.02582), we showed that appropriately choosing the early exaggeration coefficient can lead to improved embedding of swissrolls and other synthetic datasets.
* Late exaggeration: Increasing the exaggeration coefficient late in the optimization process (e.g. after 800 of 1000 iterations) can improve separation of the clusters.

Check out our [preprint](https://arxiv.org/abs/1712.09005) for more details and some benchmarks.

R and Matlab wrappers are `fast_tsne.R` and `fast_tsne.m` respectively. [Gioele La Manno](https://twitter.com/GioeleLaManno) implemented a Python (Cython) wrapper, which is available on PyPI [here](https://pypi.python.org/pypi/fitsne).

## Installation
Note that the code has been tested for OS X and Linux, but not for Windows.
The only prerequisite is [FFTW](http://www.fftw.org/), which can be downloaded and installed from the website. Then, from the root directory compile the code as:
```bash
    g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_tsne -pthread -lfftw3 -lm
```
And you're good to go! Check out `examples/` for usage.

**Note:** If you update to a new version of FIt-SNE using `git pull`, be sure to recompile. 

## References
If you use our software, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2017). Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding. (2017) *arXiv:1712.09005* ([link](https://arxiv.org/abs/1712.09005))

Our implementation is derived from the Barnes-Hut implementation:

Laurens van der Maaten (2014). Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research, 15(1):3221–3245. ([link](https://dl.acm.org/citation.cfm?id=2627435.2697068))
