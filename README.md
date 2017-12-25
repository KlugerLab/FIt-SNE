# FIt-SNE
## Introduction
t-Stochastic Neighborhood Embedding ([t-SNE](https://lvdmaaten.github.io/tsne/)) is a highly successful method for dimensionality reduction and visualization of high dimensional datasets.  A popular [implementation](https://github.com/lvdmaaten/bhtsne) of t-SNE uses the Barnes-Hut algorithm to approximate the gradient at each iteration of gradient descent. We modified this implementation as follows:

* Computation of Input Similiarities: Instead of computing nearest neighbors using vantage trees, we approximate nearest neighbors using the [ANNOY](https://github.com/spotify/annoy) library. The neighbor lookups are multithreaded to take advantage of machines with multiple cores.
* Computation of the N-body Simulation: Instead of approximating the nbody simulation using Barnes-Hut, we interpolate onto an equispaced grid and use FFT to perform the convolution, giving dramatic increases in speed and accuracy.
* Early exaggeration: In [Linderman and Steinerberger 2017](https://arxiv.org/abs/1706.02582) we showed that appropriately choosing the early exaggeration coefficient can lead to improved embedding of swissrolls and other synthetic datase
ts
* Late exaggeration: By increasing the exaggeration coefficient late in the optimization process (e.g. after 800 of 1000 iterations) can improve separation of the clusters

## Installation
Note that the code has been tested for OS X and Linux, but not for Windows.
The only prerequisite is [FFTW](http://www.fftw.org/), which can be downloaded and installed from the website. Then, from the root directory compile the code as:
```bash
    g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_tsne -pthread -lfftw3 -lm
```
And you're good to go! Check out `examples/` for usage

## References
If you use our software, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2017). Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding. arXiv preprint.

