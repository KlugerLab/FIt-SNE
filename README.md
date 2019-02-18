# FFT-accelerated Interpolation-based t-SNE (FIt-SNE)
## Introduction
t-Stochastic Neighborhood Embedding ([t-SNE](https://lvdmaaten.github.io/tsne/)) is a highly successful method for dimensionality reduction and visualization of high dimensional datasets.  A popular [implementation](https://github.com/lvdmaaten/bhtsne) of t-SNE uses the Barnes-Hut algorithm to approximate the gradient at each iteration of gradient descent. We accelerated this implementation as follows:

* Computation of the N-body Simulation: Instead of approximating the N-body simulation using Barnes-Hut, we interpolate onto an equispaced grid and use FFT to perform the convolution, dramatically reducing the time to compute the gradient at each iteration of gradient descent. See the [this](http://gauss.math.yale.edu/~gcl22/blog/numerics/low-rank/t-sne/2018/01/11/low-rank-kernels.html) post for some intuition on how it works.
* Computation of Input Similarities: Instead of computing nearest neighbors using vantage-point trees, we approximate nearest neighbors using the [Annoy](https://github.com/spotify/annoy) library. The neighbor lookups are multithreaded to take advantage of machines with multiple cores. Using "near" neighbors as opposed to strictly "nearest" neighbors is faster, but also has a smoothing effect, which can be useful for embedding some datasets (see [Linderman et al. (2017)](https://arxiv.org/abs/1711.04712)). If subtle detail is required (e.g. in identifying small clusters), then use vantage-point trees (which is also multithreaded in this implementation). 


Check out our [paper](https://www.nature.com/articles/s41592-018-0308-4) or [preprint](https://arxiv.org/abs/1712.09005) for more details and some benchmarks.

## Features
Additionally, this implementation includes the following features:
* Early exaggeration: In [Linderman and Steinerberger (2017)](https://arxiv.org/abs/1706.02582), we showed that appropriately choosing the early exaggeration coefficient can lead to improved embedding of swissrolls and other synthetic datasets. Early exaggeration is built into all t-SNE implementations; here we highlight its importance as a parameter. 
* Late exaggeration: Increasing the exaggeration coefficient late in the optimization process can improve separation of the clusters. [Kobak and Berens (2018)](https://www.biorxiv.org/content/10.1101/453449v1) suggest starting late exaggeration immediately following early exaggeration. 
* Initialization: Custom initialization can be provided from Python/Matlab/R. As suggested by [Kobak and Berens (2018)](https://www.biorxiv.org/content/10.1101/453449v1), initializing t-SNE with the first two principal components (scaled to have standard deviation 0.0001) results in an embedding which often preserves the global structure more effectively than the default random normalization. See there for other initialisation tricks.
* Variable degrees of freedom: [Kobak et al. (2019)](https://arxiv.org/abs/1902.05804) show that decreasing the degree of freedom (df) of the t-distribution (resulting in heavier tails)  reveals fine structure that is not visible in standard t-SNE.
* Perplexity combination: The perplexity parameter determines the width of the Gaussian kernel, such that small perplexity values uncover local structure while larger values reveal global structure. [Kobak and Berens (2018)](https://www.biorxiv.org/content/10.1101/453449v1) show that using combination of several perplexity values, resulting in a multi-scale embedding, can be useful. 
* All optimisation parameters can be controlled from Python/Matlab/R. For example, [Belkina et al. (2018)](https://www.biorxiv.org/content/10.1101/451690v2) highlight the importance of increasing the learning rate when embedding large data sets. 


## Installation
R, Matlab, and Python wrappers are `fast_tsne.R`, `fast_tsne.m`, and `fast_tsne.py` respectively. Each of these wrappers can be used after installing FFTW and compiling the C++ code, as below. [Gioele La Manno](https://twitter.com/GioeleLaManno) implemented a Python (Cython) wrapper, which is available on PyPI [here](https://pypi.python.org/pypi/fitsne).

**Note:** If you update to a new version of FIt-SNE using `git pull`, be sure to recompile. 

### OSX and Linux
The only prerequisite is [FFTW](http://www.fftw.org/), which can be downloaded and installed from the website. Then, from the root directory compile the code as:
```bash
g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_tsne -pthread -lfftw3 -lm
```
See [here](https://github.com/KlugerLab/FIt-SNE/issues/35) for instructions in case one does not have `sudo` rights (one can install `FFTW` in the home directory and provide its path to `g++`).

Check out `examples/` for usage. The [Python demo notebook](https://github.com/KlugerLab/FIt-SNE/blob/master/examples/test.ipynb) walks one through the most of the available options using MNIST data set.


### Windows
A Windows binary is available [here](https://github.com/KlugerLab/FIt-SNE/releases/download/v1.1.0/FItSNE-Windows-1.1.0.zip). Please extract to the `bin/` folder, and you should be all set.

If you would like to compile it yourself see below. The code has been currently tested with MS Visual Studio 2015 (i.e., MS Visual Studio Version 14).

1.  First open the provided FItSNE solution (FItSNE.sln) using MS Visual Studio and build the Release configuration. 
2.  Copy the binary file ( e.g. `x64/Release/FItSNE.exe`) generated by the build process to the `bin/` folder 
3.  For Windows, we have added all dependencies, including the [FFTW library](http://www.fftw.org/), which is distributed under the GNU General Public License. For the binary to find the FFTW DLLs, you need to either add `src/winlibs/fftw/` to your PATH, or to copy the DLLs into the `bin/` directory.

As of this commit, only the R wrapper properly calls the Windows executable. The Python and Matlab wrappers can be trivially changed to call it (just changing `bin/fast_tsne` to `bin/FItSNE.exe` in the code), and will be changed in future commits.
 
Many thanks to [Josef Spidlen](https://github.com/jspidlen) for this Windows implementation!

## Acknowledgements and References
We are grateful for members of the community who have [contributed](https://github.com/KlugerLab/FIt-SNE/graphs/contributors) to improving FIt-SNE, especially [Dmitry Kobak](https://github.com/dkobak), [Pavlin Poličar](https://github.com/pavlin-policar), and [Josef Spidlen](https://github.com/jspidlen).

If you use FIt-SNE, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nature Methods. ([link](https://www.nature.com/articles/s41592-018-0308-4))

Our implementation is derived from the Barnes-Hut implementation:

Laurens van der Maaten (2014). Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research, 15(1):3221–3245. ([link](https://dl.acm.org/citation.cfm?id=2627435.2697068))

