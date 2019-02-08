# This is a really basic function that does not do almost any sanity checks
#
# Usage example:
#	import sys; sys.path.append('../')
#   from fast_tsne import fast_tsne
#   import numpy as np
#	X = np.random.randn(1000, 50)
#	Z = fast_tsne(X, perplexity = 30)
#
# Written by Dmitry Kobak


import os
import subprocess
import struct
import numpy as np

def fast_tsne(X, theta=.5, perplexity=30, map_dims=2, max_iter=1000, 
              stop_early_exag_iter=250, K=-1, sigma=-1, nbody_algo='FFT', knn_algo='annoy',
              mom_switch_iter=250, momentum=.5, final_momentum=.8, learning_rate=200,
              early_exag_coeff=12, no_momentum_during_exag=False, n_trees=50, 
              search_k=None, start_late_exag_iter=-1, late_exag_coeff=-1,
              nterms=3, intervals_per_integer=1, min_num_intervals=50,            
              seed=-1, initialization=None, load_affinities=None,
              perplexity_list=None, df=1, return_loss=False, nthreads=None):
    """Run t-SNE. This implementation supports exact t-SNE, Barnes-Hut t-SNE and FFT-accelerated
    interpolation-based t-SNE (FIt-SNE). This is a Python wrapper to a C++ executable.

    Parameters
    ----------
    X: 2D numpy array
        Array of observations (n times p)
    perplexity: double
        Perplexity is used to determine the bandwidth of the Gaussian kernel in the input
        space.  Default 30.
    theta: double
        Set to 0 for exact t-SNE. If non-zero, then the code will use either Barnes Hut
        or FIt-SNE based on `nbody_algo`.  If Barnes Hut, then theta determins the accuracy of 
        BH approximation.   Default 0.5.
    map_dims: int
        Number of embedding dimensions. Default 2. FIt-SNE supports only 1 or 2 dimensions.
    max_iter: int
        Number of gradient descent iterations. Default 1000.
    nbody_algo: {'Barnes-Hut', 'FFT'}
        If theta is nonzero, this determins whether to use FIt-SNE or Barnes Hut approximation.
        Default is 'FFT'.
    knn_algo: {'vp-tree', 'annoy'}
        Use exact nearest neighbours with VP trees (as in BH t-SNE) or approximate nearest neighbors
        with Annoy. Default is 'annoy'.
    early_exag_coeff: double
        Coefficient for early exaggeration. Default 12.
    stop_early_exag_iter: int
        When to switch off early exaggeration. Default 250.
    late_exag_coeff: double
        Coefficient for late exaggeration. Set to -1 in order not to use late exaggeration.
        Default -1.
    start_late_exag_iter:
        When to start late exaggeration. Set to -1 in order not to use late exaggeration.
        Default -1.
    momentum: double
        Initial value of momentum. Default 0.5.
    final_momentum: double
        The value of momentum to use later in the optimisation. Default 0.8.
    mom_switch_iter: int
        Iteration number to switch from momentum to final_momentum. Default 250.
    learning_rate: double
        Learning rate. Default 200.
    no_mometum_during_exag: boolean
        Whether to switch off momentum during the early exaggeration phase (can be useful
        for experiments with large exaggeration coefficients). Default is False.
    sigma: boolean
        The standard deviation of the Gaussian kernel to be used for all points instead of 
        choosing it adaptively via perplexity. Set to -1 to use perplexity. Default is -1.
    K: int
        The number of nearest neighbours to use when using fixed sigma instead of perplexity
        calibration. Set to -1 when perplexity is used. Default is -1.
    nterms: int
        If using FIt-SNE, this is the number of interpolation points per sub-interval
    intervals_per_integer: double
        See min_num_intervals              
    min_num_intervals: int
        The interpolation grid is chosen on each step of the gradient descent. If Y is the current
        embedding, let maxloc = ceiling(max(Y.flatten)) and minloc = floor(min(Y.flatten)), i.e. 
        the points are contained in a [minloc, maxloc]^no_dims box. The number of intervals in each 
        dimension is either min_num_intervals or ceiling((maxloc-minloc)/intervals_per_integer),
        whichever is larger. min_num_intervals must be a positive integer and intervals_per_integer 
        must be positive real value. Defaults: min_num_intervals=50, intervals_per_integer = 1.
    n_trees: int
        When using Annoy, the number of search trees to use. Default is 50.
    search_k: int
        When using Annoy, the number of nodes to inspect during search. Default is 3*perplexity*n_trees
        (or K*n_trees when using fixed sigma).
    seed: int
        Seed for random initialisation. Use -1 to initialise random number generator with current time.
        Default -1.
    initialization: numpy aray
         N x no_dims array to intialize the solution. Default: None.
    load_affinities: {'load', 'save', None}
        If 'save', input similarities (p_ij) are saved into a file. If 'load', they are loaded from a file
        and not recomputed. If None, they are not saved and not loaded. Default is None.
    perplexity_list: list
        A list of perplexities to used as a perplexity combination. Input affinities are computed with each
        perplexity on the list and then averaged. Default is None.
    nthreads: int
        Number of threads to use. Default is None (i.e. use all available threads).
    df: double
        Controls the degree of freedom of t-distribution. Must be positive. The actual degree of
        freedom is 2*df-1. The standard t-SNE choice of 1 degree of freedom corresponds to df=1.
        Large df approximates Gaussian kernel. df<1 corresponds to heavier tails, which can often 
        resolve substructure in the embedding. See Kobak et al. (2019) for details. Default is 1.0.    
    return_loss: boolean
        If True, the function returns the loss values computed during optimisation 
        together with the final embedding. If False, only the embedding is returned.
        Default is False.

    Returns
    -------
    Y: numpy array
        The embedding.
    loss: numpy array
        Loss values computed during optimisation. Only returned if return_loss is True.
    """

    version_number = '1.1.0'


    # X should be a numpy array of 64-bit doubles
    X = np.array(X).astype(float)

    if perplexity_list is not None:
        perplexity = 0                # C++ requires perplexity=0 in order to use perplexity_list

    if sigma > 0 and K > 0:
        perplexity = -1               # C++ requires perplexity=-1 in order to use sigma

    if search_k is None:
        if perplexity > 0:
            search_k = 3 * perplexity * n_trees
        elif perplexity == 0:
            search_k = 3 * np.max(perplexity_list) * n_trees
        else:
            search_k = K * n_trees
        
    if nbody_algo == 'Barnes-Hut':
        nbody_algo = 1
    else:
        nbody_algo = 2
        
    if knn_algo == 'vp-tree':
        knn_algo = 2
    else:
        knn_algo = 1

    if load_affinities == 'load':
        load_affinities = 1
    elif load_affinities == 'save':
        load_affinities = 2
    else:
        load_affinities = 0
    
    if nthreads is None:
        nthreads = 0

    if no_momentum_during_exag:
        no_momentum_during_exag = 1
    else:
        no_momentum_during_exag = 0
    
    # write data file
    with open(os.getcwd() + '/data.dat', 'wb') as f:
        n, d = X.shape
        f.write(struct.pack('=i', n))   
        f.write(struct.pack('=i', d))   
        f.write(struct.pack('=d', theta))
        f.write(struct.pack('=d', perplexity))
        if perplexity==0:
            f.write(struct.pack('=i', len(perplexity_list)))
            for perpl in perplexity_list:
                f.write(struct.pack('=d', perpl))
        f.write(struct.pack('=i', map_dims))
        f.write(struct.pack('=i', max_iter))
        f.write(struct.pack('=i', stop_early_exag_iter))
        f.write(struct.pack('=i', mom_switch_iter))
        f.write(struct.pack('=d', momentum))
        f.write(struct.pack('=d', final_momentum))
        f.write(struct.pack('=d', learning_rate))
        f.write(struct.pack('=i', K))
        f.write(struct.pack('=d', sigma))
        f.write(struct.pack('=i', nbody_algo))
        f.write(struct.pack('=i', knn_algo))
        f.write(struct.pack('=d', early_exag_coeff))
        f.write(struct.pack('=i', no_momentum_during_exag))
        f.write(struct.pack('=i', n_trees))
        f.write(struct.pack('=i', search_k))
        f.write(struct.pack('=i', start_late_exag_iter))
        f.write(struct.pack('=d', late_exag_coeff))
        f.write(struct.pack('=i', nterms))
        f.write(struct.pack('=d', intervals_per_integer))
        f.write(struct.pack('=i', min_num_intervals))
        f.write(X.tobytes()) 
        f.write(struct.pack('=i', seed))
        f.write(struct.pack('=d', df))
        f.write(struct.pack('=i', load_affinities))

        if initialization is not None:
                initialization = np.array(initialization).astype(float)
                f.write(initialization.tobytes()) 
               
    # run t-sne
    subprocess.call([os.path.dirname(os.path.realpath(__file__)) + 
        '/bin/fast_tsne', version_number, 'data.dat', 'result.dat', '{}'.format(nthreads)])
            
    # read data file
    with open(os.getcwd()+'/result.dat', 'rb') as f:
        n, = struct.unpack('=i', f.read(4))  
        md, = struct.unpack('=i', f.read(4)) 
        sz = struct.calcsize('=d')
        buf = f.read(sz*n*md)
        x_tsne = [struct.unpack_from('=d', buf, sz*offset) for offset in range(n*md)]
        x_tsne = np.array(x_tsne).reshape((n,md))
        _, = struct.unpack('=i', f.read(4))  
        buf = f.read(sz*max_iter)
        loss = [struct.unpack_from('=d', buf, sz*offset) for offset in range(max_iter)]
        loss = np.array(loss).squeeze()
        loss[np.arange(1,max_iter+1)%50>0] = np.nan

    if return_loss:
        return (x_tsne, loss)
    else:
        return x_tsne


