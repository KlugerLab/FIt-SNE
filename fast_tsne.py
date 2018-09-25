# This is a really basic function that does not do almost any sanity checks
#
# It assumes that fast_tsne.py and fast_tsne binary are both located
# in the working directory.
#
# Usage example:
#	from fast_tsne import fast_tsne
#	X = np.random.randn(1000, 50)
#	Z = fast_tsne(X, perplexity = 30)
#
# Written by Dmitry Kobak


import os
import subprocess
import struct
import numpy as np

def fast_tsne(X, theta=.5, perplexity=30, map_dims=2, max_iter=1000, 
              stop_lying_iter=200, K=-1, sigma=-30, nbody_algo='FFT', knn_algo='annoy',
              early_exag_coeff=12, no_momentum_during_exag=0, n_trees=50, 
              search_k=None, start_late_exag_iter=-1, late_exag_coeff=-1,
              nterms=3, intervals_per_integer=1, min_num_intervals=50,            
              seed=-1, initialization=None, load_affinities=None,
              perplexity_list=None):

	# X should be a numpy array of 64-bit doubles
	X = np.array(X).astype(float)
    
    if search_k is None:
        if perplexity > 0:
            search_k = 3 * perplexity * n_trees
        elif perplexity == 0:
            search_k = 3 * np.max(perplexity_list) * n_trees
        else:
            search_k = 3 * K * n_trees
        
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
    
    # create temp directory if it does not exist
    if not os.path.isdir(os.getcwd()+'/temp'):
        os.mkdir(os.getcwd()+'/temp')
    
    # write data file
    with open(os.getcwd()+'/temp/data.dat', 'wb') as f:
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
        f.write(struct.pack('=i', stop_lying_iter))
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
        f.write(struct.pack('=i', load_affinities))

        if initialization is not None:
				initialization = np.array(initialization).astype(float)
                f.write(initialization.tobytes()) 
               
    # run t-sne
    subprocess.call(os.getcwd()+'/fast_tsne')
            
    # read data file
    with open(os.getcwd()+'/temp/result.dat', 'rb') as f:
        initError, = struct.unpack('=d', f.read(8))
        n, = struct.unpack('=i', f.read(4))  
        md, = struct.unpack('=i', f.read(4)) 
        sz = struct.calcsize('=d')
        buf = f.read()
        x_tsne = [struct.unpack_from('=d', buf, sz*offset) for offset in range(n*md)]
        x_tsne = np.array(x_tsne).reshape((n,md))

    return x_tsne
