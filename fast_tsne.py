# This is a really basic function that does not do almost any sanity checks
#
# Usage example:
#	import sys; sys.path.append('../')
#       from fast_tsne import fast_tsne
#       import numpy as np
#	X = np.random.randn(1000, 50)
#	Z = fast_tsne(X, perplexity = 30)
# Written by Dmitry Kobak


import os
import subprocess
import struct
import numpy as np

def fast_tsne(X, theta=.5, perplexity=30, map_dims=2, max_iter=1000, 
              stop_early_exag_iter=250, K=-1, sigma=-1, nbody_algo='FFT', knn_algo='annoy',
              mom_switch_iter=250, momentum=.5, final_momentum=.8, learning_rate=200,
              early_exag_coeff=12, no_momentum_during_exag=0, n_trees=50, 
              search_k=None, start_late_exag_iter=-1, late_exag_coeff=-1,
              nterms=3, intervals_per_integer=1, min_num_intervals=50,            
              seed=-1, initialization=None, load_affinities=None,
              perplexity_list=None, df=1):

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
    subprocess.call(os.path.dirname(os.path.realpath(__file__)) + '/bin/fast_tsne')
            
    # read data file
    with open(os.getcwd()+'/result.dat', 'rb') as f:
        n, = struct.unpack('=i', f.read(4))  
        md, = struct.unpack('=i', f.read(4)) 
        sz = struct.calcsize('=d')
        buf = f.read()
        x_tsne = [struct.unpack_from('=d', buf, sz*offset) for offset in range(n*md)]
        x_tsne = np.array(x_tsne).reshape((n,md))

    return x_tsne
