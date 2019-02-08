# Note: this script should be sourced as: source('<path to file>', chdir=T)

FAST_TSNE_SCRIPT_DIR <<-getwd() 
 
cat(sprintf("FIt-SNE R wrapper loading.\nFIt-SNE root directory was set to %s\n",  FAST_TSNE_SCRIPT_DIR))
# Compute FIt-SNE of a dataset
#       dims - dimensionality of the embedding. Default 2.
#       perplexity - perplexity is used to determine the
#           bandwidth of the Gaussian kernel in the input
#           space.  Default 30.
#       theta - Set to 0 for exact.  If non-zero, then will use either
#           Barnes Hut or FIt-SNE based on nbody_algo.  If Barnes Hut, then
#           this determins the accuracy of BH approximation.
#           Default 0.5.
#       max_iter - Number of iterations of t-SNE to run.
#           Default 1000.
#       fft_not_bh - if theta is nonzero, this determins whether to
#            use FIt-SNE or Barnes Hut approximation. Default is FIt-SNE.
#            set to be True for FIt-SNE
#       ann_not_vptree - use vp-trees (as in bhtsne) or approximate nearest neighbors (default).
#            set to be True for approximate nearest neighbors
#       exaggeration_factor - coefficient for early exaggeration
#           (>1). Default 12.
#       no_momentum_during_exag - Set to 0 to use momentum
#           and other optimization tricks. 1 to do plain,vanilla
#           gradient descent (useful for testing large exaggeration
#           coefficients)
#       stop_early_exag_iter - When to switch off early exaggeration.
#           Default 250.
#       start_late_exag_iter - When to start late
#           exaggeration. set to -1 to not use late exaggeration
#           Default -1.
#       late_exag_coeff - Late exaggeration coefficient.
#          Set to -1 to not use late exaggeration.
#           Default -1
#       nterms - If using FIt-SNE, this is the number of
#                      interpolation points per sub-interval
#       intervals_per_integer - See min_num_intervals              
#       min_num_intervals - Let maxloc = ceil(max(max(X)))
#           and minloc = floor(min(min(X))). i.e. the points are in
#           a [minloc]^no_dims by [maxloc]^no_dims interval/square.
#           The number of intervals in each dimension is either
#           min_num_intervals or ceil((maxloc -
#           minloc)/intervals_per_integer), whichever is
#           larger. min_num_intervals must be an integer >0,
#           and intervals_per_integer must be >0. Default:
#           min_num_intervals=50, intervals_per_integer =
#           1
#
#       sigma - Fixed sigma value to use when perplexity==-1
#            Default -1 (None)
#       K - Number of nearest neighbours to get when using fixed sigma
#            Default -30 (None)
#
#       initialization - N x no_dims array to intialize the solution
#            Default: None
#
#       load_affinities - 
#            If 1, input similarities are loaded from a file and not computed
#            If 2, input similarities are saved into a file.
#            If 0, affinities are neither saved nor loaded
#
#       perplexity_list - if perplexity==0 then perplexity combination will
#            be used with values taken from perplexity_list. Default: NULL
#       df - Degree of freedom of t-distribution, must be greater than 0.
#       Values smaller than 1 correspond to heavier tails, which can often 
#       resolve substructure in the embedding. See Kobak et al. (2019) for
#       details. Default is 1.0
#
fftRtsne <- function(X, 
		     dims=2, perplexity=30, theta=0.5,
		     check_duplicates=TRUE,
		     max_iter=1000,
		     fft_not_bh = TRUE,
		     ann_not_vptree = TRUE,
		     stop_early_exag_iter=250,
		     exaggeration_factor=12.0, no_momentum_during_exag=FALSE,
		     start_late_exag_iter=-1.0,late_exag_coeff=1.0,
             mom_switch_iter=250, momentum=.5, final_momentum=.8, learning_rate=200,
		     n_trees=50, search_k = -1,rand_seed=-1,
		     nterms=3, intervals_per_integer=1, min_num_intervals=50, 
		     K=-1, sigma=-30, initialization=NULL,
		     data_path=NULL, result_path=NULL,
		     load_affinities=NULL,
		     fast_tsne_path=NULL, nthreads=0, perplexity_list = NULL, 
                     get_costs = FALSE, df = 1.0,... ) {
        version_number = '1.1.0'

	if (is.null(fast_tsne_path)) {
		if(.Platform$OS.type == "unix") {
			fast_tsne_path = sprintf('%s/bin/fast_tsne', FAST_TSNE_SCRIPT_DIR )
		} else {
			fast_tsne_path = sprintf('%s/bin/FItSNE.exe', FAST_TSNE_SCRIPT_DIR)
		}
	}

	if (is.null(data_path)) {
		data_path <- tempfile(pattern='fftRtsne_data_', fileext='.dat')
	}
	if (is.null(result_path)) {
		result_path <- tempfile(pattern='fftRtsne_result_', fileext='.dat')
	}
	if (is.null(fast_tsne_path)) {
		fast_tsne_path <- system2('which', 'fast_tsne', stdout=TRUE)
	}
	fast_tsne_path <- normalizePath(fast_tsne_path)
	if (!file_test('-x', fast_tsne_path)) {
		stop(fast_tsne_path, " does not exist or is not executable; check your fast_tsne_path parameter")
	}

	is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol

	if (!is.numeric(theta) || (theta<0.0) || (theta>1.0) ) { stop("Incorrect theta.")}
	if (nrow(X) - 1 < 3 * perplexity) { stop("Perplexity is too large.")}
	if (!is.matrix(X)) { stop("Input X is not a matrix")}
	if (!(max_iter>0)) { stop("Incorrect number of iterations.")}
	if (!is.wholenumber(stop_early_exag_iter) || stop_early_exag_iter<0) { stop("stop_early_exag_iter should be a positive integer")}
	if (!is.numeric(exaggeration_factor)) { stop("exaggeration_factor should be numeric")}
	if (!is.numeric(df)) { stop("df should be numeric")}
	if (!is.wholenumber(dims) || dims<=0) { stop("Incorrect dimensionality.")}
	if (search_k == -1) {
       if (perplexity>0) {
          search_k = n_trees*perplexity*3
       } else if (perplexity==0) {
          search_k = n_trees*max(perplexity_list)*3
       } else { 
          search_k = n_trees*K
       }
    }

	if (fft_not_bh){
	  nbody_algo = 2;
	}else{
	  nbody_algo = 1;
	}

	if (is.null(load_affinities)) {
		load_affinities = 0;
	} else {
		if (load_affinities == 'load') {
			load_affinities = 1;
		} else if (load_affinities == 'save') {
			load_affinities = 2;
		} else {
			load_affinities = 0;
		}
	}
	
	if (ann_not_vptree){
	  knn_algo = 1;
	}else{
	  knn_algo = 2;
	}
	tX = c(t(X))

	f <- file(data_path, "wb")
	n = nrow(X);
	D = ncol(X);
	writeBin(as.integer(n), f,size= 4)
	writeBin( as.integer(D),f,size= 4)
	writeBin( as.numeric(theta), f,size= 8) #theta
	writeBin( as.numeric(perplexity), f,size= 8) #theta

    if (perplexity == 0) {
    	writeBin( as.integer(length(perplexity_list)), f, size=4)
	    writeBin( perplexity_list, f) 
    }

	writeBin( as.integer(dims), f,size=4) #theta
	writeBin( as.integer(max_iter),f,size=4)
	writeBin( as.integer(stop_early_exag_iter),f,size=4)
	writeBin( as.integer(mom_switch_iter),f,size=4)
	writeBin( as.numeric(momentum),f,size=8)
	writeBin( as.numeric(final_momentum),f,size=8)
	writeBin( as.numeric(learning_rate),f,size=8)
	writeBin( as.integer(K),f,size=4) #K
	writeBin( as.numeric(sigma), f,size=8) #sigma
	writeBin( as.integer(nbody_algo), f,size=4)  #not barnes hut
	writeBin( as.integer(knn_algo), f,size=4) 
	writeBin( as.numeric(exaggeration_factor), f,size=8) #compexag
	writeBin( as.integer(no_momentum_during_exag), f,size=4) 
	writeBin( as.integer(n_trees), f,size=4) 
	writeBin( as.integer(search_k), f,size=4) 
	writeBin( as.integer(start_late_exag_iter), f,size=4) 
	writeBin( as.numeric(late_exag_coeff), f,size=8) 
	
	writeBin( as.integer(nterms), f,size=4) 
	writeBin( as.numeric(intervals_per_integer), f,size=8) 
	writeBin( as.integer(min_num_intervals), f,size=4) 
	tX = c(t(X))
	writeBin( tX, f) 
	writeBin( as.integer(rand_seed), f,size=4) 
        writeBin(as.numeric(df), f, size=8)
	writeBin( as.integer(load_affinities), f,size=4) 
	if (! is.null(initialization)){ writeBin( c(t(initialization)), f) }		
        print(df)
	close(f) 

	flag= system2(command=fast_tsne_path, args=c(version_number,data_path, result_path, nthreads));
	if (flag != 0) {
		stop('tsne call failed');
	}
	f <- file(result_path, "rb")
	n <- readBin(f, integer(), n=1, size=4);
	d <- readBin(f, integer(), n=1, size=4);
	Y <- readBin(f, numeric(), n=n*d);
        Y <- t(matrix(Y, nrow=d));
        if (get_costs ) {
            tmp <- readBin(f, integer(), n=1, size=4);
            costs <- readBin(f, numeric(), n=max_iter,size=8);
            Yout <- list( Y=Y, costs=costs);
        }else {
            Yout <- Y;
        }
        close(f)
        file.remove(data_path)
        file.remove(result_path)
        return(Yout)
}





