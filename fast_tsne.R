
fftRtsne <- function(X, 
		     dims=2, perplexity=30, theta=0.5,
		     check_duplicates=TRUE,
		     max_iter=1000,
		     fmm_not_bh = TRUE,
		     stop_lying_iter=250,
		     exaggeration_factor=12.0, no_momentum_during_exag=TRUE,
		     start_late_exag_iter=-1.0,late_exag_coeff=1.0,
		     n_trees=50, search_k = -1, ...) {

	is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol

	if (!is.numeric(theta) || (theta<0.0) || (theta>1.0) ) { stop("Incorrect theta.")}
	if (nrow(X) - 1 < 3 * perplexity) { stop("Perplexity is too large.")}
	if (!is.matrix(X)) { stop("Input X is not a matrix")}
	if (!(max_iter>0)) { stop("Incorrect number of iterations.")}
	if (!is.wholenumber(stop_lying_iter) || stop_lying_iter<0) { stop("stop_lying_iter should be a positive integer")}
	if (!is.numeric(exaggeration_factor)) { stop("exaggeration_factor should be numeric")}
	if (!is.wholenumber(dims) || dims<=0) { stop("Incorrect dimensionality.")}
	if (search_k == -1) { search_k = n_trees*perplexity*3 }

	tX = c(t(X))

	f <- file("temp/data.dat", "wb")
	n = nrow(X);
	D = ncol(X);
	writeBin(as.integer(n), f,size= 4)
	writeBin( as.integer(D),f,size= 4)
	writeBin( as.numeric(0.5), f,size= 8) #theta
	writeBin( as.numeric(perplexity), f,size= 8) #theta
	writeBin( as.integer(dims), f,size=4) #theta
	writeBin( as.integer(max_iter),f,size=4)
	writeBin( as.integer(stop_lying_iter),f,size=4)
	writeBin( as.integer(-1),f,size=4) #K
	writeBin( as.numeric(-30.0), f,size=8) #sigma
	writeBin( as.integer(2), f,size=4)  #not barnes hut
	writeBin( as.integer(2), f,size=4) #compexag
	writeBin( as.numeric(exaggeration_factor), f,size=8) #compexag
	writeBin( as.integer(no_momentum_during_exag), f,size=4) 
	writeBin( as.integer(n_trees), f,size=4) 
	writeBin( as.integer(search_k), f,size=4) 
	writeBin( as.integer(start_late_exag_iter), f,size=4) 
	writeBin( as.numeric(late_exag_coeff), f,size=8) 
	tX = c(t(X))
	writeBin( tX, f) 
	close(f) 


	flag= system('bin/fast_tsne');
	if (flag != 0) {
		stop('tsne call failed');
	}
	f <- file("temp/result.dat", "rb")
	initialError <- readBin(f, integer(), n=1, size=8);
	n <- readBin(f, integer(), n=1, size=4);
	d <- readBin(f, integer(), n=1,size=4);
	Y <- readBin(f, numeric(), n=n*d);
	Yout <- t(matrix(Y, nrow=d));
	close(f)
	Yout

}





