# Note: the chdir=T argument to source() is necessary if running FIt-SNE outside of the the root directory of FIt-SNE

source('../fast_tsne.R', chdir=T)

# Using Iris dataset

iris_unique <- unique(iris) # Remove duplicates
X <- as.matrix(iris_unique[,1:4]) # Run TSNE

Y_fft <- fftRtsne(X);
plot(Y_fft,col=iris_unique$Species) # Plot the result


# And now using a toy dataset and d=1

require(MASS);
N <- 1E4;
d <- 3;
input_data <- rbind(mvrnorm(n = N/2, rep(0, d), diag(d)),
	                  mvrnorm(n = N/2, rep(100, d), diag(d)))
Y2 <- fftRtsne(input_data, 1, max_iter = 400, start_late_exag_iter=300, late_exag_coeff=10);
plot(Y2[,1],runif(length(Y2)),col=c(rep(1,N/2), rep(2,N/2))) # Plot the result
