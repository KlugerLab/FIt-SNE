source('fast_tsne.R')
iris_unique <- unique(iris) # Remove duplicates
X <- as.matrix(iris_unique[,1:4]) # Run TSNE

Y_fft <- fftRtsne(X);
plot(Y_fft,col=iris_unique$Species) # Plot the result

#Y_bh <- fftRtsne(X,fft_not_bh =   FALSE);  # compare with the bh method
#plot(Y_bh,col=iris_unique$Species) 

require(MASS);
N <- 1E4;
d <- 3;
input_data <- rbind(mvrnorm(n = N/2, rep(0, d), diag(d)),
	      mvrnorm(n = N/2, rep(100, d), diag(d))
	      )
Y2 <- fftRtsne(input_data,d,max_iter = 400, start_late_exag_iter=300, late_exag_coeff=10);
#plot(Y2[,1],runif(length(Y2)),col=c(rep(1,N/2), rep(2,N/2))) # Plot the result



Y2 <- (matrix(Y2, ncol=d));
plot(Y2[,1],Y2[,2],col=c(rep(1,N/d), rep(2,N/d))) # Plot the result
