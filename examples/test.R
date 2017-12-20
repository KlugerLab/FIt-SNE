source('fast_tsne.R')
iris_unique <- unique(iris) # Remove duplicates
X <- as.matrix(iris_unique[,1:4]) # Run TSNE
Y <- fftRtsne(X);

plot(Y,col=iris_unique$Species) # Plot the result



#Sigma <- matrix(c(10,3, 3,
#                3,2, 3,
#                3, 3, 5),3,3)

require(MASS);
N <- 1E4;
d <- 3;
input_data <- rbind(mvrnorm(n = N/2, rep(0, d), diag(d)),
	      mvrnorm(n = N/2, rep(100, d), diag(d))
	      )
Y2 <- fftRtsne(input_data,1,max_iter = 400, no_momentum_during_exag=0, start_late_exag_iter=300, late_exag_coeff=10);
plot(Y2[,1],Y2[,1],col=c(rep(1,N/2), rep(2,N/2))) # Plot the result



Y2 <- (matrix(Y, ncol=d));
plot(Y2[,1],Y2[,2],col=c(rep(1,N/2), rep(2,N/2))) # Plot the result
