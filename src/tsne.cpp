/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */



#include <iostream>
#include <time.h>
#include <fstream>
#include "nbodyfft.h"
#include <fftw3.h>
#include <math.h>
#include "annoylib.h"
#include "kissrandom.h"
#include <thread>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include "vptree.h"
#include "sptree.h"
#include "tsne.h"
#include <unistd.h>
#include <sstream>
#include <functional>

int itTest = 0;
bool measure_accuracy = false;

double cauchy(double x, double y, double bandx, double bandy ){
	return pow(1.0/(double) (1.0+pow(x-y,2)),2);
}
double cauchy2d(double x1,double x2,  double y1,double y2, double bandx, double bandy ){
	return pow(1.0/(double) (1.0+pow(x1-y1,2) + pow(x2-y2,2)),2);
	//return 1.0/(double) (1.0+pow(x-y,2));
}


using namespace std;

//Helper function for printing Y at each iteration. Useful for debugging
void print_progress ( int iter, double * Y, int N, int no_dims){

	ofstream myfile;
	std::ostringstream stringStream;
	stringStream << "dat/intermediate" << iter << ".txt";
	std::string copyOfStr = stringStream.str();
	myfile.open (stringStream.str().c_str());
	for (int j=0; j< N;j++){
		for (int i = 0; i<no_dims; i++) {
			myfile << Y[j*no_dims + i] << " ";
		}
		myfile << "\n";
	}
	myfile.close();

}

// Perform t-SNE
int TSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
		bool skip_random_init, int max_iter, int stop_lying_iter, 
		int mom_switch_iter, int K, double sigma, int nbody_algo, int knn_algo, double early_exag_coeff,double * initialError, double * costs, bool no_momentum_during_exag,
		int start_late_exag_iter, double late_exag_coeff, int n_trees,int search_k,
		int nterms, double intervals_per_integer, int min_num_intervals, unsigned int nthreads) {

	// Set random seed
	if (skip_random_init != true) {
		if(rand_seed >= 0) {
			printf("Using random seed: %d\n", rand_seed);
			srand((unsigned int) rand_seed);
		} else {
			printf("Using current time as random seed...\n");
			srand(time(NULL));
		}
	}

	// Determine whether we are using an exact algorithm
	if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }

	if (no_momentum_during_exag) {
		printf("No momentum during the exaggeration phase\n");
	}else{
		printf("Will use momentum during exaggeration phase\n");
	}
	printf("Using no_dims = %d, max_iter = %d, perplexity = %f, theta = %f, K = %d, Sigma = %lf, knn_algo = %d, early_exag_coeff = %f, data[0] = %lf", no_dims, max_iter, perplexity, theta, K, sigma, knn_algo, early_exag_coeff, X[0]);

	bool exact = (theta == .0) ? true : false;

	// Set learning parameters
	float total_time = .0;
	clock_t start, end;
	double momentum = .5, final_momentum = .8;

	//step size
	double eta = 200;

	// Allocate some memory
	double* dY    = (double*) malloc(N * no_dims * sizeof(double));
	double* uY    = (double*) malloc(N * no_dims * sizeof(double));
	double* gains = (double*) malloc(N * no_dims * sizeof(double));
	if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
	for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

	printf("Computing input similarities...\n");
	//struct timespec start_timespec, finish_timespec;
	//clock_gettime(CLOCK_MONOTONIC, &start_timespec);
	zeroMean(X, N, D);
	if (perplexity >0) {
		printf( "Using perplexity, so normalize input data (to prevent numerical problems)\n");
		double max_X = .0;
		for(int i = 0; i < N * D; i++) {
			if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
		}
		for(int i = 0; i < N * D; i++) X[i] /= max_X;
	}else{
		printf("No perplexity, so do not normalize.");
	}

	// Compute input similarities for exact t-SNE
	double* P; unsigned int* row_P; unsigned int* col_P; double* val_P;
	if(exact) {

		// Compute similarities
		printf("Theta set to 0, so running exact algorithm");
		P = (double*) malloc(N * N * sizeof(double));
		if(P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

		computeGaussianPerplexity(X, N, D, P, perplexity, sigma);

		// Symmetrize input similarities
		printf("Symmetrizing...\n");
		int nN = 0;
		for(int n = 0; n < N; n++) {
			int mN = (n + 1) * N;
			for(int m = n + 1; m < N; m++) {
				P[nN + m] += P[mN + n];
				P[mN + n]  = P[nN + m];
				mN += N;
			}
			nN += N;
		}
		double sum_P = .0;
		for(int i = 0; i < N * N; i++) sum_P += P[i];
		for(int i = 0; i < N * N; i++) P[i] /= sum_P;
		//sum_P is just a cute way of writing 2N
		printf("Finished exact calculation of the P.  Sum_p: %lf \n", sum_P);
	}

	// Compute input similarities for approximate t-SNE
	else {
		// Compute asymmetric pairwise input similarities
		if (perplexity < 0 ) {
			printf("Using manually set kernel width\n");
			computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, K, sigma, nthreads);
		}else {
			printf("Using perplexity, not the manually set kernel width.  K (number of nearest neighbors) and sigma (bandwidth) parameters are going to be ignored.\n");
			if (knn_algo == 1) {
				printf("Using ANNOY for knn search, with parameters: n_trees %d and search_k%d\n", n_trees, search_k);
				int error_code = 0;
				error_code = computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity), -1, n_trees, search_k, nthreads);
				if (error_code <0) return error_code;
			}else if (knn_algo == 2){
				printf("Using vp trees for nearest neighbor search\n");
				computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity), -1, nthreads);

			}else{
				printf("Invalid knn_algo param\n");
				free(dY);
				free(uY);
				free(gains);
				exit(1);
			}
		}

		// Symmetrize input similarities
		symmetrizeMatrix(&row_P, &col_P, &val_P, N);
		double sum_P = .0;
		for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
		for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
	}
	end = clock();


	// Initialize solution (randomly)
	if (skip_random_init != true) {
		printf("Initializing the solution\n");
		for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
	}else{
		printf("Using the given initialization\n");
	}

	// If we are doing early exaggeration, we premultiply all the P by the coefficient of early exaggeration, compexagcoef
	double max_sum_cols = 0;
	//Compute maximum possible exaggeration coefficient, if user requests
	if (early_exag_coeff == 0 ) {
		for(int n = 0; n < N; n++) {
			double running_sum = 0;
			for(int i = row_P[n]; i < row_P[n + 1]; i++) {
				running_sum += val_P[i];
				//printf("val_P[i] = %lf\n", val_P[i]);
			}
			if (running_sum > max_sum_cols ) max_sum_cols = running_sum;
		}
		early_exag_coeff = (1.0/(eta*max_sum_cols) );
		//early_exag_coeff = (1.0/(max_sum_cols) );
		printf("Max of the val_Ps is: %lf\n", max_sum_cols);
	}

	printf("Exagerating Ps by %f\n", early_exag_coeff);
	if(exact) { 
		for(int i = 0; i < N * N; i++)   { 
			P[i] *= early_exag_coeff;
		}
	}else{ 
		for(int i = 0; i < row_P[N]; i++)
			val_P[i] *= early_exag_coeff;
	} 

	print_progress (0, Y, N, no_dims);

	// Perform main training loop

	//clock_gettime(CLOCK_MONOTONIC, &finish_timespec);
	//double elapsed_input = (finish_timespec.tv_sec - start_timespec.tv_sec);
	//printf("Input similarities learned in %lf seconds\n", elapsed_input);

	if(exact) printf("Input similarities computed \nLearning embedding...\n");
	else printf("Input similarities computed (sparsity = %f)!\nLearning embedding...\n", (double) row_P[N] / ((double) N * (double) N));

	start = clock();
	if (!exact) {
		if (nbody_algo == 2) {
			printf("Using FIt-SNE approximation.\n");
		}else if(nbody_algo == 1) {
			printf("Using the Barnes Hut approximation.\n");
		}else{
			printf("Undefined algo"); exit(2);
		}
	}

	for(int iter = 0; iter < max_iter; iter++) {
		itTest = iter;

		if(exact){
			computeExactGradient(P, Y, N, no_dims, dY);
		}else{
			if (nbody_algo == 2) {
				if (no_dims == 1){
					computeFftGradientOneD(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta,nterms,intervals_per_integer, min_num_intervals);
				}else{
					computeFftGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta,nterms,intervals_per_integer, min_num_intervals);
				}
			}else if(nbody_algo == 1) {
				computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);
			}
		}
		if (measure_accuracy){
			computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);
			computeFftGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta,nterms,intervals_per_integer, min_num_intervals);
			computeExactGradientTest(Y, N, no_dims);
		}

		//some diagnostic output
		//for(int i = 0; i < 10 ; i++) {
		//printf("%d,truth: %le, Estimate: %le\n", i,dYExact[i], dY[i]);
		//printf("dY[%d]: %le\n", i, dY[i]);
		//}

		//User can specify to turn off momentum/gains until after the early exaggeration phase is completed
		if ( no_momentum_during_exag ) {
			if ( iter > stop_lying_iter ) {
				for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
				for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;
				for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
				for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
			}else {
				//During early exaggeration or compression, no trickery (i.e. no momentum, or gains).  Just good old fashion gradient descent
				for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] - dY[i];
			}
		}else{
			for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
			for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;
			for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
			for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
		}

		// Make solution zero-mean
		zeroMean(Y, N, no_dims);

		//Switch off early exaggeration
		if(iter == stop_lying_iter) {
			printf("Unexaggerating Ps by %f\n", early_exag_coeff);
			if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= early_exag_coeff; }
			else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= early_exag_coeff; }
		}
		if(iter == start_late_exag_iter) {
			printf("Exaggerating Ps by %f\n", late_exag_coeff);
			if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= late_exag_coeff; }
			else      { for(int i = 0; i < row_P[N]; i++) val_P[i] *= late_exag_coeff; }
		}
		if(iter == mom_switch_iter) momentum = final_momentum;

		// Print out progress
		if ((iter % 50 == 0 || iter == max_iter - 1)) {
			clock_t end = clock();
			double C = .0;
			if(exact) C = evaluateError(P, Y, N, no_dims);
			//else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
			costs[iter] = C;
			if(iter > 0){
				total_time += (float) (end - start) / CLOCKS_PER_SEC;
				printf("Iteration %d (50 iterations in %f seconds)\n", iter, (float) (end - start) / CLOCKS_PER_SEC);
			}
			//print_progress (iter, Y, N, no_dims);
			start = clock();
		}
	}
	end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);
	if(exact) free(P);
	else {
		free(row_P); row_P = NULL;
		free(col_P); col_P = NULL;
		free(val_P); val_P = NULL;
	}
	/*
	printf("N-body phase completed in %4.2f seconds.\n", total_time);
	   FILE * fp = fopen( "temp/time_results.txt", "a" ); // Open file for writing
	   if (fp != NULL) {
	   fprintf(fp,"vptree8, %d, %d, %f, %f\n", N, D, elapsed_input, total_time);
	   fclose(fp);
	   }
	   */
	return 0;
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*) calloc(N * D, sizeof(double));
	double* neg_f = (double*) calloc(N * D, sizeof(double));
	if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
	for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	FILE * fp;
	if (measure_accuracy){
		char buffer[500];
		sprintf(buffer,"temp/bh_gradient%d.txt", itTest);
		fp = fopen( buffer, "w" ); // Open file for writing
	}
	for(int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
		if (measure_accuracy) {
			if (i < N) 
				//fprintf(fp,"%d,  %lf\n",i, neg_f[i*2]/sum_Q);
				fprintf(fp,"%d, %.12e, %.12e, %.12e,%.12e,%.12e  %.12e\n",i,dC[i*2],dC[i*2+1], pos_f[i*2],pos_f[i*2+1], neg_f[i*2]/sum_Q, neg_f[i*2+1]/sum_Q);
		}
	}
	if (measure_accuracy) {
		fclose(fp);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

void TSNE::computeFftGradientOneD(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, int  nterms, double  intervals_per_integer, int min_num_intervals) {
	//Zero out the gradient
	for(int i = 0; i < N * D; i++){
		dC[i] = 0.0;
	}

	int ndim = 3; //Number of sets of charges.

	double * locs =(double*) malloc(N* sizeof(double));
	double * chargesQij =(double*) malloc(ndim*N* sizeof(double));
	double * potentialQij =(double*) malloc(ndim*N* sizeof(double));

	//Push all the points at which we will evaluate
	//Y is stored row major, with a row corresponding to a single point
	double rmin = 1E5;
	double rmax = -1E5;
	for (unsigned long i = 0; i < N; i++) {
		locs[i] = Y[i];
		if (Y[i] < rmin) rmin = Y[i];
		if (Y[i] > rmax) rmax = Y[i];
	}
	int nboxes =fmax(min_num_intervals, (rmax-rmin)/(double) intervals_per_integer);
	//printf("%d nodes, from %lf to %lf, so using nboxes=%d\n",nterms, rmin, rmax, nboxes);

	for (unsigned long j = 0; j < N; j++) {
		chargesQij[j] =  1;
		chargesQij[1*N+j] =  Y[j];
		chargesQij[2*N+j] =  Y[j]*Y[j];
	} 
	for (unsigned long i = 0; i < N; i++) {
		//printf("loc: %f, %f, %f, %f\n", locs[i], chargesQij[i], chargesQij[1*N+i], chargesQij[2*N+i]);
	}
	clock_t begin = clock();
	double * band = (double *) calloc(N,sizeof(double));
	kerneltype kernel = &cauchy;

	double * boxl =(double*) malloc(nboxes* sizeof(double));
	double * boxr =(double*) malloc(nboxes* sizeof(double));
	double *prods =(double*) malloc(nterms* sizeof(double));
	double *xpts =(double*) malloc(nterms* sizeof(double));
	double *xptsall =(double*) calloc(nterms*nboxes, sizeof(double));
	fftw_complex * zkvalf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*nterms*nboxes);

	precompute(rmin, rmax,nboxes, nterms, &cauchy, band,boxl, boxr,  prods, xpts, xptsall,zkvalf );
	nbodyfft(N,ndim,locs,chargesQij, nboxes, nterms,boxl, boxr,  prods, xpts, xptsall,zkvalf,potentialQij);
	free(band); free(boxl); free(boxr); free(prods); free(xpts); free(xptsall);
	fftw_free(zkvalf);

	double zSum = 0;
	for (unsigned long i = 0; i < N; i++) {
		double phi1 =  potentialQij[i];
		double phi2 =  potentialQij[1*N+i];
		double phi3 =  potentialQij[2*N+i];
		double Y_i = Y[i];

		zSum += (1 + Y_i*Y_i )*phi1 - 2*(Y_i*phi2 ) + phi3;
	}
	zSum -= N;
	//printf("zSum from the new calc is %le\n\n", zSum2);

	//Now, figure out the Gaussian component of the gradient.  This
	//coresponds to the "attraction" term of the gradient.  It was
	//calculated using a fast KNN approach, so here we just use the results
	//that were passed to this function
	clock_t startTime = clock();
	unsigned int ind2 =0;
	double * pos_f;
	pos_f = new double[N];
	// Loop over all edges in the graph
	double r, q_ij,Y_diff;
	for(unsigned int n = 0; n < N; n++) {
		pos_f [n] = 0;
		for(unsigned int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {
			// Compute pairwise distance and Q-value
			ind2 = inp_col_P[i];
			Y_diff = Y[n] - Y[ind2];
			q_ij = 1/(1+ Y_diff * Y_diff);
			pos_f [n] += inp_val_P[i] * q_ij * Y_diff;
		}
	}

	//cout << "->Attraction term" << ((double) clock() -startTime)/(double) CLOCKS_PER_SEC << endl;

	startTime = clock();
	//Make the negative term, or F_rep in the equation 3 of the paper
	double * neg_f;
	neg_f = new double[N];
	for(unsigned int n = 0; n < N; n++) {

		double Qij_y_i,Qij_y_j,Qij_y;

		Qij_y_i = Y[n] *potentialQij[n ];
		Qij_y_j = potentialQij[1*N +n ];

		Qij_y = Qij_y_i - Qij_y_j;

		//Note that we only use the Z normalization term in the F_rep,
		//because it cancels in the F_attr.  Also, note that we divide
		//it, because the denominator of q_ij^2  is Z^2, so it cancels
		//out the Z in the numerater for Equation 3
		neg_f [n] = Qij_y/zSum;

		dC[n] = pos_f[n] - neg_f[n];
	}
	for (unsigned long i = 0; i < N; i++) {
		//	printf("pos_f %f,neg_f %f\n", pos_f[i], neg_f[i]);
	}
	//cout << "->Negative term" << ((double) clock() -startTime)/(double) CLOCKS_PER_SEC << endl;

	free(potentialQij); 
	free(locs); 
	delete[] pos_f;
	delete[] neg_f;

	free(chargesQij); chargesQij = NULL;
}

// Compute gradient of the t-SNE cost function (using FFT)
void TSNE::computeFftGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, int  nterms, double  intervals_per_integer, int min_num_intervals) {
	//clock_t startTime;
	//Zero out the gradient
	for(int i = 0; i < N * D; i++){
		dC[i] = 0.0;
	}

	double * xs =(double*) malloc(N* sizeof(double));
	double * ys =(double*) malloc(N* sizeof(double));
	double minloc = 1E5;
	double maxloc = -1E5;
	//Find the min of the locs
	for (unsigned long i = 0; i < N; i++) {
		xs[i] = Y[i*2+0];
		ys[i] = Y[i*2+1];
		if ( xs[i] > maxloc) maxloc = xs[i];
		if ( xs[i] < minloc) minloc = xs[i];
		if ( ys[i] > maxloc) maxloc = ys[i];
		if ( ys[i] < minloc) minloc = ys[i];
	}
	minloc = floor(minloc);
	maxloc = ceil(maxloc);

	int ndim =4; //number of charges
	double * chargesQij =(double*) malloc(ndim*N* sizeof(double));
	double * potentialQij =(double*) malloc(ndim*N* sizeof(double));

	for (unsigned long j = 0; j < N; j++) {
		chargesQij[j] = 1;
		chargesQij[1*N+j] = Y[j*2];
		chargesQij[2*N+j] = Y[j*2 +1 ];
		chargesQij[3*N+j] = Y[j*2]*Y[j*2] +  Y[j*2 +1 ]*Y[j*2+1];
	} 

	int nlat =fmax(min_num_intervals, (maxloc-minloc)/(double) intervals_per_integer);
//	printf("%d nodes, from %lf to %lf, so using nlat=%d\n",nterms, minloc, maxloc, nlat);
	int nboxes = nlat*nlat;


	double * band = (double *) calloc(N,sizeof(double));
	double * boxl =(double*) malloc(2*nboxes* sizeof(double));
	double * boxr =(double*) malloc(2*nboxes* sizeof(double));
	double *prods =(double*) malloc(nterms* sizeof(double));
	double *xpts =(double*) malloc(nterms* sizeof(double));
	int nfourh = nterms*nlat;
	double *xptsall =(double*) calloc(nfourh*nfourh, sizeof(double));
	double *yptsall =(double*) calloc(nfourh*nfourh, sizeof(double));
	int *irearr =(int*) calloc(nfourh*nfourh, sizeof(int));

	clock_t startTime	=	clock();
	fftw_complex * zkvalf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*nfourh*2*nfourh);
	precompute2(maxloc, minloc, maxloc, minloc,nlat, nterms, &cauchy2d, band,boxl, boxr,  prods, xpts, xptsall,yptsall,irearr,zkvalf );
	nbodyfft2(N,ndim,xs, ys,chargesQij, nlat, nterms,boxl, boxr,  prods, xpts, xptsall,yptsall, irearr, zkvalf,potentialQij);

	double zSum = 0;
	for (unsigned long i = 0; i < N; i++) {
		double phi1 =  potentialQij[i];
		double phi2 =  potentialQij[1*N+i];
		double phi3 =  potentialQij[2*N+i];
		double phi4 =  potentialQij[3*N+i];
		double Y_i_1 = Y[i*2];
		double Y_i_2 = Y[i*2 +1 ];

		zSum += (1 + Y_i_1*Y_i_1 + Y_i_2*Y_i_2)*phi1 - 2*(Y_i_1*phi2 + Y_i_2*phi3) + phi4;
	}
	zSum -= N;
	//printf("zSum from the new calc is %le\n\n", zSum2);
	clock_t end2 = clock();
	double time_spent2 = (double)(end2 - startTime) / CLOCKS_PER_SEC;
	//printf("%d points in 2D with %d charges from %f to %f. nlat: %d, nterms: %d, so (nlat*nlat*nterms)^2 = %d point FFT. \nFast: %.2e seconds, %.2e per second\n", 
	//N, ndim, minloc, maxloc,nlat,nterms,nlat*nterms*2*nlat*nterms*2,  time_spent2, N/time_spent2);

	//Now, figure out the Gaussian component of the gradient.  This
	//coresponds to the "attraction" term of the gradient.  It was
	//calculated using a fast KNN approach, so here we just use the results
	//that were passed to this function
	unsigned int ind2 =0;
	double * pos_f;
	pos_f = new double[N*2];
	// Loop over all edges in the graph
	for(unsigned int n = 0; n < N; n++) {
		pos_f [n*2] = 0;
		pos_f [n*2 +1] = 0;

		for(unsigned int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {
			// Compute pairwise distance and Q-value
			ind2 = inp_col_P[i];
			double r=  (Y[n*2] - Y[ind2*2] ) * (Y[n*2] - Y[ind2*2] )  + (Y[n*2+1] - Y[ind2*2+1] ) * (Y[n*2+1] - Y[ind2*2+1] );
			double q_ij = 1/(1+ r);

			pos_f [n*2] += inp_val_P[i] * q_ij * (Y[n*2] - Y[ind2*2] );
			pos_f [n*2+1] += inp_val_P[i] * q_ij * (Y[n*2+1] - Y[ind2*2+1] );
		}
	}

	FILE* fp;
	if (measure_accuracy){
		char buffer[500];
		sprintf(buffer,"temp/fft_gradient%d.txt", itTest);
		fp = fopen( buffer, "w" ); // Open file for writing
	}

	//Make the negative term, or F_rep in the equation 3 of the paper
	double * neg_f;
	neg_f = new double[N*2];
	for(unsigned int n = 0; n < N; n++) {

		double Qij_y_i_0,Qij_y_i_1,Qij_y_j_0,Qij_y_j_1,Qij_y_0,Qij_y_1;

		Qij_y_i_0 = Y[2*n] *potentialQij[n ];
		Qij_y_i_1 = Y[2*n+1]*potentialQij[n ];

		Qij_y_j_0 = potentialQij[1*N +n ];
		Qij_y_j_1 = potentialQij[2*N +n ];

		Qij_y_0 = Qij_y_i_0 - Qij_y_j_0;
		Qij_y_1 = Qij_y_i_1 - Qij_y_j_1;

		//Note that we only use the Z normalization term in the F_rep,
		//because it cancels in the F_attr.  Also, note that we divide
		//it, because the denominator of q_ij^2  is Z^2, so it cancels
		//out the Z in the numerater for Equation 3
		neg_f [n*2] = Qij_y_0/zSum;
		neg_f [n*2+1] = Qij_y_1/zSum;

		dC[n*2+0] = pos_f[n*2] - neg_f[n*2];
		dC[n*2+1] = pos_f[n*2+1] - neg_f[n*2+1];
		//fprintf(fp, "%d, %e, %e, %e\n",n, dC[n*2+0], pos_f[n*2], neg_f[n*2]);
		if (measure_accuracy){
			fprintf(fp,"%d, %.12e, %.12e, %.12e, %.12e, %.12e  %.12e\n",n, dC[n*2+0], dC[n*2+1], pos_f[n*2],pos_f[n*2+1], neg_f[n*2], neg_f[n*2+1]);
		}
		if (n<10){
			//printf("fft: %d, %e, %e, %e\n",n, dC[n*2+0], pos_f[n*2], neg_f[n*2]);
		}

	}
	if (measure_accuracy){
		fclose(fp);
	}
	clock_t end3 = clock();
	double time_spent3 = (double)(end3 - end2) / CLOCKS_PER_SEC;
	//printf("Rest of it took %lf\n", time_spent3);

	delete[] pos_f;
	delete[] neg_f;
	free(potentialQij); 
	free(chargesQij); 

	free(xs); free(ys);free(band);
	free(boxl); free (boxr); free(prods); free(xpts); free(xptsall);
	free(yptsall); free(irearr);
	fftw_free(zkvalf);
	chargesQij = NULL;
}


void TSNE::computeExactGradientTest(double* Y, int N, int D) {
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(Y, N, D, DD);

	// Compute Q-matrix and normalization sum
	double* Q    = (double*) malloc(N * N * sizeof(double));
	if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for(int n = 0; n < N; n++) {
		for(int m = 0; m < N; m++) {
			if(n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	// Perform the computation of the gradient
	char buffer[500];
	sprintf(buffer,"temp/exact_gradient%d.txt", itTest);
	FILE * fp = fopen( buffer, "w" ); // Open file for writing
	nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		double testQij = 0;
		double testPos = 0;
		double testNeg = 0;
		double testdC = 0;
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				testNeg +=Q[nN+m]*Q[nN+m]*(Y[nD + 0] - Y[mD + 0]) / sum_Q;
			}
			mD += D;
		}
		fprintf(fp, "%d, %.12e\n",n,testNeg);
		nN += N;
		nD += D;
	}
	fclose(fp);
	free(DD);free(Q);
}

// Compute gradient of the t-SNE cost function (exact)
void TSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC) {
	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(Y, N, D, DD);

	// Compute Q-matrix and normalization sum
	double* Q    = (double*) malloc(N * N * sizeof(double));
	if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double sum_Q = .0;
	int nN = 0;
	for(int n = 0; n < N; n++) {
		for(int m = 0; m < N; m++) {
			if(n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}

	// Perform the computation of the gradient
	char buffer[500];
	sprintf(buffer,"temp/exact_gradient%d.txt", itTest);
	FILE * fp = fopen( buffer, "w" ); // Open file for writing
	nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		double testQij = 0;
		double testPos = 0;
		double testNeg = 0;
		double testdC = 0;
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for(int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
				testQij += Q[nN + m]* Q[nN +m] *(Y[nD] - Y[mD]);
				testPos +=P[nN +m]*Q[nN+m]*(Y[nD + 0] - Y[mD + 0]);
				testNeg +=Q[nN+m]*Q[nN+m]*(Y[nD + 0] - Y[mD + 0]) / sum_Q;
			}
			mD += D;
		}
		if (n < 20 ) {
			testdC = testPos - testNeg;
			//printf("dC: %e, %e testDc %e \n", dC[nD +0], dC[nD+1], testdC);

		}
		fprintf(fp, "%d, %.12e\n",n,testNeg);
		nN += N;
		nD += D;
	}
	fclose(fp);
	free(Q);

}


// Evaluate t-SNE cost function (exactly)
double TSNE::evaluateError(double* P, double* Y, int N, int D) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	double* Q = (double*) malloc(N * N * sizeof(double));
	if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(Y, N, D, DD);

	// Compute Q-matrix and normalization sum
	int nN = 0;
	double sum_Q = DBL_MIN;
	for(int n = 0; n < N; n++) {
		for(int m = 0; m < N; m++) {
			if(n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
			else Q[nN + m] = DBL_MIN;
		}
		nN += N;
	}
	for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

	// Sum t-SNE error
	double C = .0;
	for(int n = 0; n < N * N; n++) {
		C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

	// Clean up memory
	free(DD);
	free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*) calloc(D, sizeof(double));
	double sum_Q = .0;
	for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for(int n = 0; n < N; n++) {
		ind1 = n * D;
		for(int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
			for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}


// Compute input similarities with a fixed perplexity
void TSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity, double sigma) {
	if (perplexity < 0 ) {
		printf("Using manually set kernel width\n");
	}else {
		printf("Using perplexity, not the manually set kernel width\n");
	}

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
	if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
	int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
		double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		if (perplexity > 0) {
			while(!found && iter < 200) {

				// Compute Gaussian kernel row
				for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
				P[nN + n] = DBL_MIN;

				// Compute entropy of current row
				sum_P = DBL_MIN;
				for(int m = 0; m < N; m++) sum_P += P[nN + m];
				double H = 0.0;
				for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
				H = (H / sum_P) + log(sum_P);

				// Evaluate whether the entropy is within the tolerance level
				double Hdiff = H - log(perplexity);
				if(Hdiff < tol && -Hdiff < tol) {
					found = true;
				}
				else {
					if(Hdiff > 0) {
						min_beta = beta;
						if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
							beta *= 2.0;
						else
							beta = (beta + max_beta) / 2.0;
					}
					else {
						max_beta = beta;
						if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
							beta /= 2.0;
						else
							beta = (beta + min_beta) / 2.0;
					}
				}

				// Update iteration counter
				iter++;
				//printf("Beta is %lf\n", beta);
			}
		}else{
			beta = 1/sigma;
			//printf("Beta is static and %lf\n", beta);
			for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			for(int m = 0; m < N; m++) {
				if (n < 20 & m <40 ) {
					//printf("%d, %d: beta %lf, DD %lf, P: %lf\n ", n, m, beta, DD[nN + m], P[nN + m]);
				}
			}
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}

	// Clean up memory
	free(DD); DD = NULL;
}

//Use annoy
int TSNE::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, 
		double** _val_P, double perplexity, int K, double sigma, int num_trees, int search_k, unsigned int nthreads) {

	if( access( "temp/val_P.dat", F_OK ) != -1 ) {
		printf("val_P exists, loading the file.");
		*_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
		*_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
		*_val_P = (double*) calloc(N * K, sizeof(double));
		unsigned int* row_P = *_row_P;
		unsigned int* col_P = *_col_P;
		double* val_P = *_val_P;

		if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		FILE *h;
		if((h = fopen("temp/val_P.dat", "rb")) == NULL) {
			printf("Error: could not open data file.\n");
			return -2;
		}
		fread(val_P, sizeof(double), N * K, h);
		fclose(h);

		if((h = fopen("temp/col_P.dat", "rb")) == NULL) {
			printf("Error: could not open data file.\n");
			return -2;
		}
		fread(col_P, sizeof(unsigned int), N * K, h);
		fclose(h);

		if((h = fopen("temp/row_P.dat", "rb")) == NULL) {
			printf("Error: could not open data file.\n");
			return -2;
		}
		fread(row_P, sizeof(unsigned int), N +1, h);
		fclose(h);
		printf("dat files loaded successfully %u\n", row_P[1]);
		return 1;


	}else{
		//printf("K is %d, but the perplexity which we will use for beta is %lf", perplexity, K);
		if(perplexity > K) printf("Perplexity should be lower than K!\n");

		// Allocate the memory we need
		*_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
		*_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
		printf("Going to allocate N: %d, K: %d, N*K = %d\n ", N, K, N*K);
		*_val_P = (double*) calloc(N * K, sizeof(double));
		if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		unsigned int* row_P = *_row_P;
		unsigned int* col_P = *_col_P;
		double* val_P = *_val_P;
		row_P[0] = 0;
		for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

		printf("Building Annoy tree...\n");
		//Begin annoy


		AnnoyIndex<int, double, Euclidean, Kiss32Random> tree = AnnoyIndex<int, double, Euclidean, Kiss32Random>(D);

		for(int i=0; i<N; ++i){
			double *vec = (double *) malloc( D * sizeof(double) );

			for(int z=0; z<D; ++z){
				vec[z] = X[i*D+z];
			}

			tree.add_item(i, vec);
		}
		tree.build(num_trees);


		//End annoy
		printf("Done building Annoy tree. Begin nearest neighbor search... \n");

		if (perplexity >0 ) {
			printf("Calculating dynamic kernels using perplexity \n");
		}else {
			printf("Using sigma= %lf", sigma);
		}
		//Check if it returns enough neighbors
		std::vector<int> closest;
		std::vector<double> closest_distances;
		for (int n = 0; n < 100; n++){
			tree.get_nns_by_item(n, K+1, search_k, &closest, &closest_distances);
			unsigned int neighbors_count = closest.size();
			if (neighbors_count < K+1 ) {
				printf("Requesting perplexity*3=%d neighbors, but ANNOY is only giving us %u. Please increase search_k\n", K, neighbors_count);
				return -1;
			}
		}

		if (nthreads == 0) {
			nthreads = std::thread::hardware_concurrency();
		}
		//const size_t nthreads = 1;
		{
			// Pre loop
			std::cout<<"parallel ("<<nthreads<<" threads):"<<std::endl;
			std::vector<std::thread> threads(nthreads);
			for(int t = 0;t<nthreads;t++)
			{
				threads[t] = std::thread(std::bind(
							[&](const int bi, const int ei, const int t)
							{
							// loop over all items
							for(int n = bi;n<ei;n++)
							{
							// inner loop
							{
							//if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

							// Find nearest neighbors
							std::vector<int> closest;
							std::vector<double> closest_distances;
							tree.get_nns_by_item(n, K+1, search_k, &closest, &closest_distances);

							// Initialize some variables for binary search
							bool found = false;
							double beta = 1.0;
							double min_beta = -DBL_MAX;
							double max_beta =  DBL_MAX;
							double tol = 1e-5;
							double* cur_P = (double*) malloc((N - 1) * sizeof(double));
							if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

							// Iterate until we found a good perplexity
							int iter = 0; double sum_P;
							if (perplexity > 0) {
								while(!found && iter < 200) {

									// Compute Gaussian kernel row
									for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * closest_distances[m + 1] * closest_distances[m + 1]);
									//for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

									// Compute entropy of current row
									sum_P = DBL_MIN;
									for(int m = 0; m < K; m++) sum_P += cur_P[m];
									double H = .0;
									for(int m = 0; m < K; m++) H += beta * (closest_distances[m + 1] * closest_distances[m + 1] * cur_P[m]);
									//for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
									H = (H / sum_P) + log(sum_P);

									// Evaluate whether the entropy is within the tolerance level
									double Hdiff = H - log(perplexity);
									if(Hdiff < tol && -Hdiff < tol) {
										found = true;
										if(n % 10000 == 0) printf(" - point %d of %d, most recent beta calculated is %lf \n", n, N, beta);

									}
									else {
										if(Hdiff > 0) {
											min_beta = beta;
											if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
												beta *= 2.0;
											else
												beta = (beta + max_beta) / 2.0;
										}
										else {
											max_beta = beta;
											if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
												beta /= 2.0;
											else
												beta = (beta + min_beta) / 2.0;
										}
									}

									// Update iteration counter
									iter++;
								}
							}else{
								beta = 1/sigma;
								//printf("Beta is %lf\n", beta);
								for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * closest_distances[m + 1] * closest_distances[m + 1]);

								// Compute entropy of current row
								sum_P = DBL_MIN;
								for(int m = 0; m < K; m++) sum_P += cur_P[m];
							}

							// Row-normalize current row of P and store in matrix
							for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
							for(unsigned int m = 0; m < K; m++) {
								//				col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
								col_P[row_P[n] + m] = (unsigned int) closest[m + 1];
								val_P[row_P[n] + m] = cur_P[m];
							}
							//	printf("Using this perplexity, learned a sqrt(beta) of %lf, sqrt(1/beta) = %lf \n", sqrt(beta), sqrt(1/beta));
							free(cur_P);
							closest.clear();
							closest_distances.clear();
							}
							}
							},t*N/nthreads,(t+1)==nthreads?N:(t+1)*N/nthreads,t));
			}
			std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
			// Post loop
		}


		// Clean up memory


		/*
		   FILE *h;
		   if((h = fopen("temp/val_P.dat", "w+b")) == NULL) {
		   printf("Error: could not open data file.\n");
		   return;
		   }
		   fwrite(val_P, sizeof(double), N * K, h);
		   fclose(h);

		   if((h = fopen("temp/col_P.dat", "w+b")) == NULL) {
		   printf("Error: could not open data file.\n");
		   return;
		   }
		   fwrite(col_P, sizeof(unsigned int), N * K, h);
		   fclose(h);

		   if((h = fopen("temp/row_P.dat", "w+b")) == NULL) {
		   printf("Error: could not open data file.\n");
		   return;
		   }
		   fwrite(row_P, sizeof(unsigned int), N +1, h);
		   fclose(h);
		   */

	}
	return 0;
}
// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K, double sigma, unsigned int nthreads) {


	if(perplexity > K) printf("Perplexity should be lower than K!\n");

	// Allocate the memory we need
	*_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
	*_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
	*_val_P = (double*) calloc(N * K, sizeof(double));
	if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;
	row_P[0] = 0;
	for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

	// Build ball tree on data set
	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
	vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
	for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
	tree->create(obj_X);

	// Loop over all points to find nearest neighbors
	printf("Building tree...\n");

	if (nthreads == 0) {
		nthreads = std::thread::hardware_concurrency();
	}
	//const size_t nthreads = 1;
	{
		// Pre loop
		std::cout<<"parallel ("<<nthreads<<" threads):"<<std::endl;
		std::vector<std::thread> threads(nthreads);
		for(int t = 0;t<nthreads;t++)
		{
			threads[t] = std::thread(std::bind(
						[&](const int bi, const int ei, const int t)
						{
						// loop over all items
						for(int n = bi;n<ei;n++)
						{
						// inner loop
						{
						//double* cur_P = (double*) malloc((N - 1) * sizeof(double));
						std::vector<double> cur_P(K);
						//if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

						if(n % 10000 == 0) printf(" - Thread %d: %d/%d \n",t, n-bi, ei-bi );
						//if(n % 100 == 0) printf(" - point %d of %d\n", n, N);

						vector<DataPoint> indices;
						vector<double> distances;
						// Find nearest neighbors
						tree->search(obj_X[n], K + 1, &indices, &distances);

						// Initialize some variables for binary search
						bool found = false;
						double beta = 1.0;
						double min_beta = -DBL_MAX;
						double max_beta =  DBL_MAX;
						double tol = 1e-5;

						// Iterate until we found a good perplexity
						int iter = 0; double sum_P;
						while(!found && iter < 200) {

							// Compute Gaussian kernel row
							for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

							// Compute entropy of current row
							sum_P = DBL_MIN;
							for(int m = 0; m < K; m++) sum_P += cur_P[m];
							double H = .0;
							for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
							H = (H / sum_P) + log(sum_P);

							// Evaluate whether the entropy is within the tolerance level
							double Hdiff = H - log(perplexity);
							if(Hdiff < tol && -Hdiff < tol) {
								found = true;
							}
							else {
								if(Hdiff > 0) {
									min_beta = beta;
									if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
										beta *= 2.0;
									else
										beta = (beta + max_beta) / 2.0;
								}
								else {
									max_beta = beta;
									if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
										beta /= 2.0;
									else
										beta = (beta + min_beta) / 2.0;
								}
							}

							// Update iteration counter
							iter++;
						}

						//printf("\n point: %d", n);
						// Row-normalize current row of P and store in matrix
						for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
						for(unsigned int m = 0; m < K; m++) {
							col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
							val_P[row_P[n] + m] = cur_P[m];
							//printf(", %.12f(%d)", cur_P[m], m);
						}


						indices.clear();
						distances.clear();
						cur_P.clear();
						}
						}

						},t*N/nthreads,(t+1)==nthreads?N:(t+1)*N/nthreads,t));
		}
		std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
		// Post loop
	}
	printf("Done!");

	// Clean up memory
	obj_X.clear();
	delete tree;
}


// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

	// Get sparse matrix
	unsigned int* row_P = *_row_P;
	unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;

	// Count number of elements and row counts of symmetric matrix
	int* row_counts = (int*) calloc(N, sizeof(int));
	if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for(int n = 0; n < N; n++) {
		for(int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if(col_P[m] == n) present = true;
			}
			if(present) row_counts[n]++;
			else {
				row_counts[n]++;
				row_counts[col_P[i]]++;
			}
		}
	}
	int no_elem = 0;
	for(int n = 0; n < N; n++) no_elem += row_counts[n];

	// Allocate memory for symmetrized matrix
	unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
	unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
	double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
	if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	sym_row_P[0] = 0;
	for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

	// Fill the result matrix
	int* offset = (int*) calloc(N, sizeof(int));
	if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for(int n = 0; n < N; n++) {
		for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if(col_P[m] == n) {
					present = true;
					if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
						sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
					}
				}
			}

			// If (col_P[i], n) is not present, there is no addition involved
			if(!present) {
				sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
				sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
				sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
				sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
			}

			// Update offsets
			if(!present || (n <= col_P[i])) {
				offset[n]++;
				if(col_P[i] != n) offset[col_P[i]]++;
			}
		}
	}

	// Divide the result by two
	for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

	// Return symmetrized matrices
	free(*_row_P); *_row_P = sym_row_P;
	free(*_col_P); *_col_P = sym_col_P;
	free(*_val_P); *_val_P = sym_val_P;

	// Free up some memery
	free(offset); offset = NULL;
	free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
void TSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
	const double* XnD = X;
	for(int n = 0; n < N; ++n, XnD += D) {
		const double* XmD = XnD + D;
		double* curr_elem = &DD[n*N + n];
		*curr_elem = 0.0;
		double* curr_elem_sym = curr_elem + N;
		for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
			*(++curr_elem) = 0.0;
			for(int d = 0; d < D; ++d) {
				*curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
			}
			*curr_elem_sym = *curr_elem;
		}
	}
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
	if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
		nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
	nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
		nD += D;
	}
	free(mean); mean = NULL;
}


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	return x;
}
// Function that loads initial data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
/*
bool TSNE::load_initial_data(double** data ) {
	int n, d, no_dims,  rand_seed,  max_iter, stop_lying_iter,K, nbody_algo, knn_algo, no_momentum_during_exag,n_trees,search_k;
	double theta,  perplexity, sigma, early_exag_coeff;
	// Open file, read first 2 integers, allocate memory, and read the data
	FILE *h;
	if((h = fopen("temp/initial_data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(&n, sizeof(int), 1, h);											// number of datapoints
	fread(&d, sizeof(int), 1, h);											// original dimensionality
	fread(&theta, sizeof(double), 1, h);										// gradient accuracy
	fread(&perplexity, sizeof(double), 1, h);								// perplexity
	fread(&no_dims, sizeof(int), 1, h);                                      // output dimensionality
	fread(&max_iter, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&stop_lying_iter, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&K, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&sigma, sizeof(double),1,h);                                       // maximum number of iterations
	fread(&nbody_algo, sizeof(int),1,h);                                       // maximum number of iterations

	fread(&knn_algo, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&early_exag_coeff, sizeof(double),1,h);                                       // maximum number of iterations
	fread(&no_momentum_during_exag, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&n_trees, sizeof(int),1,h);                                       // maximum number of iterations
	fread(&search_k, sizeof(int),1,h);                                       // maximum number of iterations

	*data = (double*) malloc(d * n * sizeof(double));
	if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	fread(*data, sizeof(double), n * d, h);                               // the data
	if(!feof(h)) fread(&rand_seed, sizeof(int), 1, h);                       // random seed
	fclose(h);
	printf("Read the %i x %i data matrix successfully!\n", n, d);
	return true;
}
*/

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(const char *data_path, double** data, int* n, int* d, int* no_dims, double*
		theta, double* perplexity, int* rand_seed, int* max_iter, int* stop_lying_iter,
		int * K, double * sigma, int * nbody_algo, int * knn_algo, double *
		early_exag_coeff, int * no_momentum_during_exag, int * n_trees, int * search_k, int * start_late_exag_iter, double * late_exag_coeff,
		int * nterms, double * intervals_per_integer, int *min_num_intervals) {

	FILE *h;
	if((h = fopen(data_path, "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
	fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
	fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
	fread(stop_lying_iter, sizeof(int),1,h);         
	fread(K, sizeof(int),1,h);                       
	fread(sigma, sizeof(double),1,h);                
	fread(nbody_algo, sizeof(int),1,h);              
	fread(knn_algo, sizeof(int),1,h);                
	fread(early_exag_coeff, sizeof(double),1,h);     
	fread(no_momentum_during_exag, sizeof(int),1,h); 
	fread(n_trees, sizeof(int),1,h);                 
	fread(search_k, sizeof(int),1,h);                
	fread(start_late_exag_iter, sizeof(int),1,h);    
	fread(late_exag_coeff, sizeof(double),1,h);      

	fread(nterms, sizeof(int),1,h);    
	fread(intervals_per_integer, sizeof(double),1,h);      
	fread(min_num_intervals, sizeof(int),1,h);    



	*data = (double*) malloc(*d * *n * sizeof(double));
	if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	fread(*data, sizeof(double), *n * *d, h);                               // the data
	if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
	fclose(h);
	printf("Read the following parameters:\n\t n %d by d %d dataset, theta %lf\n"
			"\t perplexity %lf, no_dims %d, max_iter %d,  stop_lying_iter %d\n"
			"\t K %d, sigma %lf, nbody_algo %d, knn_algo %d, compexagcoef %lf\n"
			"\t no_momentum_during_exag %d, n_trees %d, search_k %d, start_late_exag_iter %d, late_exag_coeff %lf\n"
			"\t nterms %d, interval_per_integer %lf, min_num_intervals %d\n",  
			*n, *d, *theta, *perplexity, 
			*no_dims, *max_iter,*stop_lying_iter, 
			*K, *sigma, *nbody_algo, *knn_algo, *early_exag_coeff,
			*no_momentum_during_exag, *n_trees, *search_k, *start_late_exag_iter, *late_exag_coeff,
			*nterms, *intervals_per_integer, *min_num_intervals);
	printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(const char *result_path, double* data, int* landmarks, double* costs, int n, int d, double initialError) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen(result_path, "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&initialError, sizeof(double), 1, h);
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
	fwrite(costs, sizeof(double), n, h);
	fclose(h);
	printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}


// Function that runs the Barnes-Hut implementation of t-SNE
int main(int argc, char *argv[]) {

	// Define some variables
	int origN, N, D, no_dims, max_iter, stop_lying_iter,  K, nbody_algo, knn_algo, no_momentum_during_exag,n_trees,search_k, start_late_exag_iter;
	double   sigma, early_exag_coeff, late_exag_coeff;
	double perplexity, theta, *data, *initial_data;
	int nterms, min_num_intervals;
	double intervals_per_integer;
	int rand_seed;
	const char *data_path, *result_path;
	unsigned int nthreads;
	TSNE* tsne = new TSNE();
	

	data_path = "temp/data.dat";
	result_path = "temp/result.dat";
	nthreads = 0;
	if(argc >= 2) {
		data_path = argv[1];
	}
	if(argc >= 3) {
		result_path = argv[2];
	}
	if(argc >= 4) {
		nthreads = (unsigned int)strtoul(argv[3], (char **)NULL, 10);
	}
	std::cout<<"fast_tsne data_path: "<< data_path <<std::endl;
	std::cout<<"fast_tsne result_path: "<< result_path <<std::endl;
	std::cout<<"fast_tsne nthreads: "<< nthreads <<std::endl;

	// Read the parameters and the dataset
	if(tsne->load_data(data_path, &data, &origN, &D, &no_dims, &theta, &perplexity,
				&rand_seed, &max_iter, &stop_lying_iter, &K,
				&sigma, &nbody_algo, &knn_algo,
				&early_exag_coeff, &no_momentum_during_exag,
				&n_trees, &search_k, &start_late_exag_iter,
				&late_exag_coeff,
				&nterms, &intervals_per_integer, &min_num_intervals)) {

		bool no_momentum_during_exag_bool = true;
		if (no_momentum_during_exag == 0) no_momentum_during_exag_bool = false;
		// Make dummy landmarks
		N = origN;
		int* landmarks = (int*) malloc(N * sizeof(int));
		if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		for(int n = 0; n < N; n++) landmarks[n] = n;

		// Now fire up the SNE implementation
		double* Y = (double*) malloc(N * no_dims * sizeof(double));
		//Uncomment if using initial data, and change the run() call below
		//tsne->load_initial_data(&initial_data); 
		//printf("Initializing with initial_data\n");
		//for (int i = 0; i<no_dims; i++) {
		//for (int j=0; j< N;j++){
		////printf("%d,%d: %lf\n", i,j, initial_data[j*no_dims+i]);
		//Y[j*no_dims + i] = initial_data[j*no_dims+i];
		//}
		//}

		double* costs = (double*) calloc(max_iter, sizeof(double));
		if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		double initialError;
		//Always using random initilization.
		int error_code = 0;
		error_code = tsne->run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter, 
				stop_lying_iter,250, K, sigma, nbody_algo, knn_algo, early_exag_coeff, &initialError, 
				costs, no_momentum_during_exag_bool, start_late_exag_iter, late_exag_coeff, n_trees,search_k, 
				nterms, intervals_per_integer, min_num_intervals, nthreads);
		if (error_code <0 ) {
			exit(error_code);
		}

		// Save the results
		tsne->save_data(result_path, Y, landmarks, costs, N, no_dims, initialError);

		// Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
	}
	delete(tsne);
}
