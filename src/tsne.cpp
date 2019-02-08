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

#include "winlibs/stdafx.h"
#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <iostream>
#include <fstream>
#include "nbodyfft.h"
#include <math.h>
#include "annoylib.h"
#include "kissrandom.h"
#include <thread>
#include <float.h>
#include <cstring>
#include "vptree.h"
#include "sptree.h"
#include "tsne.h"
#include "progress_bar/ProgressBar.hpp"
#include "parallel_for.h"
#include "time_code.h"

using namespace std::chrono;
#ifdef _WIN32
#include "winlibs/unistd.h"
#else
#include <unistd.h>
#endif
#include <functional>

#define _CRT_SECURE_NO_WARNINGS


int itTest = 0;
bool measure_accuracy = false;


double squared_cauchy(double x, double y) {
    return pow(1.0 + pow(x - y, 2), -2);
}


double squared_cauchy_2d(double x1, double x2, double y1, double y2,double df) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

double general_kernel_2d(double x1, double x2, double y1, double y2, double df) {
    return pow(1.0 + ((x1 - y1)*(x1-y1) + (x2 - y2)*(x2-y2))/df, -(df));
}

double squared_general_kernel_2d(double x1, double x2, double y1, double y2, double df) {
    return pow(1.0 + ((x1 - y1)*(x1-y1) + (x2 - y2)*(x2-y2))/df, -(df+1.0));
}


using namespace std;

//Helper function for printing Y at each iteration. Useful for debugging
void print_progress(int iter, double *Y, int N, int no_dims) {

    ofstream myfile;
    std::ostringstream stringStream;
    stringStream << "dat/intermediate" << iter << ".txt";
    std::string copyOfStr = stringStream.str();
    myfile.open(stringStream.str().c_str());
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < no_dims; i++) {
            myfile << Y[j * no_dims + i] << " ";
        }
        myfile << "\n";
    }
    myfile.close();
}


// Perform t-SNE
int TSNE::run(double *X, int N, int D, double *Y, int no_dims, double perplexity, double theta, int rand_seed,
              bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
              double momentum, double final_momentum, double learning_rate, int K, double sigma,
              int nbody_algorithm, int knn_algo, double early_exag_coeff, double *costs,
              bool no_momentum_during_exag, int start_late_exag_iter, double late_exag_coeff, int n_trees, int search_k,
              int nterms, double intervals_per_integer, int min_num_intervals, unsigned int nthreads, 
              int load_affinities, int perplexity_list_length, double *perplexity_list, double df) {

    // Some logging messages
    if (N - 1 < 3 * perplexity) {
        printf("Perplexity too large for the number of data points!\n");
        exit(1);
    }

    if (no_momentum_during_exag) {
		printf("No momentum during the exaggeration phase.\n");
    } else {
        printf("Will use momentum during exaggeration phase\n");
    }
    
    // Determine whether we are using an exact algorithm
    bool exact = theta == .0;

    // Allocate some memory
    auto *dY = (double *) malloc(N * no_dims * sizeof(double));
    auto *uY = (double *) malloc(N * no_dims * sizeof(double));
    auto *gains = (double *) malloc(N * no_dims * sizeof(double));
    if (dY == nullptr || uY == nullptr || gains == nullptr) throw std::bad_alloc();

    // Initialize gradient to zeros and gains to ones.
    for (int i = 0; i < N * no_dims; i++) uY[i] = .0;
    for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    printf("Computing input similarities...\n");
    zeroMean(X, N, D);

    if (perplexity > 0 || perplexity_list_length > 0) {
        printf("Using perplexity, so normalizing input data (to prevent numerical problems)\n");
        double max_X = .0;
        for (int i = 0; i < N * D; i++) {
            if (fabs(X[i]) > max_X) max_X = fabs(X[i]);
        }
        for (int i = 0; i < N * D; i++) X[i] /= max_X;
    } else {
		printf("Not using perplexity, so data are left un-normalized.\n");
    }

    // Compute input similarities for exact t-SNE
    double *P = nullptr;
    unsigned int *row_P = nullptr;
    unsigned int *col_P = nullptr;
    double *val_P = nullptr;
    if (exact) {
		// Loading input similarities if load_affinities == 1
		if (load_affinities == 1) {
			printf("Loading exact input similarities from file...\n");

			P = (double *) malloc(N * N * sizeof(double));
			if (P == NULL) {
				printf("Memory allocation failed!\n");
				exit(1);
			}

			FILE *h;
			size_t result;
			if ((h = fopen("P.dat", "rb")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			result = fread(P, sizeof(double), N * N, h);
			fclose(h);
		} else {

			// Compute similarities
			printf("Theta set to 0, so running exact algorithm\n");
			P = (double *) malloc(N * N * sizeof(double));
			if (P == NULL) {
				printf("Memory allocation failed!\n");
				exit(1);
			}

			computeGaussianPerplexity(X, N, D, P, perplexity, sigma, perplexity_list_length, perplexity_list);

			// Symmetrize input similarities
			printf("Symmetrizing...\n");
			int nN = 0;
			for (int n = 0; n < N; n++) {
				int mN = (n + 1) * N;
				for (int m = n + 1; m < N; m++) {
					P[nN + m] += P[mN + n];
					P[mN + n] = P[nN + m];
					mN += N;
				}
				nN += N;
			}
			double sum_P = .0;
			for (int i = 0; i < N * N; i++) sum_P += P[i];
			for (int i = 0; i < N * N; i++) P[i] /= sum_P;
			//sum_P is just a cute way of writing 2N
			printf("Finished exact calculation of the P.  Sum_p: %lf \n", sum_P);
		}
		// Saving input similarities if load_affinities == 2
		if (load_affinities == 2) {
			printf("Saving exact input similarities to file...\n");
			FILE *h;
			if ((h = fopen("P.dat", "w+b")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			fwrite(P, sizeof(double), N * N, h);
			fclose(h);
		}
	}

	// Compute input similarities for approximate t-SNE
    else {
		// Loading input similarities if load_affinities == 1
		if (load_affinities == 1) {
			printf("Loading approximate input similarities from files...\n");

			row_P = (unsigned int *) malloc((N + 1) * sizeof(unsigned int));
			if (row_P == NULL) {
				printf("Memory allocation failed!\n");
				exit(1);
			}

			FILE *h;
			size_t result;
			if ((h = fopen("P_row.dat", "rb")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			result = fread(row_P, sizeof(unsigned int), N + 1, h);
			fclose(h);

			int numel = row_P[N];
			col_P = (unsigned int *) calloc(numel, sizeof(unsigned int));
			val_P = (double *) calloc(numel, sizeof(double));
			if (col_P == NULL || val_P == NULL) {
				printf("Memory allocation failed!\n");
				exit(1);
			}

			if ((h = fopen("P_val.dat", "rb")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			result = fread(val_P, sizeof(double), numel, h);
			fclose(h);

			if ((h = fopen("P_col.dat", "rb")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			result = fread(col_P, sizeof(unsigned int), numel, h);
			fclose(h);

			printf("   val_P: %f %f %f ... %f %f %f\n", val_P[0], val_P[1], val_P[2],
				   val_P[numel - 3], val_P[numel - 2], val_P[numel - 1]);
			printf("   col_P: %d %d %d ... %d %d %d\n", col_P[0], col_P[1], col_P[2],
				   col_P[numel - 3], col_P[numel - 2], col_P[numel - 1]);
			printf("   row_P: %d %d %d ... %d %d %d\n", row_P[0], row_P[1], row_P[2],
				   row_P[N - 2], row_P[N - 1], row_P[N]);
		} else {
			// Compute asymmetric pairwise input similarities
			int K_to_use;
			double sigma_to_use;

			if (perplexity < 0) {
				printf("Using manually set kernel width\n");
				K_to_use = K;
				sigma_to_use = sigma;
			} else {
				printf("Using perplexity, not the manually set kernel width.  K (number of nearest neighbors) and sigma (bandwidth) parameters are going to be ignored.\n");
				if (perplexity > 0) {
                    K_to_use = (int) 3 * perplexity;
                } else {
                    K_to_use = (int) 3 * perplexity_list[0];
                    for (int pp = 1; pp < perplexity_list_length; pp++) {
                        if ((int) 3* perplexity_list[pp] > K_to_use) {
                            K_to_use = (int) 3 * perplexity_list[pp];
						}
                    }
                }                         
				sigma_to_use = -1;
			}

			if (knn_algo == 1) {
				printf("Using ANNOY for knn search, with parameters: n_trees %d and search_k %d\n", n_trees, search_k);
				int error_code = 0;
				error_code = computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, K_to_use,
													   sigma_to_use, n_trees, search_k, nthreads,
                                                       perplexity_list_length, perplexity_list, rand_seed);
				if (error_code < 0) return error_code;
			} else if (knn_algo == 2) {
				printf("Using VP trees for nearest neighbor search\n");
				computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, K_to_use, sigma_to_use,
										  nthreads, perplexity_list_length, perplexity_list);
			} else {
				printf("Invalid knn_algo param\n");
				free(dY);
				free(uY);
				free(gains);
				exit(1);
			}

			// Symmetrize input similarities
			printf("Symmetrizing...\n");
			symmetrizeMatrix(&row_P, &col_P, &val_P, N);
			double sum_P = .0;
			for (int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
			for (int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
		}

		// Saving input similarities if load_affinities == 2
		if (load_affinities == 2) {
			printf("Saving approximate input similarities to files...\n");
			int numel = row_P[N];

			FILE *h;
			if ((h = fopen("P_val.dat", "w+b")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			fwrite(val_P, sizeof(double), numel, h);
			fclose(h);

			if ((h = fopen("P_col.dat", "w+b")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			fwrite(col_P, sizeof(unsigned int), numel, h);
			fclose(h);

			if ((h = fopen("P_row.dat", "w+b")) == NULL) {
				printf("Error: could not open data file.\n");
				return -2;
			}
			fwrite(row_P, sizeof(unsigned int), N + 1, h);
			fclose(h);

			printf("   val_P: %f %f %f ... %f %f %f\n", val_P[0], val_P[1], val_P[2],
				   val_P[numel - 3], val_P[numel - 2], val_P[numel - 1]);
			printf("   col_P: %d %d %d ... %d %d %d\n", col_P[0], col_P[1], col_P[2],
				   col_P[numel - 3], col_P[numel - 2], col_P[numel - 1]);
			printf("   row_P: %d %d %d ... %d %d %d\n", row_P[0], row_P[1], row_P[2],
				   row_P[N - 2], row_P[N - 1], row_P[N]);
		}
	}

    // Set random seed
    if (skip_random_init != true) {
        if (rand_seed >= 0) {
            printf("Using random seed: %d\n", rand_seed);
            srand((unsigned int) rand_seed);
        } else {
            printf("Using current time as random seed...\n");
            srand(time(NULL));
        }
    }

    // Initialize solution (randomly)
    if (skip_random_init != true) {
		printf("Randomly initializing the solution.\n");
        for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
	printf("Y[0] = %lf\n", Y[0]);
    } else {
		printf("Using the given initialization.\n");
    }

    // If we are doing early exaggeration, we pre-multiply all the P by the coefficient of early exaggeration
    double max_sum_cols = 0;
    // Compute maximum possible exaggeration coefficient, if user requests
    if (early_exag_coeff == 0) {
        for (int n = 0; n < N; n++) {
            double running_sum = 0;
            for (int i = row_P[n]; i < row_P[n + 1]; i++) {
                running_sum += val_P[i];
            }
            if (running_sum > max_sum_cols) max_sum_cols = running_sum;
        }
        early_exag_coeff = (1.0 / (learning_rate * max_sum_cols));
        printf("Max of the val_Ps is: %lf\n", max_sum_cols);
    }

    printf("Exaggerating Ps by %f\n", early_exag_coeff);
    if (exact) {
        for (int i = 0; i < N * N; i++) {
            P[i] *= early_exag_coeff;
        }
    } else {
        for (int i = 0; i < row_P[N]; i++)
            val_P[i] *= early_exag_coeff;
    }

    print_progress(0, Y, N, no_dims);

    // Perform main training loop
    if (exact) {
        printf("Input similarities computed \nLearning embedding...\n");
    } else {
        printf("Input similarities computed (sparsity = %f)!\nLearning embedding...\n",
               (double) row_P[N] / ((double) N * (double) N));
    }

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    
    if (!exact) {
        if (nbody_algorithm == 2) {
            printf("Using FIt-SNE approximation.\n");
        } else if (nbody_algorithm == 1) {
            printf("Using the Barnes-Hut approximation.\n");
        } else {
            printf("Error: Undefined algorithm");
            exit(2);
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {
        itTest = iter;

        if (exact) {
            // Compute the exact gradient using full P matrix
            computeExactGradient(P, Y, N, no_dims, dY,df);
        } else {
            if (nbody_algorithm == 2) {
                // Use FFT accelerated interpolation based negative gradients
                if (no_dims == 1) {
                    computeFftGradientOneD(P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms, intervals_per_integer,
                                           min_num_intervals, nthreads);
                } else {
                    if (df ==1.0) {
                        computeFftGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms, intervals_per_integer,
                                           min_num_intervals, nthreads);
                    }else {
                        computeFftGradientVariableDf(P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms, intervals_per_integer,
                                           min_num_intervals, nthreads,df );

                    }
                }
            } else if (nbody_algorithm == 1) {
                // Otherwise, compute the negative gradient using the Barnes-Hut approximation
                computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);
            }
        }

        if (measure_accuracy) {
            computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);
            computeFftGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms, intervals_per_integer,
                               min_num_intervals, nthreads);
            computeExactGradientTest(Y, N, no_dims,df);
        }

        // We can turn off momentum/gains until after the early exaggeration phase is completed
        if (no_momentum_during_exag) {
            if (iter > stop_lying_iter) {
                for (int i = 0; i < N * no_dims; i++)
                    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
                for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;
                for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];
                for (int i = 0; i < N * no_dims; i++) Y[i] = Y[i] + uY[i];
            } else {
                // During early exaggeration or compression, no trickery (i.e. no momentum, or gains). Just good old
                // fashion gradient descent
                for (int i = 0; i < N * no_dims; i++) Y[i] = Y[i] - dY[i];
            }
        } else {
            for (int i = 0; i < N * no_dims; i++)
                gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
            for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;
            for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];
            for (int i = 0; i < N * no_dims; i++) Y[i] = Y[i] + uY[i];
        }

        // Make solution zero-mean
        zeroMean(Y, N, no_dims);

        // Switch off early exaggeration
        if (iter == stop_lying_iter) {
            printf("Unexaggerating Ps by %f\n", early_exag_coeff);
            if (exact) { for (int i = 0; i < N * N; i++) P[i] /= early_exag_coeff; }
            else { for (int i = 0; i < row_P[N]; i++) val_P[i] /= early_exag_coeff; }
        }
        if (iter == start_late_exag_iter) {
            printf("Exaggerating Ps by %f\n", late_exag_coeff);
            if (exact) { for (int i = 0; i < N * N; i++) P[i] *= late_exag_coeff; }
            else { for (int i = 0; i < row_P[N]; i++) val_P[i] *= late_exag_coeff; }
        }
        if (iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if ((iter+1) % 50 == 0 || iter == max_iter - 1) {
	INITIALIZE_TIME;
        START_TIME;
            double C = .0;
            if (exact) {
                C = evaluateError(P, Y, N, no_dims,df);
            }else{
                if (nbody_algorithm == 2) {
                    C = evaluateErrorFft(row_P, col_P, val_P, Y, N, no_dims,nthreads,df);
                }else {
                    C = evaluateError(row_P, col_P, val_P, Y, N, no_dims,theta, nthreads);
                }
            }
            
            // Adjusting the KL divergence if exaggeration is currently turned on
            // See https://github.com/pavlin-policar/fastTSNE/blob/master/notes/notes.pdf, Section 3.2
            if (iter < stop_lying_iter && stop_lying_iter != -1) {
                C = C/early_exag_coeff - log(early_exag_coeff);
            }
            if (iter >= start_late_exag_iter && start_late_exag_iter != -1) {
                C = C/late_exag_coeff - log(late_exag_coeff);
            }     

            costs[iter] = C;       
    END_TIME("Computing Error");
            
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            printf("Iteration %d (50 iterations in %.2f seconds), cost %f\n", iter+1, std::chrono::duration_cast<std::chrono::milliseconds>(now-start_time).count()/(float)1000.0, C);
            start_time = std::chrono::steady_clock::now();
        }
    }

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);

    if (exact) {
        free(P);
    } else {
        free(row_P);
        free(col_P);
        free(val_P);
    }
    return 0;
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                           int N, int D, double *dC, double theta) {
    // Construct space-partitioning tree on current map
    SPTree *tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double *pos_f = (double *) calloc(N * D, sizeof(double));
    double *neg_f = (double *) calloc(N * D, sizeof(double));
    if (pos_f == NULL || neg_f == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for (int n = 0; n < N; n++) {
        tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);
    }

    // Compute final t-SNE gradient
    FILE *fp = nullptr;
    if (measure_accuracy) {
        char buffer[500];
        sprintf(buffer, "temp/bh_gradient%d.txt", itTest);
        fp = fopen(buffer, "w"); // Open file for writing
    }
    for (int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
        if (measure_accuracy) {
            if (i < N) {
                fprintf(fp, "%d, %.12e, %.12e, %.12e,%.12e,%.12e  %.12e\n", i, dC[i * 2], dC[i * 2 + 1], pos_f[i * 2],
                        pos_f[i * 2 + 1], neg_f[i * 2] / sum_Q, neg_f[i * 2 + 1] / sum_Q);
            }
        }
    }
    if (measure_accuracy) {
        fclose(fp);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}


// Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation for for one
// dimensional Ys
void TSNE::computeFftGradientOneD(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P,
                                  double *Y, int N, int D, double *dC, int n_interpolation_points,
                                  double intervals_per_integer, int min_num_intervals, unsigned int nthreads) {
    // Zero out the gradient
    for (int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Push all the points at which we will evaluate
    // Y is stored row major, with a row corresponding to a single point
    // Find the min and max values of Ys
    double y_min = INFINITY;
    double y_max = -INFINITY;
    for (unsigned long i = 0; i < N; i++) {
        if (Y[i] < y_min) y_min = Y[i];
        if (Y[i] > y_max) y_max = Y[i];
    }

    auto n_boxes = static_cast<int>(fmax(min_num_intervals, (y_max - y_min) / intervals_per_integer));

    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 3;

    auto *chargesQij = new double[N * n_terms];
    auto *potentialsQij = new double[N * n_terms]();

    // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
    for (unsigned long j = 0; j < N; j++) {
        chargesQij[j * n_terms + 0] = 1;
        chargesQij[j * n_terms + 1] = Y[j];
        chargesQij[j * n_terms + 2] = Y[j] * Y[j];
    }

    auto *box_lower_bounds = new double[n_boxes];
    auto *box_upper_bounds = new double[n_boxes];
    auto *y_tilde_spacings = new double[n_interpolation_points];
    auto *y_tilde = new double[n_interpolation_points * n_boxes]();
    auto *fft_kernel_vector = new complex<double>[2 * n_interpolation_points * n_boxes];

    precompute(y_min, y_max, n_boxes, n_interpolation_points, &squared_cauchy, box_lower_bounds, box_upper_bounds,
               y_tilde_spacings, y_tilde, fft_kernel_vector);
    nbodyfft(N, n_terms, Y, chargesQij, n_boxes, n_interpolation_points, box_lower_bounds, box_upper_bounds,
             y_tilde_spacings, y_tilde, fft_kernel_vector, potentialsQij);

    delete[] box_lower_bounds;
    delete[] box_upper_bounds;
    delete[] y_tilde_spacings;
    delete[] y_tilde;
    delete[] fft_kernel_vector;

    // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
    // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
    // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
    double sum_Q = 0;
    for (unsigned long i = 0; i < N; i++) {
        double phi1 = potentialsQij[i * n_terms + 0];
        double phi2 = potentialsQij[i * n_terms + 1];
        double phi3 = potentialsQij[i * n_terms + 2];

        sum_Q += (1 + Y[i] * Y[i]) * phi1 - 2 * (Y[i] * phi2) + phi3;
    }
    sum_Q -= N;
    this->current_sum_Q = sum_Q;

    // Now, figure out the Gaussian component of the gradient. This corresponds to the "attraction" term of the
    // gradient. It was calculated using a fast KNN approach, so here we just use the results that were passed to this
    // function
//    unsigned int ind2 = 0;
  double *pos_f = new double[N];

        PARALLEL_FOR(nthreads, N, {
            double dim1 = 0;
            for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++) {
                // Compute pairwise distance and Q-value
                unsigned int ind3 = inp_col_P[i];
                double d_ij = Y[loop_i] - Y[ind3];
                double q_ij = 1 / (1 + d_ij * d_ij);
                dim1 += inp_val_P[i] * q_ij * d_ij;
            }
                pos_f[loop_i] = dim1;

        });






    // Make the negative term, or F_rep in the equation 3 of the paper
    double *neg_f = new double[N];
    for (unsigned int n = 0; n < N; n++) {
        neg_f[n] = (Y[n] * potentialsQij[n * n_terms] - potentialsQij[n * n_terms + 1]) / sum_Q;

        dC[n] = pos_f[n] - neg_f[n];
    }

    delete[] chargesQij;
    delete[] potentialsQij;
    delete[] pos_f;
    delete[] neg_f;
}

// Compute the gradient of the t-SNE cost function using the FFT interpolation
// based approximation, with variable degree of freedom df 
void TSNE::computeFftGradientVariableDf(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                              int N, int D, double *dC, int n_interpolation_points, double intervals_per_integer,
                              int min_num_intervals, unsigned int nthreads, double df) {


    // Zero out the gradient
    for (int i = 0; i < N * D; i++) dC[i] = 0.0;

    // For convenience, split the x and y coordinate values
    auto *xs = new double[N];
    auto *ys = new double[N];

    double min_coord = INFINITY;
    double max_coord = -INFINITY;
    // Find the min/max values of the x and y coordinates
    for (unsigned long i = 0; i < N; i++) {
        xs[i] = Y[i * 2 + 0];
        ys[i] = Y[i * 2 + 1];
        if (xs[i] > max_coord) max_coord = xs[i];
        else if (xs[i] < min_coord) min_coord = xs[i];
        if (ys[i] > max_coord) max_coord = ys[i];
        else if (ys[i] < min_coord) min_coord = ys[i];
    }
    // Compute the number of boxes in a single dimension and the total number of boxes in 2d
    auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));


    //printf("min_coord: %lf, max_coord: %lf, n_boxes_per_dim: %d, (max_coord - min_coord) / intervals_per_integer) %d\n", min_coord, max_coord, n_boxes_per_dim, static_cast<int>(  (max_coord - min_coord) / intervals_per_integer));
    // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
    // arbitrary
    int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
    if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
        //Round up to nearest grid point
        int chosen_i;
        for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    //printf(" n_boxes_per_dim: %d\n", n_boxes_per_dim );
    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int squared_n_terms = 3;
    auto *SquaredChargesQij = new double[N * squared_n_terms];
    auto *SquaredPotentialsQij = new double[N * squared_n_terms]();

    // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
    for (unsigned long j = 0; j < N; j++) {
        SquaredChargesQij[j * squared_n_terms + 0] = xs[j];
        SquaredChargesQij[j * squared_n_terms + 1] = ys[j];
        SquaredChargesQij[j * squared_n_terms + 2] = 1;
    }

    // Compute the number of boxes in a single dimension and the total number of boxes in 2d
    int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

    auto *box_lower_bounds = new double[2 * n_boxes];
    auto *box_upper_bounds = new double[2 * n_boxes];
    auto *y_tilde_spacings = new double[n_interpolation_points];
    int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
    auto *x_tilde = new double[n_interpolation_points_1d]();
    auto *y_tilde = new double[n_interpolation_points_1d]();
    auto *fft_kernel_tilde = new complex<double>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];

    INITIALIZE_TIME;
    START_TIME;
    precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
                  &squared_general_kernel_2d,
                  box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde, df);
    n_body_fft_2d(N, squared_n_terms, xs, ys, SquaredChargesQij, n_boxes_per_dim, n_interpolation_points, box_lower_bounds,
                  box_upper_bounds, y_tilde_spacings, fft_kernel_tilde, SquaredPotentialsQij, nthreads);

    int not_squared_n_terms = 1;
    auto *NotSquaredChargesQij = new double[N * not_squared_n_terms];
    auto *NotSquaredPotentialsQij = new double[N * not_squared_n_terms]();

    // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
    for (unsigned long j = 0; j < N; j++) {
        NotSquaredChargesQij[j * not_squared_n_terms + 0] = 1;
    }

    precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
                  &general_kernel_2d,
                  box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde,df);
    n_body_fft_2d(N, not_squared_n_terms, xs, ys, NotSquaredChargesQij, n_boxes_per_dim, n_interpolation_points, box_lower_bounds,
                  box_upper_bounds, y_tilde_spacings, fft_kernel_tilde, NotSquaredPotentialsQij, nthreads);




    // Compute the normalization constant Z or sum of q_{ij}.
    double sum_Q = 0;
    for (unsigned long i = 0; i < N; i++) {
        double h1 = NotSquaredPotentialsQij[i * not_squared_n_terms+ 0];
        sum_Q += h1;
    }
    sum_Q -= N;

    // Now, figure out the Gaussian component of the gradient. This corresponds to the "attraction" term of the
    // gradient. It was calculated using a fast KNN approach, so here we just use the results that were passed to this
    // function
    unsigned int ind2 = 0;
    double *pos_f = new double[N * 2];
    END_TIME("Total Interpolation");
        START_TIME;
    // Loop over all edges in the graph
                PARALLEL_FOR(nthreads, N, {
                                double dim1 = 0;
                                double dim2 = 0;

                                for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++) {
                                // Compute pairwise distance and Q-value
                                    unsigned int ind3 = inp_col_P[i];
                                    double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
                                    double q_ij = 1 / (1 + d_ij/df);

                                    dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
                                    dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
                                }
                                pos_f[loop_i * 2 + 0] = dim1;
                                pos_f[loop_i * 2 + 1] = dim2;
                  });

    // Make the negative term, or F_rep in the equation 3 of the paper
    END_TIME("Attractive Forces");

    double *neg_f = new double[N * 2];
    for (unsigned int i = 0; i < N; i++) {
        double h2 = SquaredPotentialsQij[i * squared_n_terms];
        double h3 = SquaredPotentialsQij[i * squared_n_terms + 1];
        double h4 = SquaredPotentialsQij[i * squared_n_terms + 2];
        neg_f[i * 2 + 0] = ( xs[i] *h4 - h2 ) / sum_Q;
        neg_f[i * 2 + 1] = (ys[i] *h4 - h3 ) / sum_Q;

        dC[i * 2 + 0] = (pos_f[i * 2] - neg_f[i * 2]);
        dC[i * 2 + 1] = (pos_f[i * 2 + 1] - neg_f[i * 2 + 1]);


    }

    this->current_sum_Q = sum_Q;

/*        FILE *fp = nullptr;
        char buffer[500];
        sprintf(buffer, "temp/fft_gradient%d.txt", itTest);
        fp = fopen(buffer, "w"); // Open file for writing
        for (int i = 0; i < N; i++) {
                fprintf(fp, "%d,%.12e,%.12e\n", i, neg_f[i * 2] , neg_f[i * 2 + 1]);
        }
        fclose(fp);*/

    delete[] pos_f;
    delete[] neg_f;
    delete[] SquaredPotentialsQij;
    delete[] NotSquaredPotentialsQij;
    delete[] SquaredChargesQij;
    delete[] NotSquaredChargesQij;
    delete[] xs;
    delete[] ys;
    delete[] box_lower_bounds;
    delete[] box_upper_bounds;
    delete[] y_tilde_spacings;
    delete[] y_tilde;
    delete[] x_tilde;
    delete[] fft_kernel_tilde;
}

// Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation
void TSNE::computeFftGradient(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                              int N, int D, double *dC, int n_interpolation_points, double intervals_per_integer,
                              int min_num_intervals, unsigned int nthreads) {


    // Zero out the gradient
    for (int i = 0; i < N * D; i++) dC[i] = 0.0;

    // For convenience, split the x and y coordinate values
    auto *xs = new double[N];
    auto *ys = new double[N];

    double min_coord = INFINITY;
    double max_coord = -INFINITY;
    // Find the min/max values of the x and y coordinates
    for (unsigned long i = 0; i < N; i++) {
        xs[i] = Y[i * 2 + 0];
        ys[i] = Y[i * 2 + 1];
        if (xs[i] > max_coord) max_coord = xs[i];
        else if (xs[i] < min_coord) min_coord = xs[i];
        if (ys[i] > max_coord) max_coord = ys[i];
        else if (ys[i] < min_coord) min_coord = ys[i];
    }

    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 4;
    auto *chargesQij = new double[N * n_terms];
    auto *potentialsQij = new double[N * n_terms]();

    // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
    for (unsigned long j = 0; j < N; j++) {
        chargesQij[j * n_terms + 0] = 1;
        chargesQij[j * n_terms + 1] = xs[j];
        chargesQij[j * n_terms + 2] = ys[j];
        chargesQij[j * n_terms + 3] = xs[j] * xs[j] + ys[j] * ys[j];
    }

    // Compute the number of boxes in a single dimension and the total number of boxes in 2d
    auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));


    // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
    // arbitrary
    int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
    if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
        //Round up to nearest grid point
        int chosen_i;
        for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

    auto *box_lower_bounds = new double[2 * n_boxes];
    auto *box_upper_bounds = new double[2 * n_boxes];
    auto *y_tilde_spacings = new double[n_interpolation_points];
    int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
    auto *x_tilde = new double[n_interpolation_points_1d]();
    auto *y_tilde = new double[n_interpolation_points_1d]();
    auto *fft_kernel_tilde = new complex<double>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];


    INITIALIZE_TIME;
    START_TIME;
    precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
                  &squared_cauchy_2d,
                  box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde,1.0);
    n_body_fft_2d(N, n_terms, xs, ys, chargesQij, n_boxes_per_dim, n_interpolation_points, box_lower_bounds,
                  box_upper_bounds, y_tilde_spacings, fft_kernel_tilde, potentialsQij,nthreads);

    // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
    // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
    // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
    double sum_Q = 0;
    for (unsigned long i = 0; i < N; i++) {
        double phi1 = potentialsQij[i * n_terms + 0];
        double phi2 = potentialsQij[i * n_terms + 1];
        double phi3 = potentialsQij[i * n_terms + 2];
        double phi4 = potentialsQij[i * n_terms + 3];

        sum_Q += (1 + xs[i] * xs[i] + ys[i] * ys[i]) * phi1 - 2 * (xs[i] * phi2 + ys[i] * phi3) + phi4;
    }
    sum_Q -= N;

    this->current_sum_Q = sum_Q;

    double *pos_f = new double[N * 2];

    END_TIME("Total Interpolation");
        START_TIME;
    // Now, figure out the Gaussian component of the gradient. This corresponds to the "attraction" term of the
    // gradient. It was calculated using a fast KNN approach, so here we just use the results that were passed to this
    // function
                    PARALLEL_FOR(nthreads, N, {
                                double dim1 = 0;
                                double dim2 = 0;

                                for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++) {
                                // Compute pairwise distance and Q-value
                                    unsigned int ind3 = inp_col_P[i];
                                    double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
                                    double q_ij = 1 / (1 + d_ij);

                                    dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
                                    dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
                                }
                                pos_f[loop_i * 2 + 0] = dim1;
                                pos_f[loop_i * 2 + 1] = dim2;

                            });
    END_TIME("Attractive Forces");
    //printf("Attractive forces took %lf\n", (diff(start20,end20))/(double)1E6);
                            






    // Make the negative term, or F_rep in the equation 3 of the paper
    double *neg_f = new double[N * 2];
    for (unsigned int i = 0; i < N; i++) {
        neg_f[i * 2 + 0] = (xs[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 1]) / sum_Q;
        neg_f[i * 2 + 1] = (ys[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 2]) / sum_Q;

        dC[i * 2 + 0] = pos_f[i * 2] - neg_f[i * 2];
        dC[i * 2 + 1] = pos_f[i * 2 + 1] - neg_f[i * 2 + 1];
    }

    delete[] pos_f;
    delete[] neg_f;
    delete[] potentialsQij;
    delete[] chargesQij;
    delete[] xs;
    delete[] ys;
    delete[] box_lower_bounds;
    delete[] box_upper_bounds;
    delete[] y_tilde_spacings;
    delete[] y_tilde;
    delete[] x_tilde;
    delete[] fft_kernel_tilde;
}


void TSNE::computeExactGradientTest(double *Y, int N, int D, double df ) {
  // Compute the squared Euclidean distance matrix
    double *DD = (double *) malloc(N * N * sizeof(double));
    if (DD == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double *Q = (double *) malloc(N * N * sizeof(double));
    if (Q == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    double sum_Q = .0;
    int nN = 0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            if (n != m) {
                Q[nN + m] = 1.0 / pow(1.0 + DD[nN + m]/(double)df, (df));
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

    // Perform the computation of the gradient
    char buffer[500];
    sprintf(buffer, "temp/exact_gradient%d.txt", itTest);
    FILE *fp = fopen(buffer, "w"); // Open file for writing
    nN = 0;
    int nD = 0;
    for (int n = 0; n < N; n++) {
        double testQij = 0;
        double testPos = 0;
        double testNeg1 = 0;
        double testNeg2 = 0;
        double testdC = 0;
        int mD = 0;
        for (int m = 0; m < N; m++) {
            if (n != m) {
                testNeg1 += pow(Q[nN + m],(df +1.0)/df) * (Y[nD + 0] - Y[mD + 0]) / sum_Q;
                testNeg2 += pow(Q[nN + m],(df +1.0)/df) * (Y[nD + 1] - Y[mD + 1]) / sum_Q;
            }
            mD += D;
        }
        fprintf(fp, "%d, %.12e, %.12e\n", n, testNeg1,testNeg2);

        nN += N;
        nD += D;
    }
    fclose(fp);
    free(DD);
    free(Q);

}


// Compute the exact gradient of the t-SNE cost function
void TSNE::computeExactGradient(double *P, double *Y, int N, int D, double *dC, double df) {
    // Make sure the current gradient contains zeros
    for (int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    auto *DD = (double *) malloc(N * N * sizeof(double));
    if (DD == nullptr) throw std::bad_alloc();
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    auto *Q = (double *) malloc(N * N * sizeof(double));
    if (Q == nullptr) throw std::bad_alloc();

    auto *Qpow = (double *) malloc(N * N * sizeof(double));
    if (Qpow == nullptr) throw std::bad_alloc();

    double sum_Q = .0;
    int nN = 0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            if (n != m) {
                //Q[nN + m] = 1.0 / pow(1.0 + DD[nN + m]/(double)df, df);
                Q[nN + m] = 1.0 / (1.0 + DD[nN + m]/(double)df);
                Qpow[nN + m] = pow(Q[nN + m], df);
                sum_Q += Qpow[nN + m];
            }
        }
        nN += N;
    }

    // Perform the computation of the gradient
    nN = 0;
    int nD = 0;
    for (int n = 0; n < N; n++) {
        int mD = 0;
        for (int m = 0; m < N; m++) {
            if (n != m) {
                double mult = (P[nN + m] - (Qpow[nN + m] / sum_Q)) * (Q[nN + m]);
                for (int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
        }
        nN += N;
        nD += D;
    }
    free(Q);
    free(Qpow);
    free(DD);
}


// Evaluate t-SNE cost function (exactly)
double TSNE::evaluateError(double *P, double *Y, int N, int D, double df) {
    // Compute the squared Euclidean distance matrix
    double *DD = (double *) malloc(N * N * sizeof(double));
    double *Q = (double *) malloc(N * N * sizeof(double));
    if (DD == NULL || Q == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            if (n != m) {
                //Q[nN + m] = 1.0 / pow(1.0 + DD[nN + m]/(double)df, df);
                Q[nN + m] = 1.0 / (1.0 + DD[nN + m]/(double)df);
                Q[nN +m ] = pow(Q[nN +m ], df);
                sum_Q += Q[nN + m];
            } else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    //printf("sum_Q: %e", sum_Q);
    for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;
    //  for (int i = 0; i < N; i++) printf("Q[%d]: %e\n", i, Q[i]);

//printf("Q[N*N/2+1]: %e, Q[N*N-1]: %e\n", Q[N*N/2+1], Q[N*N/2+2]);

    // Sum t-SNE error
    double C = .0;
    for (int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
    }

    // Clean up memory
    free(DD);
    free(Q);
    return C;
}

// Evaluate t-SNE cost function (approximately) using FFT
double TSNE::evaluateErrorFft(unsigned int *row_P, unsigned int *col_P, double *val_P, double *Y, int N, int D,unsigned int nthreads, double df) {
    // Get estimate of normalization term

    double sum_Q = this->current_sum_Q;

    // Loop over all edges to compute t-SNE error
    double C = .0;
        PARALLEL_FOR(nthreads,N,{
        double *buff = (double *) calloc(D, sizeof(double));
        int ind1 = loop_i * D;
        double temp = 0;
        for (int i = row_P[loop_i]; i < row_P[loop_i + 1]; i++) {
            double Q = .0;
            int ind2 = col_P[i] * D;
            for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
            for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = pow(1.0 / (1.0 + Q/df),  df) / sum_Q;
            temp += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
        C += temp;
        free(buff);
    });

    // Clean up memory
    return C;
}


// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned int *row_P, unsigned int *col_P, double *val_P, double *Y, int N, int D,
                           double theta, unsigned int nthreads) {
    // Get estimate of normalization term
    SPTree *tree = new SPTree(D, Y, N);
    double *buff = (double *) calloc(D, sizeof(double));
    double sum_Q = .0;
    for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    double C = .0;
        PARALLEL_FOR(nthreads,N,{
        double *buff = (double *) calloc(D, sizeof(double));
        int ind1 = loop_i * D;
        double temp = 0;
        for (int i = row_P[loop_i]; i < row_P[loop_i + 1]; i++) {
            double Q = .0;
            int ind2 = col_P[i] * D;
            for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
            for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            temp += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
        C += temp;
        free(buff);
    });

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Converts an array of [squared] Euclidean distances into similarities aka affinities
// using a specified perplexity value (or a specified kernel width)
double TSNE::distances2similarities(double *D, double *P, int N, int n, double perplexity, double sigma, bool ifSquared)  {

    /* D          - a pointer to the array of distances
       P          - a pointer to the array of similarities
       N          - length of D and P
       n          - index of the point that should have D = 0
       perplexity - target perplexity
       sigma      - kernel width if perplexity == -1
       ifSquared  - if D contains squared distances (TRUE) or not (FALSE) */

    double sum_P;
    double beta;

    if (perplexity > 0) {
        // Using binary search to find the appropriate kernel width
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;
        int max_iter = 200;
        int iter = 0;
        bool found = false;
        beta = 1.0;

        // Iterate until we found a good kernel width
        while(!found && iter < max_iter) {
            // Apply Gaussian kernel
            for(int m = 0; m < N; m++) P[m] = exp(-beta * (ifSquared ? D[m] : D[m]*D[m]));
            if (n>=0) P[n] = DBL_MIN;

            // Compute entropy
            sum_P = DBL_MIN;
            for(int m = 0; m < N; m++) sum_P += P[m];
            double H = 0.0;
            for(int m = 0; m < N; m++) H += beta * ((ifSquared ? D[m] : D[m]*D[m]) * P[m]);
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
    }else{
        // Using fixed kernel width: no iterations needed
        beta = 1/(2*sigma*sigma);
        for(int m = 0; m < N; m++) P[m] = exp(-beta * (ifSquared ? D[m] : D[m]*D[m]));
        if (n >= 0) P[n] = DBL_MIN;

        sum_P = DBL_MIN;
        for(int m = 0; m < N; m++) sum_P += P[m];
    }

    // Normalize
    for(int m = 0; m < N; m++) P[m] /= sum_P;

    return beta;
}


// Converts an array of [squared] Euclidean distances into similarities aka affinities
// using a list of perplexities
double TSNE::distances2similarities(double *D, double *P, int N, int n, double perplexity, double sigma, bool ifSquared, 
                                    int perplexity_list_length, double *perplexity_list)  {

    // if perplexity != 0 then defaulting to using this perplexity (or fixed sigma)
    if (perplexity != 0) {
        double beta = distances2similarities(D, P, N, n, perplexity, sigma, true);
        return beta;
    }

    // otherwise averaging similarities using all perplexities in perplexity_list
	double *tmp = (double*) malloc(N * sizeof(double));
	if(tmp == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	double beta = distances2similarities(D, P, N, n, perplexity_list[0], sigma, true);

	for (int m=1; m < perplexity_list_length; m++){
		beta = distances2similarities(D, tmp, N, n, perplexity_list[m], sigma, true);
		for (int i=0; i<N; i++){
			P[i] += tmp[i];
		}
	}

	for (int i=0; i<N; i++){
		P[i] /= perplexity_list_length;
	}

	return beta;
}


// Compute input similarities using exact algorithm
void TSNE::computeGaussianPerplexity(double *X, int N, int D, double *P, double perplexity, double sigma,
                                     int perplexity_list_length, double *perplexity_list) {
    if (perplexity < 0) {
        printf("Using manually set kernel width\n");
    } else {
        printf("Using perplexity, not the manually set kernel width\n");
    }

    // Compute the squared Euclidean distance matrix
    double *DD = (double *) malloc(N * N * sizeof(double));
    if (DD == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    computeSquaredEuclideanDistance(X, N, D, DD);

    // Convert distances to similarities using Gaussian kernel row by row
    int nN = 0;
    double beta;
    for (int n = 0; n < N; n++) {
        beta = distances2similarities(&DD[nN], &P[nN], N, n, perplexity, sigma, true, 
                                      perplexity_list_length, perplexity_list);
        nN += N;
    }

    // Clean up memory
    free(DD);
}


// Compute input similarities using ANNOY
int TSNE::computeGaussianPerplexity(double *X, int N, int D, unsigned int **_row_P, unsigned int **_col_P,
                                    double **_val_P, double perplexity, int K, double sigma, int num_trees, 
                                    int search_k, unsigned int nthreads, int perplexity_list_length, 
                                    double *perplexity_list, int rand_seed) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    printf("Going to allocate memory. N: %d, K: %d, N*K = %d\n", N, K, N*K);
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    printf("Building Annoy tree...\n");
    AnnoyIndex<int, double, Euclidean, Kiss32Random> tree = AnnoyIndex<int, double, Euclidean, Kiss32Random>(D);

    if (rand_seed > 0)
    {
        tree.set_seed(rand_seed);
    }

    for(int i=0; i<N; ++i){
        double *vec = (double *) malloc( D * sizeof(double) );

        for(int z=0; z<D; ++z){
            vec[z] = X[i*D+z];
        }

        tree.add_item(i, vec);
    }
    tree.build(num_trees);
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
    printf("Done building tree. Beginning nearest neighbor search... \n");
    ProgressBar bar(N,60);
    //const size_t nthreads = 1;
    {
        // Pre loop
        std::cout << "parallel (" << nthreads << " threads):" << std::endl;
        std::vector<std::thread> threads(nthreads);
        for (int t = 0; t < nthreads; t++) {
            threads[t] = std::thread(std::bind(
                    [&](const int bi, const int ei, const int t)
                    {
                        // loop over all items
                        for(int n = bi;n<ei;n++)
                        {
                            // inner loop
                            {
                                // Find nearest neighbors
                                std::vector<int> closest;
                                std::vector<double> closest_distances;
                                tree.get_nns_by_item(n, K+1, search_k, &closest, &closest_distances);

                                double* cur_P = (double*) malloc((N - 1) * sizeof(double));
                                if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

                                double beta = distances2similarities(&closest_distances[1], cur_P, K, -1, perplexity, sigma, false,
                                                                     perplexity_list_length, perplexity_list);
                                ++bar;
                                if(t == 0 && n % 100 == 0) {
                                    bar.display();
                                //    if (perplexity >= 0) {
                                //        printf(" - point %d of %d, most recent beta calculated is %lf \n", n, N, beta);
                                //    } else {
                                //        printf(" - point %d of %d, beta is set to %lf \n", n, N, 1/sigma);
                                //    }
                                }

                                // Store current row of P in matrix
                                for(unsigned int m = 0; m < K; m++) {
                                    col_P[row_P[n] + m] = (unsigned int) closest[m + 1];
                                    val_P[row_P[n] + m] = cur_P[m];
                                }

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
    bar.display();
    printf("\n");

    return 0;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double *X, int N, int D, unsigned int **_row_P, unsigned int **_col_P, 
                                     double **_val_P, double perplexity, int K, double sigma, unsigned int nthreads,
                                     int perplexity_list_length, double *perplexity_list) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    printf("Going to allocate memory. N: %d, K: %d, N*K = %d\n", N, K, N*K);
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
    printf("Building VP tree...\n");
    VpTree<DataPoint, euclidean_distance> *tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for (int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);
    printf("Done building tree. Beginning nearest neighbor search... \n");


    ProgressBar bar(N,60);

    //const size_t nthreads = 1;
    {
        // Pre loop
        std::cout << "parallel (" << nthreads << " threads):" << std::endl;
        std::vector<std::thread> threads(nthreads);
        for (int t = 0; t < nthreads; t++) {
            threads[t] = std::thread(std::bind(
                    [&](const int bi, const int ei, const int t)
                    {
                        // loop over all items
                        for(int n = bi;n<ei;n++)
                        {
                            // inner loop
                            {
                                // Find nearest neighbors
                                std::vector<double> cur_P(K);
                                vector<DataPoint> indices;
                                vector<double> distances;
                                tree->search(obj_X[n], K + 1, &indices, &distances);

                                double beta = distances2similarities(&distances[1], &cur_P[0], K, -1, perplexity, sigma, false,
                                                                     perplexity_list_length, perplexity_list);
                                ++bar;
                                if(t == 0 && n % 100 == 0) {
                                    bar.display();
                                //    if (perplexity >= 0) {
                                //        printf(" - point %d of %d, most recent beta calculated is %lf \n", n, N, beta);
                                //    } else {
                                //        printf(" - point %d of %d, beta is set to %lf \n", n, N, 1/sigma);
                                //    }
                                }

                                for(unsigned int m = 0; m < K; m++) {
                                    col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
                                    val_P[row_P[n] + m] = cur_P[m];
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

    bar.display();
    printf("\n");
    // Clean up memory
    obj_X.clear();
    delete tree;
}


// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(unsigned int **_row_P, unsigned int **_col_P, double **_val_P, int N) {
    // Get sparse matrix
    unsigned int *row_P = *_row_P;
    unsigned int *col_P = *_col_P;
    double *val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int *row_counts = (int *) calloc(N, sizeof(int));
    if (row_counts == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) present = true;
            }
            if (present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int *sym_row_P = (unsigned int *) malloc((N + 1) * sizeof(unsigned int));
    unsigned int *sym_col_P = (unsigned int *) malloc(no_elem * sizeof(unsigned int));
    double *sym_val_P = (double *) malloc(no_elem * sizeof(double));
    if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int *offset = (int *) calloc(N, sizeof(int));
    if (offset == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    for (int n = 0; n < N; n++) {
        for (unsigned int i = row_P[n];
             i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    if (n <=
                        col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present) {
                sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || (n <= col_P[i])) {
                offset[n]++;
                if (col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P);
    *_row_P = sym_row_P;
    free(*_col_P);
    *_col_P = sym_col_P;
    free(*_val_P);
    *_val_P = sym_val_P;

    // Free up some memory
    free(offset);
    free(row_counts);
}


// Compute squared Euclidean distance matrix
void TSNE::computeSquaredEuclideanDistance(double *X, int N, int D, double *DD) {
    const double *XnD = X;
    for (int n = 0; n < N; ++n, XnD += D) {
        const double *XmD = XnD + D;
        double *curr_elem = &DD[n * N + n];
        *curr_elem = 0.0;
        double *curr_elem_sym = curr_elem + N;
        for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
            *(++curr_elem) = 0.0;
            for (int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
void TSNE::zeroMean(double *X, int N, int D) {
    // Compute data mean
    double *mean = (double *) calloc(D, sizeof(double));
    if (mean == NULL) throw std::bad_alloc();

    int nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double) N;
    }

    // Subtract data mean
    nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
        }
        nD += D;
    }
    free(mean);
}


// Generates a Gaussian random number
double TSNE::randn() {
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    return x;
}


// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(const char *data_path, double **data, double **Y, int *n,
	int *d, int *no_dims, double *theta, double *perplexity, int *rand_seed,
	int *max_iter, int *stop_lying_iter, int *mom_switch_iter, double* momentum, 
    double* final_momentum, double* learning_rate, int *K, double *sigma,
	int *nbody_algo, int *knn_algo, double *early_exag_coeff,
	int *no_momentum_during_exag, int *n_trees, int *search_k,
	int *start_late_exag_iter, double *late_exag_coeff, int *nterms,
	double *intervals_per_integer, int *min_num_intervals,
	bool *skip_random_init, int *load_affinities,
    int *perplexity_list_length, double **perplexity_list, double * df) {

	FILE *h;
	if((h = fopen(data_path, "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}

	size_t result; // need this to get rid of warnings that otherwise appear

	result = fread(n, sizeof(int), 1, h);     		            // number of datapoints
	result = fread(d, sizeof(int), 1, h);	  		            // original dimensionality
	result = fread(theta, sizeof(double), 1, h);		        // gradient accuracy
	result = fread(perplexity, sizeof(double), 1, h);	        // perplexity

    // if perplexity == 0, then what follows is the number of perplexities 
    // to combine and then the list of these perpexities
    if (*perplexity == 0) {
        result = fread(perplexity_list_length, sizeof(int), 1, h);
        *perplexity_list = (double*) malloc(*perplexity_list_length * sizeof(double));
        if(*perplexity_list == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        result = fread(*perplexity_list, sizeof(double), *perplexity_list_length, h);                      
    } else {
        perplexity_list_length = 0;
        perplexity_list = NULL;
    }

	result = fread(no_dims, sizeof(int), 1, h);                 // output dimensionality
	result = fread(max_iter, sizeof(int),1,h);                  // maximum number of iterations
	result = fread(stop_lying_iter, sizeof(int),1,h);           // when to stop early exaggeration
	result = fread(mom_switch_iter, sizeof(int),1,h);           // when to switch the momentum value
	result = fread(momentum, sizeof(double),1,h);               // initial momentum
	result = fread(final_momentum, sizeof(double),1,h);         // final momentum
	result = fread(learning_rate, sizeof(double),1,h);          // learning rate
	result = fread(K, sizeof(int),1,h);                         // number of neighbours to compute
	result = fread(sigma, sizeof(double),1,h);                  // input kernel width
	result = fread(nbody_algo, sizeof(int),1,h);                // Barnes-Hut or FFT
	result = fread(knn_algo, sizeof(int),1,h);                  // VP-trees or Annoy
	result = fread(early_exag_coeff, sizeof(double),1,h);       // early exaggeration
	result = fread(no_momentum_during_exag, sizeof(int),1,h);   // if to use momentum during early exagg
	result = fread(n_trees, sizeof(int),1,h);                   // number of trees for Annoy
	result = fread(search_k, sizeof(int),1,h);                  // search_k for Annoy
	result = fread(start_late_exag_iter, sizeof(int),1,h);      // when to start late exaggeration
	result = fread(late_exag_coeff, sizeof(double),1,h);        // late exaggeration
	result = fread(nterms, sizeof(int),1,h);                    // FFT parameter
	result = fread(intervals_per_integer, sizeof(double),1,h);  // FFT parameter
	result = fread(min_num_intervals, sizeof(int),1,h);         // FFT parameter

    if((*nbody_algo == 2) && (*no_dims > 2)){
        printf("FFT interpolation scheme supports only 1 or 2 output dimensions, not %d\n", *no_dims);
        exit(1);
    }

	*data = (double*) malloc(*d * *n * sizeof(double));
	if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	result = fread(*data, sizeof(double), *n * *d, h);          // the data
	if(!feof(h)) {
		result = fread(rand_seed, sizeof(int), 1, h);       // random seed
	}
	if(!feof(h)) {
            result = fread(df, sizeof(double),1,h);  // Number of degrees of freedom of the kernel
        }
	if(!feof(h)) {
		result = fread(load_affinities, sizeof(int), 1, h); // to load or to save affinities
	}

	// allocating space for the t-sne solution
	*Y = (double*) malloc(*n * *no_dims * sizeof(double));
	if(*Y == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	// if the file has not ended, the remaining part is the initialization
	if(!feof(h)){
		result = fread(*Y, sizeof(double), *n * *no_dims, h);
		if(result < *n * *no_dims){
			*skip_random_init = false;
		}else{
			*skip_random_init = true;
		}
	} else{
		*skip_random_init = false;
	}
	fclose(h);
	printf("Read the following parameters:\n\t n %d by d %d dataset, theta %lf,\n"
			"\t perplexity %lf, no_dims %d, max_iter %d,\n"
			"\t stop_lying_iter %d, mom_switch_iter %d,\n"
            "\t momentum %lf, final_momentum %lf,\n"
            "\t learning_rate %lf, K %d, sigma %lf, nbody_algo %d,\n"
			"\t knn_algo %d, early_exag_coeff %lf,\n"
			"\t no_momentum_during_exag %d, n_trees %d, search_k %d,\n"
			"\t start_late_exag_iter %d, late_exag_coeff %lf\n"
			"\t nterms %d, interval_per_integer %lf, min_num_intervals %d, t-dist df %lf\n",
			*n, *d, *theta, *perplexity,
			*no_dims, *max_iter,*stop_lying_iter,
            *mom_switch_iter, *momentum, *final_momentum, *learning_rate,
			*K, *sigma, *nbody_algo, *knn_algo, *early_exag_coeff,
			*no_momentum_during_exag, *n_trees, *search_k,
			*start_late_exag_iter, *late_exag_coeff,
			*nterms, *intervals_per_integer, *min_num_intervals, *df);

	printf("Read the %i x %i data matrix successfully. X[0,0] = %lf\n", *n, *d, *data[0]);

    if(*perplexity == 0){
        printf("Read the list of perplexities: ");
        for (int m=0; m<*perplexity_list_length; m++){
            printf("%f ", (*perplexity_list)[m]);
        }
        printf("\n");
    }

	if(*skip_random_init){
		printf("Read the initialization successfully.\n");
	}

	return true;
}


// Function that saves map to a t-SNE file
void TSNE::save_data(const char *result_path, double* data, double* costs, int n, int d, int max_iter) {
	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen(result_path, "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fwrite(&max_iter, sizeof(int), 1, h);
	fwrite(costs, sizeof(double), max_iter, h);
	fclose(h);
	printf("Wrote the %i x %i data matrix successfully.\n", n, d);
}


int main(int argc, char *argv[]) {
        const char version_number[] =  "1.1.0";
	printf("=============== t-SNE v%s ===============\n", version_number);

	// Define some variables
	int N, D, no_dims, max_iter, stop_lying_iter;
	int K, nbody_algo, knn_algo, no_momentum_during_exag;
        int mom_switch_iter;
        double momentum, final_momentum, learning_rate;
	int n_trees, search_k, start_late_exag_iter;
	double sigma, early_exag_coeff, late_exag_coeff;
	double perplexity, theta, *data, *initial_data;
	int nterms, min_num_intervals;
	double intervals_per_integer;
	int rand_seed = 0;
	int load_affinities = 0;
	const char *data_path, *result_path;
	unsigned int nthreads;
	TSNE* tsne = new TSNE();

	double *Y;
	bool skip_random_init;

    double *perplexity_list;
    int perplexity_list_length;
    double df;

	data_path = "data.dat";
	result_path = "result.dat";
	nthreads = 0;
        if (argc >=2 ) {
            if ( strcmp(argv[1],version_number)) {
                std::cout<<"Wrapper passed wrong version number: "<< argv[1] <<std::endl;
                exit(-1);
            }
        }else{
                std::cout<<"Please pass version number as first argument." <<std::endl;
                exit(-1);
            
        }
	if(argc >= 3) {
		data_path = argv[2];
	}
	if(argc >= 4) {
		result_path = argv[3];
	}
	if(argc >= 5) {
		nthreads = (unsigned int)strtoul(argv[3], (char **)NULL, 10);
	}
    if (nthreads == 0) {
        nthreads = std::thread::hardware_concurrency();
    }
	std::cout<<"fast_tsne data_path: "<< data_path <<std::endl;
	std::cout<<"fast_tsne result_path: "<< result_path <<std::endl;
	std::cout<<"fast_tsne nthreads: "<< nthreads <<std::endl;

	// Read the parameters and the dataset
	if(tsne->load_data(data_path, &data, &Y, &N, &D, &no_dims, &theta, &perplexity,
				&rand_seed, &max_iter, &stop_lying_iter, &mom_switch_iter, &momentum,
                &final_momentum, &learning_rate, &K,
				&sigma, &nbody_algo, &knn_algo,
				&early_exag_coeff, &no_momentum_during_exag,
				&n_trees, &search_k, &start_late_exag_iter,
				&late_exag_coeff, &nterms, &intervals_per_integer,
				&min_num_intervals, &skip_random_init, &load_affinities,
                &perplexity_list_length, &perplexity_list, &df)) {

		bool no_momentum_during_exag_bool = true;
		if (no_momentum_during_exag == 0) no_momentum_during_exag_bool = false;

		// Now fire up the SNE implementation
		double* costs = (double*) calloc(max_iter, sizeof(double));
		if(costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
		int error_code = 0;
		error_code = tsne->run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, skip_random_init, max_iter, 
				stop_lying_iter, mom_switch_iter, momentum, final_momentum, learning_rate, K, sigma, nbody_algo, knn_algo, 
                early_exag_coeff, costs, no_momentum_during_exag_bool, start_late_exag_iter, late_exag_coeff, n_trees,search_k, 
				nterms, intervals_per_integer, min_num_intervals, nthreads, load_affinities,
                perplexity_list_length, perplexity_list, df);

		if (error_code < 0) {
			exit(error_code);
		}

		// Save the results
		tsne->save_data(result_path, Y, costs, N, no_dims, max_iter);

		// Clean up the memory
		free(data); data = NULL;
		free(Y); Y = NULL;
		free(costs); costs = NULL;
	}
	delete(tsne);

	printf("Done.\n\n");
}
