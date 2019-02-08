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


#ifndef TSNE_H
#define TSNE_H

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class TSNE {

public:
    int run(double *X, int N, int D, double *Y, int no_dims, double perplexity, double theta, int rand_seed,
            bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
            double momentum, double final_momentum, double learning_rate, int K, double sigma,
            int nbody_algorithm, int knn_algo, double early_exag_coeff, double *costs,
            bool no_momentum_during_exag, int start_late_exag_iter, double late_exag_coeff, int n_trees, int search_k,
            int nterms, double intervals_per_integer, int min_num_intervals, unsigned int nthreads, int load_affinities,
            int perplexity_list_length, double *perplexity_list, double df );

    bool load_data(const char *data_path, double **data, double **Y, int *n, int *d, int *no_dims, double *theta,
            double *perplexity, int *rand_seed, int *max_iter, int *stop_lying_iter, 
            int *mom_switch_iter, double* momentum, double* final_momentum, double* learning_rate, int *K, double *sigma,
            int *nbody_algo, int *knn_algo, double* early_exag_coeff,  int *no_momentum_during_exag, int *n_trees,
            int *search_k, int *start_late_exag_iter, double *late_exag_coeff, int *nterms,
            double *intervals_per_integer, int *min_num_intervals, bool *skip_random_init, int *load_affinities,
            int *perplexity_list_length, double **perplexity_list,double *df);

    void save_data(const char *result_path, double *data, double *costs, int n, int d, int max_iter);

    void symmetrizeMatrix(unsigned int **row_P, unsigned int **col_P, double **val_P, int N); // should be static!

private:
    double current_sum_Q;
    void computeGradient(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                         int N, int D, double *dC, double theta);

    void computeFftGradientVariableDf(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                                int N, int D, double *dC, int n_interpolation_points, double intervals_per_integer,
                                int min_num_intervals, unsigned int nthreads, double df);

    void computeFftGradient(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P, double *Y,
                                int N, int D, double *dC, int n_interpolation_points, double intervals_per_integer,
                                int min_num_intervals, unsigned int nthreads);

    void computeFftGradientOneD(double *P, unsigned int *inp_row_P, unsigned int *inp_col_P, double *inp_val_P,
                                    double *Y, int N, int D, double *dC, int n_interpolation_points,
                                    double intervals_per_integer, int min_num_intervals, unsigned int nthreads);

    void computeExactGradient(double *P, double *Y, int N, int D, double *dC, double df);

    void computeExactGradientTest(double *Y, int N, int D, double df);

    double evaluateError(double *P, double *Y, int N, int D, double df);

    double evaluateError(unsigned int *row_P, unsigned int *col_P, double *val_P, double *Y, int N, int D,
                         double theta, unsigned int nthreads);

    double evaluateErrorFft(unsigned int *row_P, unsigned int *col_P, double *val_P, double *Y, int N, int D, unsigned int nthreads, double df);
    void zeroMean(double *X, int N, int D);

	double distances2similarities(double *D, double *P, int N, int n, double perplexity, double sigma, bool ifSquared);

	double distances2similarities(double *D, double *P, int N, int n, double perplexity, double sigma, bool ifSquared,
                                  int perplexity_list_length, double *perplexity_list);

    void computeGaussianPerplexity(double *X, int N, int D, double *P, double perplexity, double sigma,
                                   int perplexity_list_length, double *perplexity_list);

    void computeGaussianPerplexity(double *X, int N, int D, unsigned int **_row_P, unsigned int **_col_P,
                                   double **_val_P, double perplexity, int K, double sigma, unsigned int nthreads,
                                   int perplexity_list_length, double *perplexity_list);

    int computeGaussianPerplexity(double *X, int N, int D, unsigned int **_row_P, unsigned int **_col_P,
                                  double **_val_P, double perplexity, int K, double sigma, int num_trees, int search_k,
                                  unsigned int nthreads, int perplexity_list_length, double *perplexity_list,
                                  int rand_seed);

    void computeSquaredEuclideanDistance(double *X, int N, int D, double *DD);

    double randn();
};

#endif
