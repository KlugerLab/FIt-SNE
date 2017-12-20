#ifndef NBODYFFT_H
#define NBODYFFT_H
#include <fftw3.h>
	typedef double (*kerneltype)(double, double, double, double);
	typedef double (*kerneltype2d)(double, double,double,double, double, double);
	int precompute2(double xmax, double xmin, double ymax, double ymin, int nlat, int nterms, kerneltype2d ker,double * band,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall,double *yptsall,int *irearr, fftw_complex * zkvalf );
	int nbodyfft2(int n, int ndim, double* xs, double * ys, double * charges, int nlat, int nterms,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall, double *yptsall,int* irearr, fftw_complex * zkvalf, double * pot);
	int precompute(double rmin, double rmax, int nboxes, int nterms, kerneltype ker,double * band,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall, fftw_complex * zkvalf );

	int nbodyfft(int n, int ndim, double* locs, double * charges, int nboxes, int nterms,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall, fftw_complex * zkvalf, double * pot);
#endif
