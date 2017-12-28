#include "nbodyfft.h"
#define PI 3.14159265358979323846

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int precompute2( double xmax, double xmin, double ymax, double ymin,  int nlat, int nterms, kerneltype2d ker, double * band,double * boxl, double * boxr,  double * prods, double * xpts, double * xptsall, double *yptsall, int * irearr, fftw_complex * zkvalf ) {

	/*
	 * Set up the boxes
	 */
	int nboxes = nlat*nlat;
	double boxwidth = (xmax - xmin)/(double) nlat;

	//Left and right bounds of each box
	int nn = nboxes*nterms*nterms;
	if (boxl == NULL) {
		printf("Malloc failed\n"); 
		exit(-1);
	}

	int ii=0;
	//printf("boxwidth: %lf, nlat %d, xmax %f xmin %f\n", boxwidth, nlat, xmax, xmin);
	for (int i = 0; i < nlat; i++){
		for (int j = 0; j < nlat; j++){
			boxl[0*nboxes + ii] = j*boxwidth + xmin;
			boxr[0*nboxes + ii] = (j+1)*(boxwidth) + xmin;

			boxl[1*nboxes + ii] = i*boxwidth + ymin;
			boxr[1*nboxes + ii] = (i+1)*(boxwidth) + ymin;
			//printf("box %d, %lf to %lf and %lf %lf \n", i, boxl[i], boxr[i] , boxl[nboxes+ii], boxr[nboxes+ii]);
			ii++;
		}
	}

	//Coordinates of each (equispaced) interpolation node for a single box
	double h = 2/(double)nterms;

	xpts[0] = -1 + h/2.0;
	for (int i=1; i< nterms; i++){
		xpts[i] = xpts[0] + (i)*h;
	}
	/*
	 * Interpolate kernel using lagrange polynomials
	 */

	//Get the denominators for the lagrange polynomials (this is getprods())
	for (int i= 0; i< nterms; i ++ ){
		prods[i] = 1;
		for (int j=0;j < nterms; j ++ ){
			if (i != j) {
				prods[i] = prods[i] * (xpts[i] - xpts[j]);
			}
		}
//		printf("Prods[%d] xpts[%d] = %lf, %lf\n", i,i, prods[i], xpts[i]);
	}


	//Coordinates of each (equispaced) interpolation node for all boxes
	int nfourh = nterms*nlat;
	int nfour = 2*nterms*nlat;
	h = h*boxwidth/2;
	double xstart = xmin + h/2;
	double ystart = ymin + h/2;


	ii = 0;
	for (int i= 0; i< nfourh; i ++ ){
		for (int j= 0; j< nfourh; j ++ ){
			xptsall[ii] = xstart + (i)*h;
			yptsall[ii] = ystart + (j)*h;
			//printf("%d xptsall %lf, %lf \n", ii, xptsall[ii],  yptsall[ii]);
			ii++;
		}
	}

	ii = 0;
	for (int ilat = 0; ilat < nlat; ilat++){
		for (int jlat = 0; jlat < nlat; jlat++){
			for (int i = 0; i < nterms; i++){
				for (int j = 0; j < nterms; j++){
					int iy = (ilat)*nterms + j;
					int ix = (jlat)*nterms + i;
					int ipt = (ix)*nlat*nterms + iy;
					irearr[ii] = ipt;
					//printf("irearr[%d]=%d\n", ii, ipt);
					ii++;
				}
			}
		}
	}


	//Kernel evaluated at interpolation nodes. Make it circulant
	double * zkvals = (double *) calloc(nfour*nfour,sizeof(double));
	ii = 0;
	for (int i = 0; i< nfourh; i++){
		for (int j = 0; j< nfourh; j++){
			double tmp = ker(xptsall[0],yptsall[0], xptsall[ii],yptsall[ii], band[i], band[i]);
			//printf("tmp[%d,%d] %f, %f, %f,%f = %f\n", i,j,xptsall[0],yptsall[0], xptsall[ii],yptsall[ii],tmp);
			zkvals[(i+nfourh)*nfour + (j+nfourh)] = tmp;
			zkvals[(nfourh-i)*nfour + (j+nfourh)] = tmp;
			zkvals[(i+nfourh)*nfour + (nfourh -j )] = tmp;
			zkvals[(nfourh-i)*nfour + (nfourh-j)] = tmp;

			ii++;
		}
	}
	for (int i = 20; i< 30; i++){
		for (int j = 30; j< 31; j++){

			//printf("zkvals[%d,%d] = %f\n", i,j,zkvals[i*nfour + j]);
		}
	}

	//FFT of the kernel
	fftw_complex * zkvali = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nfour*nfour);
	fftw_plan p;
	p = fftw_plan_dft_2d(nfour,nfour, zkvali, zkvalf, FFTW_FORWARD, FFTW_ESTIMATE);
	for (int i =0; i< nfour*nfour; i++){
		zkvali[i][0] =  0;
		zkvali[i][1] =  0;
	}

	for (int i = 0; i< nfour*nfour; i++){
		zkvali[i][0] =  zkvals[i];
	}
	fftw_execute(p); 

	/*
	for (int i = 20; i< 30; i++){
		for (int j = 30; j< 31; j++){
			printf("zkvalsf[%d,%d] = %f\n", i,j,zkvalf3[i*nfour + j][0]/(nfour*nfour));
		}
	}
	*/

	fftw_destroy_plan(p);
	fftw_free(zkvali);  

	free(zkvals);
	//The rest of this should be in a separate function...
	
	return 1;
}
int nbodyfft2(int n, int ndim, double* xs, double *ys, double * charges, int nlat, int nterms,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall, double *yptsall,int* irearr, fftw_complex * zkvalf, double * outpot){
	int nboxes = nlat*nlat;
	int nn = nboxes*nterms*nterms;

	double rmin = boxl[0];
	double boxwidth = boxr[0] - boxl[0];

	int * boxcount =(int*) calloc((nboxes +1), sizeof(int));
	int * boxcounti =(int*) calloc(nboxes +1, sizeof(int));
	int * boxsort =(int*) malloc(n* sizeof(int));
	if (boxcount == NULL) {
		printf("Malloc failed\n"); 
		exit(-1);
	}


	//Initialize the charges and the locations
	for (int i=0; i< n; i++){
		int ixs = (xs[i] - rmin)/boxwidth;
		int iys = (ys[i] - rmin)/boxwidth;
		if (ixs > nboxes) {
			ixs = nboxes - 1;
		}
		if (ixs < 0) {
			ixs = 0;
		}
		if (iys > nboxes) {
			iys = nboxes - 1;
		}
		if (iys < 0) {
			iys = 0;
		}

		int icol =  ixs;
		int irow =  iys;
		
		int iadr = (irow)*nlat + icol;

		boxcount[iadr] += 1;
		boxsort[i] = iadr;
		//printf("%d: %f,%f, in box %d, x: %.2f - %.2f,y: %.2f - %.2f which has %d\n", i, xs[i],ys[i],  iadr, boxl[0*nboxes + iadr], boxr[0*nboxes + iadr],boxl[1*nboxes + iadr], boxr[1*nboxes + iadr], boxcount[iadr]);
	}


	int * iarr =(int*) malloc(n* sizeof(int));
	double * chargessort =(double*) malloc(ndim*n* sizeof(double));

	//boxsort[i] = ibox: the box for the ith point
	//Set the offset of each box
	int * boxoffset =(int*) malloc((nboxes+1)* sizeof(int));
	boxoffset[0] = 0;
	for (int ibox = 1; ibox<nboxes +1; ibox++){
		boxoffset[ibox] = boxoffset[ibox-1] + boxcount[ibox-1];
	}

	for (int ibox = 0; ibox<nboxes; ibox++){
		boxcounti[ibox] = 0;
	}

	//The number of points in each box (so far)
	for (int i=0; i<n; i++){
		int iadr =boxsort[i];
		int indx = boxoffset[iadr] + boxcounti[iadr];
		iarr[indx] = i;
		boxcounti[iadr] = boxcounti[iadr] +1;
		//printf("%f, %f , iarr[%d] = %d, iadr = %d, boxoffset[iadr] = %d, boxcounti[iadr] = %d\n", xs[iarr[indx]],ys[iarr[indx]],  indx,i,iadr, boxoffset[iadr], boxcounti[iadr]);
	}

	   for (int i=0; i<n; i++){
	   int ibox = boxsort[iarr[i]];
	   //printf("%d (%d): %f, in box %d, %.2f - %.2f which has %d\n", i,iarr[i], locs[iarr[i]], ibox, boxl[ibox], boxr[ibox],boxcount[ibox]);
	   }

	//Locsort

	//FILE *f = fopen("iarr.txt", "w");
	double * xsort =(double*) malloc(n* sizeof(double));
	double * ysort =(double*) malloc(n* sizeof(double));
	for (int i=0; i<n; i++){
		xsort[i] = xs[iarr[i]];
		ysort[i] = ys[iarr[i]];
		//fprintf(f, "%d,", iarr[i]);
		for (int idim=0; idim<ndim; idim++){
			chargessort[idim*n+i] = charges[idim*n +iarr[i]];
		}
	}

	for (int i=0; i<10; i++){
	//	printf("Charge %d at %f,%f, sorted %f,%f: %f, sorted; %f\n", i, xs[i], ys[i], xsort[i],xsort[i], charges[i], chargessort[0*n+i]);
	}



	//tlocssort is the translated locations
	double * xsp = (double *) malloc(n*sizeof(double));
	double * ysp = (double *) malloc(n*sizeof(double));
	for (int ibox=0; ibox<nboxes;ibox++){
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			double xmin = boxl[ibox];
			double xmax = boxr[ibox];
			xsp[i] = ((xsort[i] - xmin)/(xmax - xmin))*2 - 1;

			double ymin = boxl[nboxes+ ibox];
			double ymax = boxr[nboxes+ ibox];
			ysp[i] = ((ysort[i] - ymin)/(ymax - ymin))*2 - 1;
			//printf("i %d  %f,%f  tlocssort[i] %f,%f  xmin %f xmax %f, ymin %f ymax %f\n", i, xsort[i],ysort[i],  xsp[i],ysp[i], xmin, xmax, ymin, ymax);
		}
	}


	//Get the L_j vals
	

	double * ydiff = (double*) malloc(n*nterms*sizeof(double));
	double * yprods = (double*) malloc(n*nterms*sizeof(double));
	for (int j =0; j < n; j++) {
		yprods[j] = 1;
		for (int i =0; i < nterms; i++) {
			ydiff[j + i*n] = xsp[j] - xpts[i];
			yprods[j] = yprods[j]*ydiff[j+i*n];
		}
	}


	double * svalsx = (double*) malloc(n*nterms*sizeof(double));
	for (int j =0; j < n; j++) {
		for (int i =0; i < nterms; i++) {
			if ( fabs(ydiff[j+i*n]) >= 1e-6) {
				svalsx[j+i*n] = yprods[j]/prods[i]/ydiff[j+i*n];
			}	
			if ( fabs(ydiff[j+i*n]) < 1e-6) {
				svalsx[j+i*n] =  1/prods[i];
				for (int k =1; k < nterms; k++) {
					if(i != k) {
						svalsx[j+i*n] = svalsx[j+i*n] *ydiff[j+k*n];
					}
				}
			}
		}
	}



	

	//L_j for y
	for (int j =0; j < n; j++) {
		yprods[j] = 1;
		for (int i =0; i < nterms; i++) {
			ydiff[j + i*n] = ysp[j] - xpts[i];
			yprods[j] = yprods[j]*ydiff[j+i*n];
		}
	}


	double * svalsy = (double*) malloc(n*nterms*sizeof(double));
	for (int j =0; j < n; j++) {
		for (int i =0; i < nterms; i++) {
			if ( fabs(ydiff[j+i*n]) >= 1e-6) {
				svalsy[j+i*n] = yprods[j]/prods[i]/ydiff[j+i*n];
					//printf("svals[%d] = %lf\n", j+i*n, svals[j+i*n]);
			}	
			if ( fabs(ydiff[j+i*n]) < 1e-6) {
				svalsy[j+i*n] =  1/prods[i];
				for (int k =1; k < nterms; k++) {
					if(i != k) {
						svalsy[j+i*n] = svalsy[j+i*n] *ydiff[j+k*n];
					}
				}
			}
		}
	}
	for (int j =0; j < 5; j++) {
		//printf("svals: %f, %f\n", svalsx[j], svalsy[j]);
	}


	//Compute mpol, which is interpolation
	
	double * mpol = (double *) calloc(sizeof(double), nn*ndim);
	double * loc = (double *) calloc(sizeof(double), nn*ndim);

	for (int ibox=0; ibox<nboxes;ibox++){
		int istart = ibox*nterms*nterms;
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			for (int idim=0;idim<ndim; idim++){
				for (int impx=0; impx<nterms; impx++){
					for (int impy=0; impy<nterms; impy++){
						//printf("ibox: %d i:%d, idim: %d, imp: %d\n", ibox, i, idim, imp);
						int ii = (impx)*nterms + impy;
						mpol[idim + (istart+ii)*ndim] += svalsy[impy*n + i]*svalsx[impx*n + i]*chargessort[idim*n+i];
						//printf("%d %f = %f*%f*%f\n",idim + (istart+ii)*ndim, mpol[idim + (istart+ii)*ndim], svalsx[impy*n + i],svalsy[impx*n + i], chargessort[idim*n+i]);
						//printf("mpol2[%d] = %lf\n", i, mpol[idim + (istart+imp)*ndim]);
					}
				}
			}
		}
	}

	//Mpol to loc
	
	int nfourh = nterms*nlat;
	int nfour = 2*nterms*nlat;

	//fftw_init_threads();
	//fftw_plan_with_nthreads(2);
	//fftw_import_wisdom_from_filename("patientmillionwisdom.txt");
	for (int idim =0; idim< ndim; idim++){
		double * mpolsort = (double *) calloc(sizeof(double), nn);
		for (int i =0; i< nn; i++){
			mpolsort[irearr[i]] = mpol[i*ndim+idim];
		}
		for (int j =0; j < 100; j++) {
			//	printf("mpolsort[%d]=%lf\n",j, mpolsort[j]);
		}

		double * zmpol = (double *) calloc(sizeof(double), nfour*nfour);
		for (int i =0; i< nfourh; i++){
			for (int j =0; j< nfourh; j++){
				int ii =  i*nfourh +j;
				zmpol[i*nfour + j] = mpolsort[ii];
				//printf("zmpol[%d,%d] = %lf\n", i,j , mpolsort[ii]);
			}
		}


		//clock_t begin = clock();
		fftw_plan p,p2;
		//FFT of zmpol
		//printf("doing %d, by %d\n", nfour, nfour);
		fftw_complex * zmpoli = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nfour*nfour);
		fftw_complex * zmpolf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nfour*nfour);
		p = fftw_plan_dft_2d(nfour,nfour, zmpoli, zmpolf, FFTW_FORWARD, FFTW_ESTIMATE);
		p2 = fftw_plan_dft_2d(nfour,nfour, zmpolf, zmpolf, FFTW_FORWARD, FFTW_ESTIMATE);

		for (int i =0; i< nfour*nfour; i++){
			zmpoli[i][0] =  0;
			zmpoli[i][1] =  0;
		}
		for (int i = 0; i< nfour*nfour; i++){
			zmpoli[i][0] =  zmpol[i];
		}

		fftw_execute(p); 
		//fftw_export_wisdom_to_filename("patientmillionwisdom.txt");

//		fftw_destroy_plan(p);
//		fftw_free(zmpoli);
//
		//Take hadamard product

		for (int i =0; i< nfour*nfour; i++){
			//(x_ + y_*i) * (u_ + v_*i) = (x*u - y*v) + (x*v+y*u)i
			double x_ = zmpolf[i][0];
			double y_ = zmpolf[i][1];
			double u_ = zkvalf[i][0];
			double v_ = zkvalf[i][1];
			zmpolf[i][0] = (x_*u_ - y_*v_);
			zmpolf[i][1] = (x_*v_ + y_*u_);
					//printf("(%lf + %lfi) (%lf + %lfi) = %lf + %lfi\n", x_, y_, u_,v_, zmpolf[i][0], zmpolf[i][1]);
		}
		for (int i=0;i<5; i++){
			for (int j=0;j<5; j++){
				//printf("before zmpolf[%d,%d : %lf\n ", i,j, zmpolf[i*nfour+j][0]);
			}
		}

		//Inverse it!
		fftw_execute(p2); 
		for (int i=0;i<nfour; i++){
			for (int j=0;j<nfour; j++){
				//printf("zmpolf[%d,%d : %lf\n ", i,j, zmpolf[i*nfour+j][0]/(double)(nfour*nfour));
				//printf("%0.4e,  ", zmpolf[i*nfour+j][0]/(double)(nfour*nfour));
			}
			//printf("\n%d", i+1);
		//	printf("\n", i+1);
		}
	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Per iteration FFTW took:  %.2e seconds, so %.2e per second\n", time_spent, n/time_spent);

		for (int i=0;i<nfourh; i++){
			for (int j=0;j<nfourh; j++){
				int ii = i*nfourh+j;
				int rowval = (nfourh-i);
				int colval = (nfourh-j);
				mpolsort[ii] = zmpolf[rowval*nfour+colval][0]/(double)(nfour*nfour);
				//printf("%d row: %d col %d is %lf\n", ii, rowval, colval, mpolsort[ii]);
			}
		}
		for (int i =0; i< nfourh*nfourh; i++){
			loc[i*ndim + idim] = mpolsort[irearr[i]];
			//printf("loc[%d]: %lf\n", i, loc[i]);
		}

		fftw_free(zmpoli);  
		fftw_free(zmpolf);  
		fftw_destroy_plan(p);
		fftw_destroy_plan(p2);
		free(zmpol);
		free(mpolsort);
	}
	double * pot = (double *) calloc(n*ndim,sizeof(double));
	for (int ibox=0; ibox<nboxes;ibox++){
		int istart = ibox*nterms*nterms;
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			for (int idim=0;idim<ndim; idim++){
				//outpot[i*ndim +idim] = 0;
				for (int j=0; j<nterms; j++){
					for (int l=0; l<nterms; l++){
						int ii = j*nterms + l;
						pot[i*ndim +idim]  += svalsx[j*n + i]*svalsy[l*n+i]*loc[(istart+ii)*ndim+idim];
					}
				}
			}
		}
	}

	for (int i=0; i<n;i++){
		//printf("pot[%d]= %lf\n", i, pot[i]);
	}

	for (int i=0; i<n;i++){
		for (int j=0; j<ndim;j++){
			outpot[j*n+iarr[i]] = pot[i*ndim+j];
		}
	}
	free(boxcount); free(boxcounti); free(boxsort); free(iarr); free(chargessort);
	free(boxoffset); free(ydiff); free(yprods);free(svalsx);free(svalsy);free(mpol); free(loc);
	free(pot);
	free(xsort); free(ysort); free(xsp); free(ysp);

	return 1;

}
int precompute( double rmin, double rmax,  int nboxes, int nterms, kerneltype ker, double * band,double * boxl, double * boxr,  double * prods, double * xpts, double * xptsall, fftw_complex * zkvalf ) {

	/*
	 * Set up the boxes
	 */
	double boxwidth = (rmax - rmin)/(double) nboxes;

	//Left and right bounds of each box


	int nn = nterms*nboxes;
	if (boxl == NULL) {
		printf("Malloc failed\n"); 
		exit(-1);
	}

	for (int boxi = 0; boxi < nboxes; boxi++){
		boxl[boxi] = boxi*boxwidth + rmin;
		boxr[boxi] = (boxi+1)*(boxwidth) + rmin;
		//printf("box %d, %lf to %lf\n", boxi, boxl[boxi], boxr[boxi]);
	}

	//Coordinates of each (equispaced) interpolation node for a single box
	double h = 2/(double)nterms;
	xpts[0] = -1 + h/2.0;
	for (int i=1; i< nterms; i++){
		xpts[i] = xpts[0] + (i)*h;
	}
	/*
	 * Interpolate kernel using lagrange polynomials
	 */

	//Get the denominators for the lagrange polynomials (this is getprods())
	for (int i= 0; i< nterms; i ++ ){
		prods[i] = 1;
		for (int j=0;j < nterms; j ++ ){
			if (i != j) {
				prods[i] = prods[i] * (xpts[i] - xpts[j]);
			}
		}
//		printf("Prods[%d] xpts[%d] = %lf, %lf\n", i,i, prods[i], xpts[i]);
	}


	//Coordinates of each (equispaced) interpolation node for all boxes
	int ii=0;
	for (int i= 0; i< nboxes; i ++ ){
		for (int j= 0; j< nterms; j ++ ){
			xptsall[ii] = boxl[i] + (xpts[j] +1)/(double)2.0*boxwidth;
			//printf("%d xptsall %lf\n", ii, xptsall[ii]);
			ii++;
		}
	}

	//Kernel evaluated at interpolation nodes. Make it circulant
	double * zkval = (double *) calloc(2*nn,sizeof(double));
	for (int i = 0; i< nn; i++){
		zkval[i] = 0;
		//zkval[i+nn] = 1/(1+pow(xptsall[1] - xptsall[i],2));
		zkval[i+nn] = ker(xptsall[0],xptsall[i], band[i], band[i]);
	}
	zkval[0] = 0;
	for (int i = 1; i< nn; i++){
		zkval[i] =  zkval[2*nn - i];
	}

	//FFT of the kernel
	fftw_complex * zkvali = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*nn);
	for (int i =0; i< 2*nn; i++){
		zkvali[i][0] =  0;
		zkvali[i][1] =  0;
	}

	for (int i = 1; i< 2*nn; i++){
		zkvali[i][0] =  zkval[i];
	}

	fftw_plan p;
	p = fftw_plan_dft_1d(2*nn, zkvali, zkvalf, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p); 

	fftw_destroy_plan(p);
	fftw_free(zkvali); 
	free(zkval);
	//fftw_free(zkvalo);  don't forget to do this

	//The rest of this should be in a separate function...
	
	return 1;
}
int nbodyfft(int n, int ndim, double* locs, double * charges, int nboxes, int nterms,double *boxl, double *boxr,  double * prods, double * xpts, double * xptsall, fftw_complex * zkvalf, double * outpot){

	int nn = nterms*nboxes;
	double rmin = boxl[0];
	double boxwidth = boxr[0] - boxl[0];

	int * boxoffset =(int*) malloc((nboxes+1)* sizeof(int));
	int * boxcount =(int*) calloc((nboxes +1), sizeof(int));
	int * boxcounti =(int*) calloc(nboxes +1, sizeof(int));
	if (boxcount == NULL) {
		printf("Malloc failed\n"); 
		exit(-1);
	}


	int * boxsort =(int*) malloc(n* sizeof(int));
	int * iarr =(int*) malloc(n* sizeof(int));
	double * chargessort =(double*) malloc(ndim*n* sizeof(double));

	//Initialize the charges and the locations
	for (int i=0; i< n; i++){
		int ibox = (locs[i] - rmin)/boxwidth;
		if (ibox > nboxes) {
			ibox = nboxes - 1;
		}
		if (ibox < 0) {
			ibox = 0;
		}
		boxsort[i] = ibox;
		boxcount[ibox] = boxcount[ibox] + 1;
		//printf("%d: %f, in box %d, %.2f - %.2f which has %d\n", i, locs[i], ibox, boxl[ibox], boxr[ibox],boxcount[ibox]);
	}


	//boxsort[i] = ibox: the box for the ith point
	//Set the offset of each box
	boxoffset[0] = 0;
	for (int ibox = 1; ibox<nboxes +1; ibox++){
		boxoffset[ibox] = boxoffset[ibox-1] + boxcount[ibox-1];
	}

	for (int ibox = 0; ibox<nboxes; ibox++){
		boxcounti[ibox] = 0;
	}

	//The number of points in each box (so far)
	for (int i=0; i<n; i++){
		int ibox =boxsort[i];
		int indx = boxoffset[ibox] + boxcounti[ibox];
		iarr[indx] = i;
		boxcounti[ibox] = boxcounti[ibox] +1;
		//	printf("%f, iarr[%d] = %d, ibox = %d, boxoffset[ibox] = %d, boxcounti[ibox] = %d\n", locs[iarr[indx]], indx,i,ibox, boxoffset[ibox], boxcounti[ibox]);
	}

	/*
	   for (int i=0; i<n; i++){
	   int ibox = boxsort[iarr[i]];
	   printf("%d (%d): %f, in box %d, %.2f - %.2f which has %d\n", i,iarr[i], locs[iarr[i]], ibox, boxl[ibox], boxr[ibox],boxcount[ibox]);
	   }
	   */

	//Locsort

	//FILE *f = fopen("iarr.txt", "w");
	double * xs =(double*) malloc(n* sizeof(double));
	for (int i=0; i<n; i++){
		xs[i] = locs[iarr[i]];
		//fprintf(f, "%d,", iarr[i]);
		for (int idim=0; idim<ndim; idim++){
			chargessort[idim*n+i] = charges[idim*n +iarr[i]];
		}
	}

	for (int i=0; i<10; i++){
		//printf("Charge %d at %f, sorted %f: %f, sorted; %f\n", i, locs[i], xs[i], charges[i], chargessort[0*n+i]);
	}



	//tlocssort is the translated locations
	double * xsp = (double *) malloc(n*sizeof(double));
	for (int ibox=0; ibox<nboxes;ibox++){
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			double xmin = boxl[ibox];
			double xmax = boxr[ibox];
			xsp[i] = ((xs[i] - xmin)/(xmax - xmin))*2 - 1;
	//		printf("i %d locs[i] %f tlocssort[i] %f xmin %f xmax %f\n", i, xs[i], xsp[i], xmin, xmax);
		}
	}


	//Get the L_j vals
	
	double * ydiff = (double*) malloc(n*nterms*sizeof(double));
	double * yprods = (double*) malloc(n*nterms*sizeof(double));
	for (int j =0; j < n; j++) {
		yprods[j] = 1;
		for (int i =0; i < nterms; i++) {
			ydiff[j + i*n] = xsp[j] - xpts[i];
			yprods[j] = yprods[j]*ydiff[j+i*n];
		}
	}


	double * svals = (double*) malloc(n*nterms*sizeof(double));
	for (int j =0; j < n; j++) {
		for (int i =0; i < nterms; i++) {
			if ( fabs(ydiff[j+i*n]) >= 1e-6) {
				svals[j+i*n] = yprods[j]/prods[i]/ydiff[j+i*n];
					//printf("svals[%d] = %lf\n", j+i*n, svals[j+i*n]);
			}	
			if ( fabs(ydiff[j+i*n]) < 1e-6) {
				svals[j+i*n] =  1/prods[i];
				for (int k =1; k < nterms; k++) {
					if(i != k) {
						svals[j+i*n] = svals[j+i*n] *ydiff[j+k*n];
					}
				}
			}
		}
	}


	//Compute mpol, which is interpolation
	
	double * mpol = (double *) calloc(sizeof(double), nn*ndim);
	double * loc = (double *) calloc(sizeof(double), nn*ndim);

	for (int ibox=0; ibox<nboxes;ibox++){
		int istart = ibox*nterms;
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			for (int idim=0;idim<ndim; idim++){
				for (int imp=0; imp<nterms; imp++){
					//printf("ibox: %d i:%d, idim: %d, imp: %d\n", ibox, i, idim, imp);
					mpol[idim + (istart+imp)*ndim] += svals[imp*n + i]*chargessort[idim*n+i];
					//printf("mpol2[%d] = %lf\n", i, mpol[idim + (istart+imp)*ndim]);
				}
			}
		}
	}

	//Mpol to loc
	double * zmpol = (double *) calloc(sizeof(double), 2*nn*ndim);
	for (int i =0; i< nn; i++){
		for (int idim =0; idim< ndim; idim++){
			zmpol[(i+nn)*ndim + idim] = mpol[(i)*ndim + idim];
			//printf("zmpol[%d] = %lf\n", (i)*ndim + idim, mpol[(i)*ndim + idim]);
		}
	}


	//clock_t begin = clock();
	fftw_plan p;
	//FFT of zmpol
	for (int idim =0; idim< ndim; idim++){
		fftw_complex * zmpoli = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*nn);
		fftw_complex * zmpolf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*nn);
		for (int i =0; i< 2*nn; i++){
			zmpoli[i][0] =  0;
			zmpoli[i][1] =  0;
		}
		for (int i = 0; i< 2*nn; i++){
			zmpoli[i][0] =  zmpol[i*ndim + idim];
			//printf("zmpoli[i][0] =%lf\n", zmpoli[i][0]);
		}

		p = fftw_plan_dft_1d(2*nn, zmpoli, zmpolf, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(p); 

		fftw_destroy_plan(p);
		fftw_free(zmpoli);

		//Take hadamard product
		
		for (int i =0; i< 2*nn; i++){
			//(x_ + y_*i) * (u_ + v_*i) = (x*u - y*v) + (x*v+y*u)i
			double x_ = zmpolf[i][0];
			double y_ = zmpolf[i][1];
			double u_ = zkvalf[i][0];
			double v_ = zkvalf[i][1];
			zmpolf[i][0] = (x_*u_ - y_*v_);
			zmpolf[i][1] = (x_*v_ + y_*u_);
	//		printf("(%lf + %lfi) (%lf + %lfi) = %lf + %lfi\n", x_, y_, u_,v_, zmpolf[i][0], zmpolf[i][1]);
		}

		//Inverse it!
		p = fftw_plan_dft_1d(2*nn, zmpolf, zmpolf, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(p); 
		fftw_destroy_plan(p);
		for (int i=0;i<nn; i++){
			loc[i*ndim+idim] = zmpolf[i][0]/(double)(nn*2.0);
		}
		fftw_free(zmpolf);
	}
	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Per iteration FFTW took:  %.2e seconds, so %.2e per second\n", time_spent, n/time_spent);

	double * pot = (double *) calloc(n*ndim,sizeof(double));
	for (int ibox=0; ibox<nboxes;ibox++){
		int istart = ibox*nterms;
		for (int i=boxoffset[ibox]; i<boxoffset[ibox+1];i++){
			for (int idim=0;idim<ndim; idim++){
				outpot[i*ndim +idim] = 0;
				for (int j=0; j<nterms; j++){
					pot[i*ndim +idim]  += svals[j*n + i]*loc[(istart+j)*ndim+idim];
				}
			}
		}
	}


	for (int i=0; i<n;i++){
		for (int j=0; j<ndim;j++){
			outpot[j*n+iarr[i]] = pot[i*ndim+j];
		}
	}
	free(boxoffset); free(boxcount); free(boxcounti);
	free(boxsort); free(iarr); free(chargessort);
	free(xs); free(xsp);free(ydiff); free(yprods);
	free(svals);free(mpol); free(loc); free(zmpol); free(pot);
	return 1;

}

