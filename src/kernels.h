#ifndef _KERNELS_H_
#define _KERNELS_H_

void myMemcpyD2Sdev_wrapper(float *vecS, double *vecD, int q);
void myMemcpyS2Ddev_wrapper(double *vecD, float *vecS, int q);

void permute_kernel_wrapper(double *x, int *ipiv, int q);
void permute_single_kernel_wrapper(float *x, int *ipiv, int q);

void gen_dcVt_kernel_wrapper(double *ut, double *dcVt, int p, int ldUt,
                             dim3 grid, dim3 block);
void gen_dcVt_single_kernel_wrapper(float *ut, float *dcVt, int p, int ldUt,
                                    dim3 grid, dim3 block);

void gen_dcVt_part_kernel_wrapper(double *ut, double *dcVt, int p, int ldUt, int shift,
                                  dim3 grid, dim3 block);
void gen_dcVt_part_single_kernel_wrapper(float *ut, float *dcVt, int p, int ldUt, int shift,
                                         dim3 grid, dim3 block);

void
gen_PWLut_kernel_wrapper(double *ut, double *PWLtime, double *PWLval, int *PWLnumPts,
			 double tstep, int p, int ldUt,
			 dim3 grid, dim3 block);
void
gen_PWLut_single_kernel_wrapper(float *ut, float *PWLtime, float *PWLval, int *PWLnumPts,
                                double tstep, int p, int ldUt,
                                dim3 grid, dim3 block);

void
gen_PULSEut_kernel_wrapper(double *ut, double *PULSEtime, double *PULSEval,
			   double tstep, int p, int ldUt,
			   dim3 grid, dim3 block);
void
gen_PULSEut_single_kernel_wrapper(float *ut, float *PULSEtime, float *PULSEval,
                                  double tstep, int p, int ldUt,
                                  dim3 grid, dim3 block);

void
gen_PULSEut_part_kernel_wrapper(double *ut, double *PULSEtime, double *PULSEval,
                                double tstep, int p, int ldUt, int shift,
                                dim3 grid, dim3 block);

void
gen_PULSEut_part_single_kernel_wrapper(float *ut, float *PULSEtime, float *PULSEval,
                                       double tstep, int p, int ldUt, int shift,
                                       dim3 grid, dim3 block);
#endif
