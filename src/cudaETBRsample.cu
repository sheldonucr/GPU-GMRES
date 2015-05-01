/*
 * This file contains the GPU based source loading and LU triangular solve
 * for transient simulation of ETBR system.
 *
 * There are some CUDA GPU kernel functions to carry out
 * the permutation job for pivoted LU factors,
 * and the parallel source interpolation on all time steps.
 *
 * Author: Xue-Xin Liu
 *         2011-Nov-16
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>

#include <cutil_inline.h>

#include <stdio.h>

#include "etbr.h"
#include "gpuData.h"
#include "kernels.h"

void cudaETBRinit(double *ut_dev, int nBat, int ldUt)
{
  cudaMalloc((void**)&ut_dev, nBat*ldUt*sizeof(double));
}

void cudaETBRclear(double *ut_dev)
{
  cudaFree(ut_dev);
}

