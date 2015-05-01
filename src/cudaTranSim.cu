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

//#include <cutil_inline.h>
#include <helper_cuda.h>

#include <stdio.h>

#include "etbr.h"
#include "gpuData.h"
#include "kernels.h"

void myMemcpyD2S2(float *dst, double *src, int n)
{
  for(int i=0; i<n; i++)
    dst[i] = (float)src[i];
}

//16384

extern "C" void cudaTranSim(gpuETBR *myGPUetbr)
{
  cudaEvent_t start, stop;  cudaEventCreate(&start);  cudaEventCreate(&stop);  float time;
  cudaEventRecord(start, 0);
    
  printf("     cudaTranSim start.\n");

  int deviceCount, dev;
  cudaGetDeviceCount(&deviceCount);
  cudaDeviceProp deviceProp;
  dev=0;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaSetDevice(dev);
  printf("   Device %d: \"%s\" has been selected.\n", dev,deviceProp.name);

  cublasStatus_t cublas_status=cublasInit();
  if(cublas_status != CUBLAS_STATUS_SUCCESS)  printf("CUBLAS failed to initialize.\n");

  int numPts=myGPUetbr->numPts, q=myGPUetbr->q, m=myGPUetbr->m,// n=myGPUetbr->n,
    nIS=myGPUetbr->nIS, nVS=myGPUetbr->nVS,
    nport=myGPUetbr->nport, partLen=0, shift=0, i;
  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ipiv_dev), q*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->L_hCG_dev), q*q*sizeof(double)) );
  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->U_hCG_dev), q*q*sizeof(double)) );
  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->hC_dev), q*q*sizeof(double)) );
  double *tmpDqVecDev, *tmpDqXrDev;
  checkCudaErrors( cudaMalloc((void**)&(tmpDqVecDev), q*sizeof(double)) );
  checkCudaErrors( cudaMalloc((void**)&(tmpDqXrDev), q*sizeof(double)) );
  // float *tmpSqVecDev;
  // checkCudaErrors( cudaMalloc((void**)&(tmpSqVecDev), q*sizeof(float)) );
  
  if(myGPUetbr->use_cuda_double) {
    if(m*(myGPUetbr->ldUt)*sizeof(double) < 400000000)
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_dev), m*(myGPUetbr->ldUt)*sizeof(double)) );
    else {
      partLen = PART_LEN;//1024; //
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_dev), m*partLen*sizeof(double)) );
    }
    //checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->V_dev), n*q*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->LV_dev), nport*q*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->Br_dev), q*m*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->xr_dev), q*numPts*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_dev), nport*numPts*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->dcVt_dev), nVS*sizeof(double)) );
  }
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    if(m*(myGPUetbr->ldUt)*sizeof(float) < 400000000)
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_single_dev), m*(myGPUetbr->ldUt)*sizeof(float)) );
    else {
      partLen = PART_LEN;
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_single_dev), m*partLen*sizeof(float)) );
    }

    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->LV_single_dev), nport*q*sizeof(float)) );
    // checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->L_hCG_single_dev), q*q*sizeof(float)) );
    // checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->U_hCG_single_dev), q*q*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->hC_single_dev), q*q*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->Br_single_dev), q*m*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->xr_single_dev), q*numPts*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_single_dev), nport*numPts*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->dcVt_single_dev), nVS*sizeof(float)) );    
  }

  cudaMemcpy(myGPUetbr->ipiv_dev, myGPUetbr->ipiv_host, q*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(myGPUetbr->L_hCG_dev, myGPUetbr->L_hCG_host, q*q*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(myGPUetbr->U_hCG_dev, myGPUetbr->U_hCG_host, q*q*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(myGPUetbr->hC_dev, myGPUetbr->hC_host, q*q*sizeof(double), cudaMemcpyHostToDevice);
  if(myGPUetbr->use_cuda_double) {
    //cudaMemcpy(myGPUetbr->V_dev, myGPUetbr->V_host, n*q*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->LV_dev, myGPUetbr->LV_host, nport*q*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->Br_dev, myGPUetbr->Br_host, q*m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->xr_dev, myGPUetbr->xr0_host, q*sizeof(double), cudaMemcpyHostToDevice); // only ic is copied
    cudaMemcpy(myGPUetbr->dcVt_dev, myGPUetbr->dcVt_host, nVS*sizeof(double), cudaMemcpyHostToDevice);
  }
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    cudaMemcpy(myGPUetbr->LV_single_dev, myGPUetbr->LV_single_host, nport*q*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(myGPUetbr->L_hCG_single_dev, myGPUetbr->L_hCG_single_host, q*q*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(myGPUetbr->U_hCG_single_dev, myGPUetbr->U_hCG_single_host, q*q*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->hC_single_dev, myGPUetbr->hC_single_host, q*q*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->Br_single_dev, myGPUetbr->Br_single_host, q*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(myGPUetbr->xr_single_dev, myGPUetbr->xr0_single_host, q*sizeof(float), cudaMemcpyHostToDevice); // only ic is copied
    cudaMemcpy(myGPUetbr->dcVt_single_dev, myGPUetbr->dcVt_single_host, nVS*sizeof(float), cudaMemcpyHostToDevice);
  }
  // for(int j=0; j<q; j++)  printf("  x[%d]=%6.4e\n",j,myGPUetbr->xr0_host[j]);

  /* The following section need CPU generated source info. */
  /* 
  cudaMemcpy(myGPUetbr->ut_dev, myGPUetbr->ut_host, m*(numPts-1)*sizeof(double), cudaMemcpyHostToDevice);
  cublasDgemm('N', 'N', q, numPts-1, m, 1.0, myGPUetbr->Br_dev, q, myGPUetbr->ut_dev, m,
	      0.0, myGPUetbr->xr_dev+q, q); // B*u for all time steps
  */
  /********* CPU generated source info transfered. *********/

  /*******************************************************************/
  /* The following section use parallel GPU to generate source info. */
  dim3 genUtGrd(nIS, ((numPts)+BLK_SIZE_UTGEN-1)/BLK_SIZE_UTGEN);//-1
  dim3 genUtVdcGrd(nVS, ((numPts)+BLK_SIZE_UTGEN-1)/BLK_SIZE_UTGEN);//-1
  if(partLen) {
    genUtGrd.y = ((partLen)+BLK_SIZE_UTGEN-1)/BLK_SIZE_UTGEN;
    genUtVdcGrd.y = ((partLen)+BLK_SIZE_UTGEN-1)/BLK_SIZE_UTGEN;
  }
  dim3 genUtBlk(BLK_SIZE_UTGEN);
  if(partLen==0) { // *****************************
    if(myGPUetbr->use_cuda_double) {
      gen_dcVt_kernel_wrapper//<<<genUtVdcGrd, genUtBlk>>>
	(myGPUetbr->ut_dev, myGPUetbr->dcVt_dev, numPts, myGPUetbr->ldUt,//-1
         genUtVdcGrd, genUtBlk);
    }
    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      gen_dcVt_single_kernel_wrapper//<<<genUtVdcGrd, genUtBlk>>>
	(myGPUetbr->ut_single_dev, myGPUetbr->dcVt_single_dev, numPts, myGPUetbr->ldUt, //-1
         genUtVdcGrd, genUtBlk);
    }

    if(myGPUetbr->PWLcurExist) {
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLnumPts_dev), nIS*sizeof(int)) );
      cudaMemcpy(myGPUetbr->PWLnumPts_dev, myGPUetbr->PWLnumPts_host, nIS*sizeof(int),cudaMemcpyHostToDevice);
      if(myGPUetbr->use_cuda_double) {
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	cudaMemcpy(myGPUetbr->PWLtime_dev, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS*sizeof(double),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(myGPUetbr->PWLval_dev, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS*sizeof(double),
		   cudaMemcpyHostToDevice);
	gen_PWLut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PWLtime_dev, myGPUetbr->PWLval_dev,
           myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
           genUtGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	cudaMemcpy(myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLtime_single_host,
		   nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(myGPUetbr->PWLval_single_dev, myGPUetbr->PWLval_single_host,
		   nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice);
	gen_PWLut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLval_single_dev,
           myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt, //-1
           genUtGrd, genUtBlk);
      }
    }

    if(myGPUetbr->PULSEcurExist) {
      if(myGPUetbr->use_cuda_double) {
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_dev), nIS*5*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_dev), nIS*2*sizeof(double)) );
	cudaMemcpy(myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEtime_host, nIS*5*sizeof(double),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(myGPUetbr->PULSEval_dev, myGPUetbr->PULSEval_host, nIS*2*sizeof(double),
		   cudaMemcpyHostToDevice);
	gen_PULSEut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEval_dev,
           myGPUetbr->tstep, numPts, myGPUetbr->ldUt, //-1
           genUtGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_single_dev), nIS*5*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_single_dev), nIS*2*sizeof(double)) );
	myGPUetbr->PULSEtime_single_host = (float*)malloc( nIS*5*sizeof(float));
	myGPUetbr->PULSEval_single_host = (float*)malloc( nIS*2*sizeof(float));
	myMemcpyD2S2(myGPUetbr->PULSEtime_single_host, myGPUetbr->PULSEtime_host, nIS*5);
	myMemcpyD2S2(myGPUetbr->PULSEval_single_host, myGPUetbr->PULSEval_host, nIS*2);
	cudaMemcpy(myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEtime_single_host,
		   nIS*5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(myGPUetbr->PULSEval_single_dev, myGPUetbr->PULSEval_single_host,
		   nIS*2*sizeof(float), cudaMemcpyHostToDevice);
	gen_PULSEut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
	  (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
	   myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEval_single_dev,
	   myGPUetbr->tstep, numPts, myGPUetbr->ldUt, //-1
           genUtGrd, genUtBlk);
      }
    }

    if(myGPUetbr->use_cuda_double) {
      cublasDgemm('N', 'T', q, numPts, m, 1.0, myGPUetbr->Br_dev, q,//-1
		  myGPUetbr->ut_dev, myGPUetbr->ldUt,
		  0.0, myGPUetbr->xr_dev, q); // B*u for all time steps +q
    }
    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      cublasSgemm('N', 'T', q, numPts, m, 1.0, myGPUetbr->Br_single_dev, q,//-1
		  myGPUetbr->ut_single_dev, myGPUetbr->ldUt,
		  0.0, myGPUetbr->xr_single_dev, q); // B*u for all time steps +q
    }
  }
  else { // partLen > 0 ********************************************
    printf("      Due to the large number of sources and transient steps,\n");
    printf("      the parallel loading process needs to be work on\n");
    printf("      separate time segments in order to save GPU memory.\n");
    for(shift=0; shift<numPts; shift+=partLen) {//-1
      
      genUtGrd.y = partLen/BLK_SIZE_UTGEN;
      genUtVdcGrd.y = partLen/BLK_SIZE_UTGEN;
      
      
      if(myGPUetbr->use_cuda_double) {
	gen_dcVt_part_kernel_wrapper//<<<genUtVdcGrd, genUtBlk>>>
	  (myGPUetbr->ut_dev,
	   myGPUetbr->dcVt_dev, numPts, partLen, shift,//-1
           genUtVdcGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	gen_dcVt_part_single_kernel_wrapper//<<<genUtVdcGrd, genUtBlk>>>
	  (myGPUetbr->ut_single_dev,
	   myGPUetbr->dcVt_single_dev, numPts, partLen, shift,//-1
           genUtVdcGrd, genUtBlk);
      }

      if(myGPUetbr->PWLcurExist) {
	printf("      Under Construction: part by part evaluation of PWL sources.\n"); while(!getchar()) ;
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLnumPts_dev), nIS*sizeof(int)) );
	cudaMemcpy(myGPUetbr->PWLnumPts_dev, myGPUetbr->PWLnumPts_host, nIS*sizeof(int),cudaMemcpyHostToDevice);
	if(myGPUetbr->use_cuda_double) {
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	  cudaMemcpy(myGPUetbr->PWLtime_dev, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS*sizeof(double),
		     cudaMemcpyHostToDevice);
	  cudaMemcpy(myGPUetbr->PWLval_dev, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS*sizeof(double),
		     cudaMemcpyHostToDevice);
	  gen_PWLut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
            (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
             myGPUetbr->PWLtime_dev, myGPUetbr->PWLval_dev,
             myGPUetbr->PWLnumPts_dev,
             myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
             genUtGrd, genUtBlk);
	}
	if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	  cudaMemcpy(myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLtime_single_host,
		     nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice);
	  cudaMemcpy(myGPUetbr->PWLval_single_dev, myGPUetbr->PWLval_single_host,
		     nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice);
	  gen_PWLut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
            (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
             myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLval_single_dev,
             myGPUetbr->PWLnumPts_dev,
             myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
             genUtGrd, genUtBlk);
	}
      }
      if(myGPUetbr->PULSEcurExist) {
	if(myGPUetbr->use_cuda_double) {
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_dev), nIS*5*sizeof(double)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_dev), nIS*2*sizeof(double)) );
	  cudaMemcpy(myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEtime_host, nIS*5*sizeof(double),
		     cudaMemcpyHostToDevice);
	  cudaMemcpy(myGPUetbr->PULSEval_dev, myGPUetbr->PULSEval_host, nIS*2*sizeof(double),
		     cudaMemcpyHostToDevice);
	  gen_PULSEut_part_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
	    (myGPUetbr->ut_dev + myGPUetbr->nVS*partLen,
	     myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEval_dev,
	     myGPUetbr->tstep, numPts, partLen, shift,//-1
             genUtGrd, genUtBlk);
	}
	if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_single_dev), nIS*5*sizeof(double)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_single_dev), nIS*2*sizeof(double)) );
	  myGPUetbr->PULSEtime_single_host = (float*)malloc( nIS*5*sizeof(float));
	  myGPUetbr->PULSEval_single_host = (float*)malloc( nIS*2*sizeof(float));
	  myMemcpyD2S2(myGPUetbr->PULSEtime_single_host, myGPUetbr->PULSEtime_host, nIS*5);
	  myMemcpyD2S2(myGPUetbr->PULSEval_single_host, myGPUetbr->PULSEval_host, nIS*2);
	  cudaMemcpy(myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEtime_single_host,
		     nIS*5*sizeof(float), cudaMemcpyHostToDevice);
	  cudaMemcpy(myGPUetbr->PULSEval_single_dev, myGPUetbr->PULSEval_single_host,
		     nIS*2*sizeof(float), cudaMemcpyHostToDevice);
	  gen_PULSEut_part_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
	    (myGPUetbr->ut_single_dev + myGPUetbr->nVS*partLen,
	     myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEval_single_dev,
	     myGPUetbr->tstep, numPts, partLen, shift,//-1
             genUtGrd, genUtBlk);
	}
      }

      if(shift+partLen <= numPts-1) {
	if(myGPUetbr->use_cuda_double) {
	  cublasDgemm('N', 'T', q, partLen, m, 1.0, myGPUetbr->Br_dev, q,
		      myGPUetbr->ut_dev, partLen,
		      0.0, myGPUetbr->xr_dev+shift*q, q); // B*u for all time steps +q
	}
	if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	  cublasSgemm('N', 'T', q, partLen, m, 1.0, myGPUetbr->Br_single_dev, q,
		      myGPUetbr->ut_single_dev, partLen,
		      0.0, myGPUetbr->xr_single_dev+shift*q, q); // B*u for all time steps +q
	}
      }
      else {
	if(myGPUetbr->use_cuda_double) {
	  cublasDgemm('N', 'T', q, numPts-shift, m, 1.0, myGPUetbr->Br_dev, q,//-1
		      myGPUetbr->ut_dev, partLen,
		      0.0, myGPUetbr->xr_dev+shift*q, q); // B*u for all time steps +q
	}
	if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	  cublasSgemm('N', 'T', q, numPts-shift, m, 1.0, myGPUetbr->Br_single_dev, q,//-1
		      myGPUetbr->ut_single_dev, partLen,
		      0.0, myGPUetbr->xr_single_dev+shift*q, q); // B*u for all time steps +q
	}
      }
    }
  }

  if(myGPUetbr->use_cuda_double) {
    cudaMemcpy(myGPUetbr->xr_dev, myGPUetbr->xr0_host, q*sizeof(double), cudaMemcpyHostToDevice); // only ic is copied
  }
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    cudaMemcpy(myGPUetbr->xr_single_dev, myGPUetbr->xr0_single_host, q*sizeof(float), cudaMemcpyHostToDevice); // only ic is copied
  }

  /*************** Parallel source generation finished. ****************/
  
  if(myGPUetbr->use_cuda_double) {
    for(i=1; i<numPts; i++) {
      cublasDgemv('N', q, q, 1.0, myGPUetbr->hC_dev, q, myGPUetbr->xr_dev+(i-1)*q, 1,
		  1.0, myGPUetbr->xr_dev+i*q, 1); // 1/h*C*x + B*u
      permute_kernel_wrapper//<<<1,q>>>
        (myGPUetbr->xr_dev+i*q, myGPUetbr->ipiv_dev, q); // pivoting
      cublasDtrsv('L', 'N', 'U', q, myGPUetbr->L_hCG_dev, q,
		  myGPUetbr->xr_dev+i*q, 1); // L*y = b
      cublasDtrsv('U', 'N', 'N', q, myGPUetbr->U_hCG_dev, q,
		  myGPUetbr->xr_dev+i*q, 1); // U*x = y
    }
    cublasDgemm('N', 'N', nport, numPts, q, 1.0, myGPUetbr->LV_dev, nport, myGPUetbr->xr_dev, q,
		0.0, myGPUetbr->x_dev, nport);
    cudaMemcpy(myGPUetbr->x_host, myGPUetbr->x_dev, nport*numPts*sizeof(double), cudaMemcpyDeviceToHost);
  }
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    myMemcpyS2Ddev_wrapper//<<<1,q>>>
      (tmpDqXrDev, myGPUetbr->xr_single_dev, q);
    for(i=1; i<numPts; i++) {
      myMemcpyS2Ddev_wrapper//<<<1,q>>>
        (tmpDqVecDev, myGPUetbr->xr_single_dev+i*q, q);
      
      cublasDgemv('N', q, q, 1.0, myGPUetbr->hC_dev, q, tmpDqXrDev, 1,
		  1.0, tmpDqVecDev, 1); // 1/h*C*x + B*u
      permute_kernel_wrapper//<<<1,q>>>
        (tmpDqVecDev, myGPUetbr->ipiv_dev, q); // pivoting
      cublasDtrsv('L', 'N', 'U', q, myGPUetbr->L_hCG_dev, q,
		  tmpDqVecDev, 1); // L*y = b
      cublasDtrsv('U', 'N', 'N', q, myGPUetbr->U_hCG_dev, q,
		  tmpDqVecDev, 1); // U*x = y
      cudaMemcpy(tmpDqXrDev, tmpDqVecDev, q*sizeof(double), cudaMemcpyDeviceToDevice);
      myMemcpyD2Sdev_wrapper//<<<1,q>>>
        (myGPUetbr->xr_single_dev+i*q, tmpDqVecDev, q);
      /*
      cublasSgemv('N', q, q, 1.0, myGPUetbr->hC_single_dev, q, myGPUetbr->xr_single_dev+(i-1)*q, 1,
		  1.0, myGPUetbr->xr_single_dev+i*q, 1); // 1/h*C*x + B*u
      permute_single_kernel<<<1,q>>>(myGPUetbr->xr_single_dev+i*q,
				     myGPUetbr->ipiv_dev, q); // pivoting
      cublasStrsv('L', 'N', 'U', q, myGPUetbr->L_hCG_dev, q,
		  myGPUetbr->xr_single_dev+i*q, 1); // L*y = b
      cublasStrsv('U', 'N', 'N', q, myGPUetbr->U_hCG_dev, q,
		  myGPUetbr->xr_single_dev+i*q, 1); // U*x = y
      */
    }
    cublasSgemm('N', 'N', nport, numPts, q, 1.0, myGPUetbr->LV_single_dev, nport,
                myGPUetbr->xr_single_dev, q,
		0.0, myGPUetbr->x_single_dev, nport);
    cudaMemcpy(myGPUetbr->x_single_host, myGPUetbr->x_single_dev, nport*numPts*sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("                GPU parallel Time: %6.4e\n", 1e-3*time); // convert from millisecond to second
  cudaEventDestroy( start ); cudaEventDestroy( stop );
}


extern "C" void gpuRelatedDataInit(gpuETBR *myGPUetbr)
{
  myGPUetbr->PWLnumPts_host=NULL; myGPUetbr->PWLnumPts_dev=NULL;
  myGPUetbr->PWLtime_host=NULL; myGPUetbr->PWLtime_dev=NULL;// maximum length is MAX_PWL_PTS
  myGPUetbr->PWLval_host=NULL; myGPUetbr->PWLval_dev=NULL; // maximum length is MAX_PWL_PTS

  myGPUetbr->PWLtime_single_host=NULL; myGPUetbr->PWLtime_single_dev=NULL;// maximum length is MAX_PWL_PTS
  myGPUetbr->PWLval_single_host=NULL; myGPUetbr->PWLval_single_dev=NULL; // maximum length is MAX_PWL_PTS

  myGPUetbr->PULSEtime_host=NULL; myGPUetbr->PULSEtime_dev=NULL;
  myGPUetbr->PULSEval_host=NULL; myGPUetbr->PULSEval_dev=NULL;

  myGPUetbr->PULSEtime_single_host=NULL; myGPUetbr->PULSEtime_single_dev=NULL;
  myGPUetbr->PULSEval_single_host=NULL; myGPUetbr->PULSEval_single_dev=NULL;
}

extern "C" void gpuRelatedDataFree(gpuETBR *myGPUetbr)
{
  printf("     Free memory in myGPUetbr\n");
  //int numPts=myGPUetbr->numPts, n=myGPUetbr->n, q=myGPUetbr->q, m=myGPUetbr->m;

  if(myGPUetbr->PWLnumPts_host != NULL)
    free(myGPUetbr->PWLnumPts_host);
    
  if(myGPUetbr->PWLnumPts_dev != NULL)
    cudaFree(myGPUetbr->PWLnumPts_dev);

  if(myGPUetbr->PWLtime_host != NULL)
    free(myGPUetbr->PWLtime_host);

  if(myGPUetbr->PWLtime_dev != NULL)
    cudaFree(myGPUetbr->PWLtime_dev);// maximum length is MAX_PWL_PTS

  if(myGPUetbr->PWLval_host != NULL)
    free(myGPUetbr->PWLval_host);
  
  if(myGPUetbr->PWLval_dev != NULL)
    cudaFree(myGPUetbr->PWLval_dev); // maximum length is MAX_PWL_PTS

  if(myGPUetbr->PWLtime_single_host != NULL)
    free(myGPUetbr->PWLtime_single_host);

  if(myGPUetbr->PWLtime_single_dev != NULL)
    cudaFree(myGPUetbr->PWLtime_single_dev);// maximum length is MAX_PWL_PTS

  if(myGPUetbr->PWLval_single_host != NULL)
    free(myGPUetbr->PWLval_single_host);

  if(myGPUetbr->PWLval_single_dev != NULL)
    cudaFree(myGPUetbr->PWLval_single_dev); // maximum length is MAX_PWL_PTS

  if(myGPUetbr->PULSEtime_host != NULL)
    free(myGPUetbr->PULSEtime_host);

  if(myGPUetbr->PULSEtime_dev != NULL)
    cudaFree(myGPUetbr->PULSEtime_dev);

  if(myGPUetbr->PULSEval_host != NULL)
    free(myGPUetbr->PULSEval_host);

  if(myGPUetbr->PULSEval_dev != NULL)
    cudaFree(myGPUetbr->PULSEval_dev);

  if(myGPUetbr->PULSEtime_single_host != NULL)
    free(myGPUetbr->PULSEtime_single_host);

  if(myGPUetbr->PULSEtime_single_dev != NULL)
    cudaFree(myGPUetbr->PULSEtime_single_dev);

  if(myGPUetbr->PULSEval_single_host != NULL)
    free(myGPUetbr->PULSEval_single_host);
  
  if(myGPUetbr->PULSEval_single_dev != NULL)
    cudaFree(myGPUetbr->PULSEval_single_dev);

  //cublasShutdown();
  //cutilDeviceReset();
}

extern "C" void cudaTranSim_shutdown(gpuETBR *myGPUetbr)
{
  printf("     cudaTranSim shutdown\n");
  //int numPts=myGPUetbr->numPts, n=myGPUetbr->n, q=myGPUetbr->q, m=myGPUetbr->m;

  //cudaFree(myGPUetbr->V_dev);
  cudaFree(myGPUetbr->L_hCG_dev);
  cudaFree(myGPUetbr->U_hCG_dev);
  cudaFree(myGPUetbr->hC_dev);
  cudaFree(myGPUetbr->Br_dev);
  cudaFree(myGPUetbr->xr_dev);
  cudaFree(myGPUetbr->x_dev);

  cublasShutdown();
  cudaDeviceReset();
  //cutilDeviceReset();
}


  // printf("       genUtBlk(%d,%d)\n", genUtBlk.x, genUtBlk.y);
  // printf("       genUtGrd(%d,%d)\n", genUtGrd.x, genUtGrd.y);
  // printf("       genUtVdcGrd(%d,%d)\n", genUtVdcGrd.x, genUtVdcGrd.y);
  // printf("       numPts-1=%d\n",numPts-1);

    /*
    FILE *fp;  
    char filenameH[] = "testSaveUtHostETBR.bin";
    int numPts_1=numPts-1;
    fp = fopen(filenameH, "wb");
      fwrite(&m, sizeof(int), 1, fp);
      fwrite(&numPts_1, sizeof(int), 1, fp);
      fwrite(myGPUetbr->ut_host, sizeof(double), numPts_1*m, fp);
    fclose(fp);
    printf("        >>> >>> Binary data file saved in testSaveUtHostETBR.bin\n");

    cudaMemcpy(myGPUetbr->ut_host, myGPUetbr->ut_dev, (myGPUetbr->ldUt)*m*sizeof(double), cudaMemcpyDeviceToHost);
    char filenameD[] = "testSaveUtDevETBR.bin";
    fp = fopen(filenameD, "wb");
      fwrite(&(myGPUetbr->ldUt), sizeof(int), 1, fp);
      fwrite(&m, sizeof(int), 1, fp);
      fwrite(myGPUetbr->ut_host, sizeof(double), (myGPUetbr->ldUt)*m, fp);
    fclose(fp);
    printf("        >>> >>> Binary data file saved in testSaveUtDevETBR.bin\n");
    while(!getchar()) ;
    */
