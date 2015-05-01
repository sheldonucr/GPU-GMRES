#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
//#include <cutil_inline.h>
//#include <cublas_v2.h>
//#include <cusparse_v2.h>

//#include "etbr.h"

#include "gmres.h"
#include "defs.h"
#include "SpMV.h"
#include "gpuData.h"
#include "kernels.h"


void wrapperGPUforPG(ucr_cs_dl *left,
                     ucr_cs_dl *right, ucr_cs_dl *G, ucr_cs_dl *B,
                     int *invPort, int nport,
                     //double *rhsDC,
                     ucr_cs_dl *LG, ucr_cs_dl *UG, int *pinvG, int *qinvG,
                     ucr_cs_dl *LA, ucr_cs_dl *UA, int *pinvA, int *qinvA,
                     gpuETBR *myGPUetbr)
{
  struct timeval st, et;
  float fmtTime=0, srcEvalTime=0, beTime=0;
  
  int deviceCount, dev=0;
  cudaGetDeviceCount(&deviceCount);
  printf("CUDA device count: %d\n",deviceCount);
  if (deviceCount == 0) { 
    printf("No device supporting CUDA\n"); 
    exit(-1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaSetDevice(dev);
  printf("   Device %d: \"%s\" has been selected.\n", dev,deviceProp.name);
  
  //cublasStatus_t cublas_status=cublasInit();
  //if(cublas_status != CUBLAS_STATUS_SUCCESS)  printf("CUBLAS failed to initialize.\n");
  // cublasStatus_t statCublas;
  // cublasHandle_t handleCublas;
  // statCublas = cublasCreate(&handleCublas);
  // if( stat != CUBLAS_STATUS_SUCCESS ) {
  //   printf ( "CUBLAS i n i t i a l i z a t i o n f a i l e d \n" ) ;
  //   exit(-1) ;
  // }
  cusparseStatus_t statusCusparse;
  cusparseHandle_t handleCusparse=0;
  cusparseMatDescr_t descrB=0;
  statusCusparse = cusparseCreate(&handleCusparse);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS) {
    printf( "CUSPARSE Library initialization failed\n");
    exit(-1);
  }
  float sOne=1.0, sZero=0.0;
  /* create and setup matrix descriptor */
  statusCusparse= cusparseCreateMatDescr(&descrB);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS) {
    printf( "Matrix descriptor initialization failed\n");
    exit(-1);
  }
  cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);

  /* XXLiu: Convert from CSC to CSR. */
  if(left->nz == -1)
    printf("Converting CSC format (cs_dl) to CSR format.\n");
  else
    printf("ERROR: cs_dl contains triplet format and is not supported.\n");

  gettimeofday(&st, NULL);
  MySpMatrix GmySpM, BmySpM, leftMySpM, rightMySpM,//, left1MySpM, CmySpM
    LportMySpM,
    LGmySpM, UGmySpM, PGmySpM, QGmySpM,
    LAmySpM, UAmySpM, PAmySpM, QAmySpM;

  LDcsc2cscMySpMatrix( &GmySpM, G);
  LDcsc2cscMySpMatrix( &BmySpM, B);
  LDcsc2cscMySpMatrix( &leftMySpM, left);
  /*
  LDcsc2csrMySpMatrix( &left1MySpM, left);
  addUnitMatrix( &left1MySpM );
  */
  LDcsc2cscMySpMatrix( &rightMySpM, right);
  /*
  writeCSRmySpMatrix( &GmySpM, "csrFileG.dat");
  writeCSRmySpMatrix( &leftMySpM, "csrFileA.dat");
  writeCSRmySpMatrix( &rightMySpM, "csrFilehC.dat");
  writeCSRmySpMatrix( &BmySpM, "csrFileB.dat");
  */
  LDcsc2cscMySpMatrix( &LGmySpM, LG);
  LDcsc2cscMySpMatrix( &UGmySpM, UG);
  vec2csrMySpMatrix( &PGmySpM, pinvG, G->m);
  vec2csrMySpMatrix( &QGmySpM, qinvG, G->m);
  LDcsc2cscMySpMatrix( &LAmySpM, LA);
  LDcsc2cscMySpMatrix( &UAmySpM, UA);
  vec2csrMySpMatrix( &PAmySpM, pinvA, G->m);
  vec2csrMySpMatrix( &QAmySpM, qinvA, G->m);
  printf("Completed the converting of CSC format (cs_dl) to CSR format.\n");

  LportMySpM.isCSR = 1;
  LportMySpM.numRows = nport;
  LportMySpM.numCols = G->m;
  LportMySpM.numNZEntries = nport;
  LportMySpM.rowIndices = (int*)malloc( (nport + 1)*sizeof(int));
  LportMySpM.indices = (int*)malloc( nport*sizeof(int));
  LportMySpM.val = (float*)malloc( nport*sizeof(float));
  for(int i=0; i<nport; i++) {
    LportMySpM.rowIndices[i] = i;
    LportMySpM.indices[i] = invPort[i];
    LportMySpM.val[i] = 1.0;
  }
  LportMySpM.rowIndices[ LportMySpM.numRows ] =  nport;

  SpMatrixGPU Gsparse, Asparse, Bsparse, LportSparse,
    hCsparse, LGsparse, UGsparse, PGsparse, QGsparse,
    LAsparse, UAsparse, PAsparse, QAsparse;
  gpuMallocCpyCSRmySpM(&Gsparse, &GmySpM);
  gpuMallocCpyCSRmySpM(&Asparse, &leftMySpM);
  gpuMallocCpyCSRmySpM(&Bsparse, &BmySpM);
  gpuMallocCpyCSRmySpM(&hCsparse, &rightMySpM);
  gpuMallocCpyCSRmySpM(&LportSparse, &LportMySpM);
  
  gpuMallocCpyCSRmySpM(&LGsparse, &LGmySpM);
  gpuMallocCpyCSRmySpM(&UGsparse, &UGmySpM);
  gpuMallocCpyCSRmySpM(&PGsparse, &PGmySpM);
  gpuMallocCpyCSRmySpM(&QGsparse, &QGmySpM);

  gpuMallocCpyCSRmySpM(&LAsparse, &LAmySpM);
  gpuMallocCpyCSRmySpM(&UAsparse, &UAmySpM);
  gpuMallocCpyCSRmySpM(&PAsparse, &PAmySpM);
  gpuMallocCpyCSRmySpM(&QAsparse, &QAmySpM);

  gettimeofday(&et, NULL);
  fmtTime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
  printf("   Format conversion time: %.2f (ms)\n", fmtTime);
  
  
  /*
  int k=0;
  for(int i=0; i<UG->nzmax; i++)
    if(UG->x[i] == 0)
      k = k+1;
  printf("   k: %d\n",k);
  */
  /*
  writeCSRmySpMatrix( &LGmySpM, "csrFileLG.dat");
  writeCSRmySpMatrix( &UGmySpM, "csrFileUG.dat");
  writeCSRmySpMatrix( &PGmySpM, "csrFilePG.dat");
  writeCSRmySpMatrix( &QGmySpM, "csrFileQG.dat"); //---
  writeCSRmySpMatrix( &LAmySpM, "csrFileLA.dat");
  writeCSRmySpMatrix( &UAmySpM, "csrFileUA.dat");
  writeCSRmySpMatrix( &PAmySpM, "csrFilePA.dat");
  writeCSRmySpMatrix( &QAmySpM, "csrFileQA.dat"); //---
  */
  /*
  MySpMatrixDouble LGmySpMdouble, UGmySpMdouble, LAmySpMdouble, UAmySpMdouble;
  LDcsc2csrMySpMatrixDouble( &LGmySpMdouble, LG);
  LDcsc2csrMySpMatrixDouble( &UGmySpMdouble, UG);
  LDcsc2csrMySpMatrixDouble( &LAmySpMdouble, LA);
  LDcsc2csrMySpMatrixDouble( &UAmySpMdouble, UA);
  */
  /*
  writeCSRmySpMatrixDouble( &LGmySpMdouble, "csrFileLGd.dat");
  writeCSRmySpMatrixDouble( &UGmySpMdouble, "csrFileUGd.dat"); //---
  writeCSRmySpMatrixDouble( &LAmySpMdouble, "csrFileLAd.dat");
  writeCSRmySpMatrixDouble( &UAmySpMdouble, "csrFileUAd.dat");
  */

  //int *G_csrRowPtr, *G_csrColIdx;
  //float *G_csrVal;
  //LDcsc2csr( (int)G->nzmax, (int)G->m, (int)G->n,
  //           G->p, G->i, G->x,
  //           &G_csrRowPtr, &G_csrColIdx, &G_csrVal);
  //
  //int *left_csrRowPtr, *left_csrColIdx;
  //float *left_csrVal;
  //LDcsc2csr( (int)left->nzmax, (int)left->m, (int)left->n,
  //          left->p, left->i, left->x,
  //          &left_csrRowPtr, &left_csrColIdx, &left_csrVal);
  //
  //int *right_csrRowPtr, *right_csrColIdx;
  //float *right_csrVal;
  //LDcsc2csr((int)right->nzmax, (int)right->m, (int)right->n,
  //          right->p, right->i, right->x,
  //          &right_csrRowPtr, &right_csrColIdx, &right_csrVal);
  //
  //int *B_csrRowPtr, *B_csrColIdx;
  //float *B_csrVal;
  //LDcsc2csr((int)B->nzmax, (int)B->m, (int)B->n,
  //          B->p, B->i, B->x,
  //          &B_csrRowPtr, &B_csrColIdx, &B_csrVal);
  //------------------------------------
  // writeCSR( (int)left->m, (int)left->n, (int)left->nzmax,
  //           left_csrRowPtr, left_csrColIdx, left_csrVal,
  //           "csrFileG.dat");
  // writeCSR( (int)left->m, (int)left->n, (int)left->nzmax,
  //           left_csrRowPtr, left_csrColIdx, left_csrVal,
  //           "csrFileA.dat");
  // writeCSR( (int)right->m, (int)right->n, (int)right->nzmax,
  //           right_csrRowPtr, right_csrColIdx, right_csrVal,
  //           "csrFilehC.dat");
  // writeCSR( (int)B->m, (int)B->n, (int)B->nzmax,
  //           B_csrRowPtr, B_csrColIdx, B_csrVal,
  //           "csrFileB.dat");

  //-------------------------------------------
  gettimeofday(&st, NULL);

  int numPts=myGPUetbr->numPts,m=myGPUetbr->m, n=myGPUetbr->n, //q=myGPUetbr->q, 
    nIS=myGPUetbr->nIS, nVS=myGPUetbr->nVS, // nport=myGPUetbr->nport, 
    partLen=0, shift=0;//, i;
  
  cusparseSolveAnalysisInfo_t LG_info, UG_info, LA_info, UA_info;
  cusparseMatDescr_t L_des, U_des;
  assert(cusparseCreateMatDescr(&L_des) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateMatDescr(&U_des) == CUSPARSE_STATUS_SUCCESS);

  cusparseSetMatType(L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatFillMode(L_des, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(L_des, CUSPARSE_DIAG_TYPE_UNIT);
  cusparseSetMatIndexBase(L_des, CUSPARSE_INDEX_BASE_ZERO);

  cusparseSetMatType(U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatFillMode(U_des, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(U_des, CUSPARSE_INDEX_BASE_ZERO);

  assert(cusparseCreateSolveAnalysisInfo(&LG_info) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateSolveAnalysisInfo(&UG_info) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateSolveAnalysisInfo(&LA_info) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateSolveAnalysisInfo(&UA_info) == CUSPARSE_STATUS_SUCCESS);

  statusCusparse = cusparseScsrsv_analysis
    (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
     n, LGmySpM.numNZEntries, L_des,
     LGmySpM.d_val, LGmySpM.d_rowIndices, LGmySpM.d_indices, LG_info);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
    printf("    ERROR: csrsv_analysis for LG --- %d.\n", statusCusparse);
  assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  statusCusparse = cusparseScsrsv_analysis
    (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
     n, UGmySpM.numNZEntries, U_des,
     UGmySpM.d_val, UGmySpM.d_rowIndices, UGmySpM.d_indices, UG_info);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
    printf("    ERROR: csrsv_analysis for UG --- %d.\n", statusCusparse);
  assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  
  statusCusparse = cusparseScsrsv_analysis
    (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
     n, LAmySpM.numNZEntries, L_des,
     LAmySpM.d_val, LAmySpM.d_rowIndices, LAmySpM.d_indices, LA_info);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
    printf("    ERROR: csrsv_analysis for LA --- %d.\n", statusCusparse);
  assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  statusCusparse = cusparseScsrsv_analysis
    (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
     n, UAmySpM.numNZEntries, U_des,
     UAmySpM.d_val, UAmySpM.d_rowIndices, UAmySpM.d_indices, UA_info);
  if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
    printf("    ERROR: csrsv_analysis for UA --- %d.\n", statusCusparse);
  assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);

  double *utTd_dev;
  if(myGPUetbr->use_cuda_double) {
    if(m*(myGPUetbr->ldUt)*sizeof(double) < 400000000) {
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_dev), m*(myGPUetbr->ldUt)*sizeof(double)) );
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_dev), n*numPts*sizeof(double)) );
      checkCudaErrors( cudaMalloc((void**)&(utTd_dev), m*(myGPUetbr->ldUt)*sizeof(double)) );
    }
    else {
      partLen = PART_LEN;//1024; //
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_dev), m*partLen*sizeof(double)) );
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_dev), n*partLen*sizeof(double)) );
      checkCudaErrors( cudaMalloc((void**)&(utTd_dev), m*partLen*sizeof(double)) );
    }
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->dcVt_dev), nVS*sizeof(double)) );
    checkCudaErrors( cudaMemcpy(myGPUetbr->dcVt_dev, myGPUetbr->dcVt_host,
                              nVS*sizeof(double), cudaMemcpyHostToDevice) );
  }
  float *utTs_dev, *xtmpS_dev, *xtmpluS_dev, *xportS_dev, *xOldS_dev, *xNewS_dev;
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    checkCudaErrors( cudaMalloc((void**)&(xtmpS_dev), n*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(xtmpluS_dev), n*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(utTs_dev), m*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(xOldS_dev), n*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&(xNewS_dev), n*sizeof(float)) );
    checkCudaErrors( cudaMalloc( (void**)&xportS_dev, nport*numPts*sizeof(float)) );
    if(m*(myGPUetbr->ldUt)*sizeof(float) < 400000000) {
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_single_dev), m*(myGPUetbr->ldUt)*sizeof(float)) );
      //checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_single_dev), n*numPts*sizeof(float)) );
      //checkCudaErrors( cudaMalloc((void**)&(utTs_dev), m*(myGPUetbr->ldUt)*sizeof(float)) );
    }
    else {
      partLen = PART_LEN;
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->ut_single_dev), m*partLen*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->x_single_dev), n*partLen*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&(utTs_dev), m*partLen*sizeof(float)) );
    }
    checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->dcVt_single_dev), nVS*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(myGPUetbr->dcVt_single_dev, myGPUetbr->dcVt_single_host,
                              nVS*sizeof(float), cudaMemcpyHostToDevice) );
  }
  //-------------------------------------------

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
	(myGPUetbr->ut_single_dev, myGPUetbr->dcVt_single_dev, numPts, myGPUetbr->ldUt,//-1
         genUtVdcGrd, genUtBlk);
    }

    if(myGPUetbr->PWLcurExist) {
      checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLnumPts_dev), nIS*sizeof(int)) );
      checkCudaErrors( cudaMemcpy(myGPUetbr->PWLnumPts_dev, myGPUetbr->PWLnumPts_host,
                                nIS*sizeof(int),cudaMemcpyHostToDevice) );
      if(myGPUetbr->use_cuda_double) {
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PWLtime_dev, myGPUetbr->PWLtime_host,
                                  nIS*MAX_PWL_PTS*sizeof(double), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PWLval_dev, myGPUetbr->PWLval_host,
                                  nIS*MAX_PWL_PTS*sizeof(double), cudaMemcpyHostToDevice) );
	gen_PWLut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PWLtime_dev, myGPUetbr->PWLval_dev,
           myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
           genUtVdcGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLtime_single_host,
                                  nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PWLval_single_dev, myGPUetbr->PWLval_single_host,
                                  nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice) );
	gen_PWLut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLval_single_dev,
           myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
           genUtGrd, genUtBlk);
      }
    }

    if(myGPUetbr->PULSEcurExist) {
      if(myGPUetbr->use_cuda_double) {
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_dev), nIS*5*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_dev), nIS*2*sizeof(double)) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEtime_host,
                                  nIS*5*sizeof(double), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEval_dev, myGPUetbr->PULSEval_host,
                                  nIS*2*sizeof(double), cudaMemcpyHostToDevice) );
	gen_PULSEut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
          (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
           myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEval_dev,
           myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
           genUtGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_single_dev), nIS*5*sizeof(double)) );
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_single_dev), nIS*2*sizeof(double)) );
	myGPUetbr->PULSEtime_single_host = (float*)malloc( nIS*5*sizeof(float));
	myGPUetbr->PULSEval_single_host = (float*)malloc( nIS*2*sizeof(float));
	myMemcpyD2S(myGPUetbr->PULSEtime_single_host, myGPUetbr->PULSEtime_host, nIS*5);
	myMemcpyD2S(myGPUetbr->PULSEval_single_host, myGPUetbr->PULSEval_host, nIS*2);
	checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEtime_single_host,
                                  nIS*5*sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEval_single_dev, myGPUetbr->PULSEval_single_host,
                                  nIS*2*sizeof(float), cudaMemcpyHostToDevice) );
	gen_PULSEut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
	  (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
	   myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEval_single_dev,
	   myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
           genUtGrd, genUtBlk);
      }
    }
    gettimeofday(&et, NULL);
    srcEvalTime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
    printf("    Src eval time : %.2f (ms)\n", srcEvalTime);

    gettimeofday(&st, NULL);
    if(myGPUetbr->use_cuda_double) {
      exit(-1);
      // cusparseDcsrmm(handle, 'T', q, numPts, m, 1.0, myGPUetbr->Br_dev, q,
      //   	  myGPUetbr->ut_dev, myGPUetbr->ldUt,
      //   	  0.0, myGPUetbr->xr_dev+q, q); // B*u for all time steps
    }
    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      // for(int i=0; i<numPts; i++) {
      //   cublasScopy(m, myGPUetbr->ut_single_dev+i, myGPUetbr->ldUt,
      //               utTs_dev+i*m, 1);
      // }
      // cusparseScsrmm(handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
      //                m, numPts, n, BmySpM.numNZEntries,
      //                &sOne, descrB, BmySpM.d_val, BmySpM.d_rowIndices, BmySpM.d_indices,
      //                utTs_dev, m, &sZero, myGPUetbr->x_single_dev, n);
      for(int i=0; i<numPts; i++) {
        cublasScopy(m, myGPUetbr->ut_single_dev+i, myGPUetbr->ldUt,
                    utTs_dev, 1);
        cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
                       m, n, BmySpM.numNZEntries,
                       &sOne, descrB, BmySpM.d_val, BmySpM.d_rowIndices, BmySpM.d_indices,
                       utTs_dev, &sZero, xNewS_dev);
        if(i==0) {
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         n, n, n,
                         &sOne, descrB, PGmySpM.d_val, PGmySpM.d_rowIndices, PGmySpM.d_indices,
                         xNewS_dev, &sZero, xtmpS_dev);
          statusCusparse = cusparseScsrsv_solve
            (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
             n, &sOne, L_des, LGmySpM.d_val, LGmySpM.d_rowIndices, LGmySpM.d_indices,
             LG_info, xtmpS_dev, xtmpluS_dev);
          statusCusparse = cusparseScsrsv_solve
            (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
             n, &sOne, U_des, UGmySpM.d_val, UGmySpM.d_rowIndices, UGmySpM.d_indices,
             UG_info, xtmpluS_dev, xtmpS_dev);
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         n, n, n,
                         &sOne, descrB, QGmySpM.d_val, QGmySpM.d_rowIndices, QGmySpM.d_indices,
                         xtmpS_dev, &sZero, xOldS_dev);
        }
        else {
          // checkCudaErrors( cudaMemcpy(xtmpS_dev, myGPUetbr->x_single_dev+(i-1)*n,
          //                           n*sizeof(float), cudaMemcpyDeviceToDevice) );
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
                         n, n, rightMySpM.numNZEntries,
                         &sOne, descrB, rightMySpM.d_val, rightMySpM.d_rowIndices, rightMySpM.d_indices,
                         xOldS_dev, &sOne, xNewS_dev);
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         n, n, n,
                         &sOne, descrB, PAmySpM.d_val, PAmySpM.d_rowIndices, PAmySpM.d_indices,
                         xNewS_dev, &sZero, xtmpS_dev);
          statusCusparse = cusparseScsrsv_solve
            (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
             n, &sOne, L_des, LAmySpM.d_val, LAmySpM.d_rowIndices, LAmySpM.d_indices,
             LA_info, xtmpS_dev, xtmpluS_dev);
          statusCusparse = cusparseScsrsv_solve
            (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
             n, &sOne, U_des, UAmySpM.d_val, UAmySpM.d_rowIndices, UAmySpM.d_indices,
             UA_info, xtmpluS_dev, xtmpS_dev);
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         n, n, n,
                         &sOne, descrB, QAmySpM.d_val, QAmySpM.d_rowIndices, QAmySpM.d_indices,
                         xtmpS_dev, &sZero, xOldS_dev);
        }
        cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       nport, n, nport,
                       &sOne, descrB, LportMySpM.d_val, LportMySpM.d_rowIndices, LportMySpM.d_indices,
                       xOldS_dev, &sZero, xportS_dev+i*nport);
      }
      checkCudaErrors( cudaMemcpy(myGPUetbr->x_single_host, xportS_dev,
                                nport*numPts*sizeof(float), cudaMemcpyDeviceToHost) );
    }
    gettimeofday(&et, NULL);
    beTime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
    printf("    BE time: %.2f (ms)\n", beTime);
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
	  (myGPUetbr->ut_dev, myGPUetbr->dcVt_dev, numPts, partLen, shift,//-1
           genUtVdcGrd, genUtBlk);
      }
      if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	gen_dcVt_part_single_kernel_wrapper//<<<genUtVdcGrd, genUtBlk>>>
	  (myGPUetbr->ut_single_dev, myGPUetbr->dcVt_single_dev, numPts, partLen, shift,//-1
           genUtVdcGrd, genUtBlk);
      }

      if(myGPUetbr->PWLcurExist) {
	printf("      Under Construction: part by part evaluation of PWL sources.\n"); while(!getchar()) ;
	checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLnumPts_dev), nIS*sizeof(int)) );
	checkCudaErrors( cudaMemcpy(myGPUetbr->PWLnumPts_dev, myGPUetbr->PWLnumPts_host,
                                  nIS*sizeof(int),cudaMemcpyHostToDevice) );
	if(myGPUetbr->use_cuda_double) {
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_dev), nIS*MAX_PWL_PTS*sizeof(double)) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PWLtime_dev, myGPUetbr->PWLtime_host,
                                    nIS*MAX_PWL_PTS*sizeof(double), cudaMemcpyHostToDevice) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PWLval_dev, myGPUetbr->PWLval_host,
                                    nIS*MAX_PWL_PTS*sizeof(double), cudaMemcpyHostToDevice) );
	  gen_PWLut_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
            (myGPUetbr->ut_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
             myGPUetbr->PWLtime_dev, myGPUetbr->PWLval_dev,
             myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
             genUtGrd, genUtBlk);
	}
	if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLtime_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PWLval_single_dev), nIS*MAX_PWL_PTS*sizeof(float)) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLtime_single_host,
                                    nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PWLval_single_dev, myGPUetbr->PWLval_single_host,
                                    nIS*MAX_PWL_PTS*sizeof(float), cudaMemcpyHostToDevice) );
	  gen_PWLut_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
            (myGPUetbr->ut_single_dev + myGPUetbr->nVS*myGPUetbr->ldUt,
             myGPUetbr->PWLtime_single_dev, myGPUetbr->PWLval_single_dev,
             myGPUetbr->PWLnumPts_dev, myGPUetbr->tstep, numPts, myGPUetbr->ldUt,//-1
             genUtGrd, genUtBlk);
	}
      }
      if(myGPUetbr->PULSEcurExist) {
	if(myGPUetbr->use_cuda_double) {
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEtime_dev), nIS*5*sizeof(double)) );
	  checkCudaErrors( cudaMalloc((void**)&(myGPUetbr->PULSEval_dev), nIS*2*sizeof(double)) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEtime_dev, myGPUetbr->PULSEtime_host,
                                    nIS*5*sizeof(double), cudaMemcpyHostToDevice) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEval_dev, myGPUetbr->PULSEval_host,
                                    nIS*2*sizeof(double), cudaMemcpyHostToDevice) );
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
	  myMemcpyD2S(myGPUetbr->PULSEtime_single_host, myGPUetbr->PULSEtime_host, nIS*5);
	  myMemcpyD2S(myGPUetbr->PULSEval_single_host, myGPUetbr->PULSEval_host, nIS*2);
	  checkCudaErrors(cudaMemcpy(myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEtime_single_host,
                                   nIS*5*sizeof(float), cudaMemcpyHostToDevice) );
	  checkCudaErrors( cudaMemcpy(myGPUetbr->PULSEval_single_dev, myGPUetbr->PULSEval_single_host,
                                    nIS*2*sizeof(float), cudaMemcpyHostToDevice) );
	  gen_PULSEut_part_single_kernel_wrapper//<<<genUtGrd, genUtBlk>>>
	    (myGPUetbr->ut_single_dev + myGPUetbr->nVS*partLen,
	     myGPUetbr->PULSEtime_single_dev, myGPUetbr->PULSEval_single_dev,
	     myGPUetbr->tstep, numPts, partLen, shift,//-1
             genUtGrd, genUtBlk);
	}
      }

      // if(shift+partLen <= numPts-1) {
      //   if(myGPUetbr->use_cuda_double) {
      //     cublasDgemm('N', 'T', q, partLen, m, 1.0, myGPUetbr->Br_dev, q,
      //   	      myGPUetbr->ut_dev, partLen,
      //   	      0.0, myGPUetbr->xr_dev+q+shift*q, q); // B*u for all time steps
      //   }
      //   if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      //     cublasSgemm('N', 'T', q, partLen, m, 1.0, myGPUetbr->Br_single_dev, q,
      //   	      myGPUetbr->ut_single_dev, partLen,
      //   	      0.0, myGPUetbr->xr_single_dev+q+shift*q, q); // B*u for all time steps
      //   }
      // }
      // else {
      //   if(myGPUetbr->use_cuda_double) {
      //     cublasDgemm('N', 'T', q, numPts-1-shift, m, 1.0, myGPUetbr->Br_dev, q,
      //   	      myGPUetbr->ut_dev, partLen,
      //   	      0.0, myGPUetbr->xr_dev+q+shift*q, q); // B*u for all time steps
      //   }
      //   if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      //     cublasSgemm('N', 'T', q, numPts-1-shift, m, 1.0, myGPUetbr->Br_single_dev, q,
      //   	      myGPUetbr->ut_single_dev, partLen,
      //   	      0.0, myGPUetbr->xr_single_dev+q+shift*q, q); // B*u for all time steps
      //   }
      // }
    }
  }
  /*************** Parallel source generation finished. ****************/


  /*
  float *ut_host;
  if(myGPUetbr->use_cuda_single) {
    if(partLen==0) {
      ut_host=(float*)malloc(m*(myGPUetbr->ldUt)*sizeof(float));
      checkCudaErrors( cudaMemcpy(ut_host, myGPUetbr->ut_single_dev,
                                m*(myGPUetbr->ldUt)*sizeof(float), cudaMemcpyDeviceToHost) );
      // writeUt( ut_host, m, myGPUetbr->ldUt, "utFile.dat");
    }
    else {
      ut_host=(float*)malloc(m*partLen*sizeof(float));
      checkCudaErrors( cudaMemcpy(ut_host, myGPUetbr->ut_single_dev,
                                m*partLen*sizeof(float), cudaMemcpyDeviceToHost) );
      writeUt( ut_host, m, partLen, "utFile.dat");
    }
  }
  */
  //------------------------------------

  // free(G_csrRowPtr);  free(G_csrColIdx);  free(G_csrVal);
  // free(left_csrRowPtr);  free(left_csrColIdx);  free(left_csrVal);
  // free(right_csrRowPtr);  free(right_csrColIdx);  free(right_csrVal);
  // free(B_csrRowPtr);  free(B_csrColIdx);  free(B_csrVal);
  // free(xGMRES);
  // free(rhsBu);
  //exit(0);
}
