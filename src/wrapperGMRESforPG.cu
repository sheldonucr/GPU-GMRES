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

#include "gmres_interface_pg.h"
#include "preconditioner.h"

void wrapperGMRESforPG(ucr_cs_dl *left,
                       ucr_cs_dl *right, ucr_cs_dl *G, ucr_cs_dl *B,
                       int *invPort, int nport,
                       //double *rhsDC,
                       //ucr_cs_dl *LG, ucr_cs_dl *UG, int *pinvG, int *qinvG,
                       //ucr_cs_dl *LA, ucr_cs_dl *UA, int *pinvA, int *qinvA,
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
    LportMySpM;
    // LGmySpM, UGmySpM, PGmySpM, QGmySpM,
    // LAmySpM, UAmySpM, PAmySpM, QAmySpM;

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
  // LDcsc2cscMySpMatrix( &LGmySpM, LG);
  // LDcsc2cscMySpMatrix( &UGmySpM, UG);
  // vec2csrMySpMatrix( &PGmySpM, pinvG, G->m);
  // vec2csrMySpMatrix( &QGmySpM, qinvG, G->m);
  // LDcsc2cscMySpMatrix( &LAmySpM, LA);
  // LDcsc2cscMySpMatrix( &UAmySpM, UA);
  // vec2csrMySpMatrix( &PAmySpM, pinvA, G->m);
  // vec2csrMySpMatrix( &QAmySpM, qinvA, G->m);
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
    hCsparse;//, LGsparse, UGsparse, PGsparse, QGsparse,
  //LAsparse, UAsparse, PAsparse, QAsparse;
  gpuMallocCpyCSRmySpM(&Gsparse, &GmySpM);
  gpuMallocCpyCSRmySpM(&Asparse, &leftMySpM);
  gpuMallocCpyCSRmySpM(&Bsparse, &BmySpM);
  gpuMallocCpyCSRmySpM(&hCsparse, &rightMySpM);
  gpuMallocCpyCSRmySpM(&LportSparse, &LportMySpM);
  
  // gpuMallocCpyCSRmySpM(&LGsparse, &LGmySpM);
  // gpuMallocCpyCSRmySpM(&UGsparse, &UGmySpM);
  // gpuMallocCpyCSRmySpM(&PGsparse, &PGmySpM);
  // gpuMallocCpyCSRmySpM(&QGsparse, &QGmySpM);
  // 
  // gpuMallocCpyCSRmySpM(&LAsparse, &LAmySpM);
  // gpuMallocCpyCSRmySpM(&UAsparse, &UAmySpM);
  // gpuMallocCpyCSRmySpM(&PAsparse, &PAmySpM);
  // gpuMallocCpyCSRmySpM(&QAsparse, &QAmySpM);

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

  // statusCusparse = cusparseScsrsv_analysis
  //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
  //    n, LGmySpM.numNZEntries, L_des,
  //    LGmySpM.d_val, LGmySpM.d_rowIndices, LGmySpM.d_indices, LG_info);
  // if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
  //   printf("    ERROR: csrsv_analysis for LG --- %d.\n", statusCusparse);
  // assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  // statusCusparse = cusparseScsrsv_analysis
  //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
  //    n, UGmySpM.numNZEntries, U_des,
  //    UGmySpM.d_val, UGmySpM.d_rowIndices, UGmySpM.d_indices, UG_info);
  // if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
  //   printf("    ERROR: csrsv_analysis for UG --- %d.\n", statusCusparse);
  // assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  // 
  // statusCusparse = cusparseScsrsv_analysis
  //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
  //    n, LAmySpM.numNZEntries, L_des,
  //    LAmySpM.d_val, LAmySpM.d_rowIndices, LAmySpM.d_indices, LA_info);
  // if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
  //   printf("    ERROR: csrsv_analysis for LA --- %d.\n", statusCusparse);
  // assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);
  // statusCusparse = cusparseScsrsv_analysis
  //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
  //    n, UAmySpM.numNZEntries, U_des,
  //    UAmySpM.d_val, UAmySpM.d_rowIndices, UAmySpM.d_indices, UA_info);
  // if(statusCusparse != CUSPARSE_STATUS_SUCCESS)
  //   printf("    ERROR: csrsv_analysis for UA --- %d.\n", statusCusparse);
  // assert(statusCusparse == CUSPARSE_STATUS_SUCCESS);

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
          // cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          //                n, n, n,
          //                &sOne, descrB, PGmySpM.d_val, PGmySpM.d_rowIndices, PGmySpM.d_indices,
          //                xNewS_dev, &sZero, xtmpS_dev);
          // statusCusparse = cusparseScsrsv_solve
          //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
          //    n, &sOne, L_des, LGmySpM.d_val, LGmySpM.d_rowIndices, LGmySpM.d_indices,
          //    LG_info, xtmpS_dev, xtmpluS_dev);
          // statusCusparse = cusparseScsrsv_solve
          //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
          //    n, &sOne, U_des, UGmySpM.d_val, UGmySpM.d_rowIndices, UGmySpM.d_indices,
          //    UG_info, xtmpluS_dev, xtmpS_dev);
          // cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          //                n, n, n,
          //                &sOne, descrB, QGmySpM.d_val, QGmySpM.d_rowIndices, QGmySpM.d_indices,
          //                xtmpS_dev, &sZero, xOldS_dev);
        }
        else {
          // checkCudaErrors( cudaMemcpy(xtmpS_dev, myGPUetbr->x_single_dev+(i-1)*n,
          //                           n*sizeof(float), cudaMemcpyDeviceToDevice) );
          cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
                         n, n, rightMySpM.numNZEntries,
                         &sOne, descrB, rightMySpM.d_val, rightMySpM.d_rowIndices, rightMySpM.d_indices,
                         xOldS_dev, &sOne, xNewS_dev);
          // cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          //                n, n, n,
          //                &sOne, descrB, PAmySpM.d_val, PAmySpM.d_rowIndices, PAmySpM.d_indices,
          //                xNewS_dev, &sZero, xtmpS_dev);
          // statusCusparse = cusparseScsrsv_solve
          //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
          //    n, &sOne, L_des, LAmySpM.d_val, LAmySpM.d_rowIndices, LAmySpM.d_indices,
          //    LA_info, xtmpS_dev, xtmpluS_dev);
          // statusCusparse = cusparseScsrsv_solve
          //   (handleCusparse, CUSPARSE_OPERATION_TRANSPOSE,
          //    n, &sOne, U_des, UAmySpM.d_val, UAmySpM.d_rowIndices, UAmySpM.d_indices,
          //    UA_info, xtmpluS_dev, xtmpS_dev);
          // cusparseScsrmv(handleCusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          //                n, n, n,
          //                &sOne, descrB, QAmySpM.d_val, QAmySpM.d_rowIndices, QAmySpM.d_indices,
          //                xtmpS_dev, &sZero, xOldS_dev);
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
  
  float *xGMRES=(float*)malloc( (B->m)*sizeof(float) );
  for(int i=0; i<(B->m); i++)
    xGMRES[i] = 0.0;

  float *rhsBu=(float*)malloc( (B->m)*sizeof(float) );
  //myMemcpyD2S( rhsBu, rhsDC, B->m );
  cudaMemcpy(rhsBu, myGPUetbr->ut_single_dev, (B->m)*sizeof(float), cudaMemcpyDeviceToHost);

  
  /*
  SpMatrixGPU Left1sparse;
  gpuMallocCpyCSRmySpM(&Left1sparse, &left1MySpM);  
  */
  // printf("Preconditioners available to the solver are:\n"
  //        "        No preconditioner --- 0\n"
  //        "        Diagonal preconditioner --- 1\n"
  //        "        ILU0 preconditioner --- 2\n"
  //        "        Ainv preconditioner --- 3.\n");
  // printf("Please make you selection: ");
  // PreconditionerType prcdtp = ILUK;
  // int ttt;  cin >> ttt;
  // switch(ttt){
  // case 0:
  //   prcdtp = NONE;
  //   printf("No preconditioner will apply.\n");
  //   break;
  // case 1:
  //   prcdtp = DIAG;
  //   printf("Diagonal preconditioner will apply.\n");
  //   break;
  // case 2:
  //   prcdtp = ILU0;
  //   printf("ILU0 preconditioner will apply.\n");
  //   break;
  // case 3:
  //   prcdtp = AINV;
  //   printf("Ainv preconditioner will apply.\n");
  //   break;
  // default:
  //   cerr<<"Unknown type!"<<endl;
  //   exit(-1);
  // }
  // printf("\n");
  
  // Preconditioner *precondG;
  // switch(prcdtp){
  // case NONE:
  //   precondG = (Preconditioner *)new MyNONE();
  //   break;
  // case DIAG:
  //   precondG = (Preconditioner *)new MyDIAG();
  //   break;
  // case ILU0:
  //   precondG = (Preconditioner *)new MyILU0();
  //   break;
  // case ILUK:
  //   precondG = (Preconditioner *)new MyILUK();
  //   break;
  // case AINV:
  //   precondG = (Preconditioner *)new MyAINV();
  //   break;
  // default:
  //   printf("Unknow type\n");
  //   exit(-1);
  // }
  // precondG->Initilize(GmySpM);
  // printf("Preconditioner has been constructed.\n");
  /*
  writeCSR(GmySpM.numRows, GmySpM.numCols, ((MyILUK*)precondG)->LrowIndices_ITSOL[GmySpM.numRows],
           ((MyILUK*)precondG)->LrowIndices_ITSOL, ((MyILUK*)precondG)->Lindices_ITSOL, ((MyILUK*)precondG)->Lval_ITSOL,
           "csrFileL1_ILUK.dat");
  writeCSR(GmySpM.numRows, GmySpM.numCols, ((MyILUK*)precondG)->UrowIndices_ITSOL[GmySpM.numRows],
           ((MyILUK*)precondG)->UrowIndices_ITSOL, ((MyILUK*)precondG)->Uindices_ITSOL, ((MyILUK*)precondG)->Uval_ITSOL,
           "csrFileU1_ILUK.dat");
  */
  /*
  writeCSR(GmySpM.numRows, GmySpM.numCols, ((MyILU0*)precondG)->l_rowIndices[GmySpM.numRows],
           ((MyILU0*)precondG)->l_rowIndices, ((MyILU0*)precondG)->l_indices, ((MyILU0*)precondG)->l_val,
           "csrFileL1_ILU0.dat");
  writeCSR(GmySpM.numRows, GmySpM.numCols, ((MyILU0*)precondG)->u_rowIndices[GmySpM.numRows],
           ((MyILU0*)precondG)->u_rowIndices, ((MyILU0*)precondG)->u_indices, ((MyILU0*)precondG)->u_val,
           "csrFileU1_ILU0.dat");
  */
  // writeCSRmySpMatrix( &left1MySpM, "csrFileLeft1.dat");

  // int max_it = max_iter;
  // float tol = tolerance;
  // int result = GMRES(GmySpM.val, GmySpM.rowIndices, GmySpM.indices, xGMRES, rhsBu, B->m,
  //       	     restart, &max_it, &tol, *precondG);
  // //int result = GMRES(leftMySpM.val, leftMySpM.rowIndices, leftMySpM.indices, xGMRES, rhsBu, B->m,
  // //      	     restart, &max_it, &tol, *precondG);
  // printf("\n");
  // printf("CPU GMRES flag = %d\n", result);
  // printf("  iterations performed: %d\n", max_it);
  // printf("  tolerance achieved: %8.6e\n", tol);
  // //printf("  Time: %.2f (ms)\n", cputime);


  // free(G_csrRowPtr);  free(G_csrColIdx);  free(G_csrVal);
  // free(left_csrRowPtr);  free(left_csrColIdx);  free(left_csrVal);
  // free(right_csrRowPtr);  free(right_csrColIdx);  free(right_csrVal);
  // free(B_csrRowPtr);  free(B_csrColIdx);  free(B_csrVal);
  // free(xGMRES);
  // free(rhsBu);
  //exit(0);
}

// // void memcpyD2S(float *dest, double *src, int N)
// // {
// //   for(int i=0; i<N; i++)
// //     dest[i] = (float)src[i];
// // }
// 
// void addUnitMatrix(MySpMatrix *M )
// {
//   int numRows=M->numRows, numCols=M->numCols, nnz=M->numNZEntries;
//   int *rowPtrNew=(int*)malloc((numRows+1)*sizeof(int));
//   int *colIdxNew=(int*)malloc((nnz+min(numRows,numCols))*sizeof(int));
//   float *valNew=(float*)malloc((nnz+min(numRows,numCols))*sizeof(float));
// 
//   int *rowPtr=M->rowIndices;
//   int *colIdx=M->indices;
//   float *val=M->val;
// 
//   int k=0;
//   rowPtrNew[0] = 0;
//   for(int i=0; i<numRows; i++) {
//     int diagSet=0;
//     int lb=rowPtr[i], ub=rowPtr[i+1];
//     if(lb == ub) {
//       colIdxNew[k] = i;  valNew[k] = 1.0;
//       k++;
//     }
//     else {
//       for(int j=lb; j<ub; j++) {
//         if(colIdx[j] < i) {
//           colIdxNew[k] = colIdx[j];  valNew[k] = val[j];
//           k++;
//         }
//         else if(colIdx[j] == i) {
//           colIdxNew[k] = colIdx[j];  valNew[k] = (val[j]==0.0) ? 1.0 : val[j];
//           k++;
//           diagSet = 1;
//         }
//         else {
//           if(diagSet == 0) {
//             colIdxNew[k] = i;  valNew[k] = 1.0;
//             k++;
//             diagSet = 1;
//           }
//           colIdxNew[k] = colIdx[j];  valNew[k] = val[j];
//           k++;
//         }
//       }
//       if(diagSet == 0) {
//         colIdxNew[k] = i;  valNew[k] = 1.0;
//         k++;
//         diagSet = 1;
//       }
//     }
//     rowPtrNew[i+1] = k;
//   }
//   free(rowPtr); free(colIdx); free(val);
//   M->rowIndices = rowPtrNew;
//   M->indices = colIdxNew;
//   M->val = valNew;
//   M->numNZEntries = k;
// }
// 
// void sort(int *col_idx, float *a, int start, int end)
// {
//   int i, j, it;
//   float dt;
// 
//   for (i=end-1; i>start; i--)
//     for(j=start; j<i; j++)
//       if (col_idx[j] > col_idx[j+1]){
// 
// 	if (a){
// 	  dt=a[j]; 
// 	  a[j]=a[j+1]; 
// 	  a[j+1]=dt;
//         }
// 	it=col_idx[j]; 
// 	col_idx[j]=col_idx[j+1]; 
// 	col_idx[j+1]=it;
// 	  
//       }
// }
// 
// void sortDouble(int *col_idx, double *a, int start, int end)
// {
//   int i, j, it;
//   double dt;
// 
//   for (i=end-1; i>start; i--)
//     for(j=start; j<i; j++)
//       if (col_idx[j] > col_idx[j+1]){
// 
// 	if (a){
// 	  dt=a[j]; 
// 	  a[j]=a[j+1]; 
// 	  a[j+1]=dt;
//         }
// 	it=col_idx[j]; 
// 	col_idx[j]=col_idx[j+1]; 
// 	col_idx[j+1]=it;
// 	  
//       }
// }
// 
// 
// /* converts COO format to CSR format, in-place,
//  * if SORT_IN_ROW is defined, each row is sorted in column index.
//  * On return, i_idx contains row_start position */
// void coo2csr_in(int n, int nz, float *a, int *i_idx, int *j_idx)
// {
//   int *row_start;
//   row_start = (int *)malloc((n+1)*sizeof(int));
//   if (!row_start){
//     printf ("coo2csr_in: cannot allocate temporary memory\n");
//     exit (1);
//   }
// 
//   int i, j;
//   int init, i_next, j_next, i_pos;
//   float dt, a_next;
// 
//   for (i=0; i<=n; i++) row_start[i] = 0;
//   /* determine row lengths */
//   for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
//   for (i=0; i<n; i++) row_start[i+1] += row_start[i]; // csrRowPtr
// 
//   for (init=0; init<nz; ){
//     dt = a[init];
//     i = i_idx[init];
//     j = j_idx[init];
//     i_idx[init] = -1; // flag
//     while (1){
//       i_pos = row_start[i];
//       a_next = a[i_pos];
//       i_next = i_idx[i_pos];
//       j_next = j_idx[i_pos];
// 
//       a[i_pos] = dt;
//       j_idx[i_pos] = j;
//       i_idx[i_pos] = -1;
//       row_start[i]++;
//       if (i_next < 0) break;
//       dt = a_next;
//       i = i_next;
//       j = j_next;
// 
//     }
//     init++;
//     while ( (init < nz) && (i_idx[init] < 0))  init++;
//   }
//   /* shift back row_start */
//   for (i=0; i<n; i++) i_idx[i+1] = row_start[i];
//   i_idx[0] = 0;
// 
//   for (i=0; i<n; i++){
//     sort (j_idx, a, i_idx[i], i_idx[i+1]);
//   }
// 
//   free(row_start);
// }
// 
// void coo2csrDouble_in(int n, int nz, double *a, int *i_idx, int *j_idx)
// {
//   int *row_start;
//   row_start = (int *)malloc((n+1)*sizeof(int));
//   if (!row_start){
//     printf ("coo2csr_in: cannot allocate temporary memory\n");
//     exit (1);
//   }
// 
//   int i, j;
//   int init, i_next, j_next, i_pos;
//   double dt, a_next;
// 
//   for (i=0; i<=n; i++) row_start[i] = 0;
//   /* determine row lengths */
//   for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
//   for (i=0; i<n; i++) row_start[i+1] += row_start[i]; // csrRowPtr
// 
//   for (init=0; init<nz; ){
//     dt = a[init];
//     i = i_idx[init];
//     j = j_idx[init];
//     i_idx[init] = -1; // flag
//     while (1){
//       i_pos = row_start[i];
//       a_next = a[i_pos];
//       i_next = i_idx[i_pos];
//       j_next = j_idx[i_pos];
// 
//       a[i_pos] = dt;
//       j_idx[i_pos] = j;
//       i_idx[i_pos] = -1;
//       row_start[i]++;
//       if (i_next < 0) break;
//       dt = a_next;
//       i = i_next;
//       j = j_next;
// 
//     }
//     init++;
//     while ( (init < nz) && (i_idx[init] < 0))  init++;
//   }
//   /* shift back row_start */
//   for (i=0; i<n; i++) i_idx[i+1] = row_start[i];
//   i_idx[0] = 0;
// 
//   for (i=0; i<n; i++){
//     sortDouble(j_idx, a, i_idx[i], i_idx[i+1]);
//   }
// 
//   free(row_start);
// }
// 
// 
// // /* converts COO format to CSR format, not in-place,
// //  * if SORT_IN_ROW is defined, each row is sorted in column index */
// // void coo2csr(int n, int nz, double *a, int *i_idx, int *j_idx,
// // 	     double *csr_a, int *col_idx, int *row_start)
// // {
// //   int i, l;
// // 
// //   for (i=0; i<=n; i++) row_start[i] = 0;
// //   /* determine row lengths */
// //   for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
// //   for (i=0; i<n; i++) row_start[i+1] += row_start[i];
// // 
// //   /* go through the structure  once more. Fill in output matrix. */
// //   for (l=0; l<nz; l++){
// //     i = row_start[i_idx[l]];
// //     csr_a[i] = a[l];
// //     col_idx[i] = j_idx[l];
// //     row_start[i_idx[l]]++;
// //   }
// // 
// //   /* shift back row_start */
// //   for (i=n; i>0; i--) row_start[i] = row_start[i-1];
// //   row_start[0] = 0;
// // 
// //   for (i=0; i<n; i++){
// //     sort (col_idx, csr_a, row_start[i], row_start[i+1]);
// //   }
// // }
// 
// void LDcsc2csr(long int nnz, long int m, long int n,
//                long int *cscColPtr, long int *cscRowIdx, double *cscVal,
//                int **csrRowPtrIn, int **csrColIdxIn, float **csrValIn)
// { 
//   *csrRowPtrIn=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
//   *csrColIdxIn=(int*)malloc(nnz*sizeof(int));
//   *csrValIn=(float*)malloc(nnz*sizeof(float));
// 
//   int *csrRowPtr = *csrRowPtrIn;
//   int *csrColIdx = *csrColIdxIn;
//   float *csrVal = *csrValIn;
// 
//   for(int j=0; j<n; j++) { // Convert to COO format first.
//     int lb=cscColPtr[j], ub=cscColPtr[j+1];
//     for(int i=lb; i<ub; i++)
//       csrColIdx[i] = j;
//   }
//   //int *rowIdx=(int*)malloc(nnz*sizeof(int));
//   for(int i=0; i<nnz; i++) {
//     csrRowPtr[i] = (int)cscRowIdx[i];
//     csrVal[i] = (float)cscVal[i];
//   }
// 
//   coo2csr_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
// 
//   // for(int i=0; i<nnz-1; i++) {
//   //   for(int j=i+1; j<nnz; j++) {
//   //     if(rowIdx[i] > rowIdx[j]) {
//   //       int tmpRowIdx=rowIdx[i], tmpColIdx=csrColIdx[i];
//   //       double tmpVal=cscVal[i];
//   //       rowIdx[i] = rowIdx[j];  rowIdx[j] = tmpRowIdx;
//   //       csrColIdx[i] = csrColIdx[j];  csrColIdx[j] = tmpColIdx;
//   //       csrVal[i] = csrVal[j];  csrVal[j] = tmpVal;
//   //     }
//   //   }
//   // }
//   // 
//   // int i=0;
//   // csrRowPtr[0] = 0;
//   // for(int j=0; j<nnz; j++) {
//   //   if(rowIdx[j] > i) { /* Empty rows are considered. */
//   //     for(int k=i+1; k<=rowIdx[j]; k++)
//   //       csrRowPtr[k] = j;
//   //     i = rowIdx[j];
//   //   }
//   // }
//   // for( ; i<m+1; i++)
//   //   csrRowPtr[i] = nnz;
//   // free(rowIdx);
// }
// 
// 
// void LDcsc2csrMySpMatrix(MySpMatrix *mySpM, ucr_cs_dl *M)
// {
//   mySpM->isCSR = 1;
//   
//   int nnz = M->nzmax;
//   int m = M->m;
//   int n = M->n;
// 
//   mySpM->numRows = m;
//   mySpM->numCols = n;
//   mySpM->numNZEntries = nnz;
// 
//   mySpM->rowIndices=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
//   mySpM->indices=(int*)malloc(nnz*sizeof(int));
//   mySpM->val=(float*)malloc(nnz*sizeof(float));
// 
//   int *csrRowPtr = mySpM->rowIndices;
//   int *csrColIdx = mySpM->indices;
//   float *csrVal = mySpM->val;
// 
//   for(int j=0; j<n; j++) { // Convert to COO format first.
//     int lb=M->p[j], ub=M->p[j+1];
//     for(int i=lb; i<ub; i++)
//       csrColIdx[i] = j;
//   }
//   //int *rowIdx=(int*)malloc(nnz*sizeof(int));
//   for(int i=0; i<nnz; i++) {
//     csrRowPtr[i] = (int) M->i[i];
//     csrVal[i] = (float) M->x[i];
//     // if( csrVal[i] == 0 && M->x[i] != 0 )
//     //   printf(" Accuracy loss M->x[%d]=%6.4e\n",i,M->x[i]);
//   }
// 
//   coo2csr_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
// }
// void LDcsc2cscMySpMatrix(MySpMatrix *mySpM, ucr_cs_dl *M)
// {
//   mySpM->isCSR = 0;
//   
//   int nnz = M->nzmax;
//   int m = M->m;
//   int n = M->n;
// 
//   mySpM->numRows = m;
//   mySpM->numCols = n;
//   mySpM->numNZEntries = nnz;
// 
//   mySpM->rowIndices=(int*)malloc(((n+1))*sizeof(int)); // Only first m+1 elements are useful on return.
//   mySpM->indices=(int*)malloc(nnz*sizeof(int));
//   mySpM->val=(float*)malloc(nnz*sizeof(float));
// 
//   int *csrRowPtr = mySpM->rowIndices;
//   int *csrColIdx = mySpM->indices;
//   float *csrVal = mySpM->val;
// 
//   for(int j=0; j<n+1; j++)
//     csrRowPtr[j] = M->p[j];
//   for(int i=0; i<nnz; i++) {
//     csrColIdx[i] = M->i[i];
//     csrVal[i] = (float) M->x[i];
//     // if( csrVal[i] == 0 && M->x[i] != 0 )
//     //   printf(" Accuracy loss M->x[%d]=%6.4e\n",i,M->x[i]);
//   }
// 
//   //coo2csr_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
// }
// 
// void LDcsc2csrMySpMatrixDouble(MySpMatrixDouble *mySpM, ucr_cs_dl *M)
// {
//   int nnz = M->nzmax;
//   int m = M->m;
//   int n = M->n;
// 
//   mySpM->numRows = m;
//   mySpM->numCols = n;
//   mySpM->numNZEntries = nnz;
// 
//   mySpM->rowIndices=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
//   mySpM->indices=(int*)malloc(nnz*sizeof(int));
//   mySpM->val=(double*)malloc(nnz*sizeof(double));
// 
//   int *csrRowPtr = mySpM->rowIndices;
//   int *csrColIdx = mySpM->indices;
//   double *csrVal = mySpM->val;
// 
//   for(int j=0; j<n; j++) { // Convert to COO format first.
//     int lb=M->p[j], ub=M->p[j+1];
//     for(int i=lb; i<ub; i++)
//       csrColIdx[i] = j;
//   }
//   //int *rowIdx=(int*)malloc(nnz*sizeof(int));
//   for(int i=0; i<nnz; i++) {
//     csrRowPtr[i] = (int) M->i[i];
//     csrVal[i] = M->x[i];
//   }
// 
//   coo2csrDouble_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
// }
// 
// /* for (k = 0 ; k < n ; k++)
//  *     x [p ? p [k] : k] = b [k] ; */
// void vec2csrMySpMatrix(MySpMatrix *mySpM, int *p, int n)
// {
//   mySpM->isCSR = 1;
// 
//   mySpM->numRows = n;
//   mySpM->numCols = n;
//   mySpM->numNZEntries = n;
// 
//   mySpM->rowIndices=(int*)malloc((n+1)*sizeof(int)); // Only first m+1 elements are useful on return.
//   mySpM->indices=(int*)malloc(n*sizeof(int));
//   mySpM->val=(float*)malloc(n*sizeof(float));
// 
//   int *csrRowPtr = mySpM->rowIndices;
//   int *csrColIdx = mySpM->indices;
//   float *csrVal = mySpM->val;
//   for(int i=0; i<n; i++) {
//     csrVal[i] = 1.0;
//     csrColIdx[p[i]] = i;
//   }
//   for(int i=0; i<n+1; i++)
//     csrRowPtr[i] = i;
// }
// 
// void writeCSR(int m, int n, int nnz,
//               int *rowPtr, int *colIdx, float *val,
//               const char *filename)
// {
//   FILE *f;
//   f = fopen(filename,"wb");
//   if(!f) {
//     fprintf(stdout,"Cannot open file: %s\n",filename);
//     exit(-1);
//   }
// 
//   fwrite(&m, sizeof(int), 1, f);
//   fwrite(&n, sizeof(int), 1, f);
//   fwrite(&nnz, sizeof(int), 1, f);
// 
//   fwrite(rowPtr, sizeof(int), m+1, f);
//   fwrite(colIdx, sizeof(int), nnz, f);
//   fwrite(val, sizeof(float), nnz, f);
//   
//   fclose(f);
// }
// 
// void writeCSRmySpMatrix(MySpMatrix *M, const char *filename)
// {
//   FILE *f;
//   f = fopen(filename,"wb");
//   if(!f) {
//     fprintf(stdout,"Cannot open file: %s\n",filename);
//     exit(-1);
//   }
// 
//   fwrite(&(M->numRows), sizeof(int), 1, f);
//   fwrite(&(M->numCols), sizeof(int), 1, f);
//   fwrite(&(M->numNZEntries), sizeof(int), 1, f);
// 
//   fwrite(M->rowIndices, sizeof(int), M->numRows + 1, f);
//   fwrite(M->indices, sizeof(int), M->numNZEntries, f);
//   fwrite(M->val, sizeof(float), M->numNZEntries, f);
//   
//   fclose(f);
// }
// 
// void writeCSRmySpMatrixDouble(MySpMatrixDouble *M, const char *filename)
// {
//   FILE *f;
//   f = fopen(filename,"wb");
//   if(!f) {
//     fprintf(stdout,"Cannot open file: %s\n",filename);
//     exit(-1);
//   }
// 
//   fwrite(&(M->numRows), sizeof(int), 1, f);
//   fwrite(&(M->numCols), sizeof(int), 1, f);
//   fwrite(&(M->numNZEntries), sizeof(int), 1, f);
// 
//   fwrite(M->rowIndices, sizeof(int), M->numRows + 1, f);
//   fwrite(M->indices, sizeof(int), M->numNZEntries, f);
//   fwrite(M->val, sizeof(double), M->numNZEntries, f);
//   
//   fclose(f);
// }
// 
// void writeUt(float *val, const int lenU, const int nTs, const char *filename)
// {
//   FILE *f;
//   f = fopen(filename,"wb");
//   if(!f) {
//     fprintf(stdout,"Cannot open file: %s\n",filename);
//     exit(-1);
//   }
// 
//   fwrite(&lenU, sizeof(int), 1, f);
//   fwrite(&nTs, sizeof(int), 1, f);
//   fwrite(val, sizeof(float), lenU*nTs, f);
//   
//   fclose(f);
// }
