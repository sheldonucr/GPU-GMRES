/*
 * Data structure for GPU based transient simulation.
 *
 * The structure is passed to related functions where data
 * is recorded from CPU preparation and copied to GPU for parallel computing.
 *
 * Even if the "-gpu" option at command line is not given,
 * this data structure is still declared, but the member data is not fulfilled.
 *
 * Author: Xue-Xin Liu
 *         2011-Nov-02
 */
#pragma once
#ifndef __GPU_DATA_H_
#define __GPU_DATA_H_

#include <stdio.h>
#include "SpMV.h"

// for GPU kernel of IPIV
#define MAX_THREADS 512

// for GPU kernel of Ut generation
#define BLK_SIZE_UTGEN 128

// for PWL info,
// a pair of time and value is called PTS here and is counted as one.
#define MAX_PWL_PTS 64

#define HALFWARPSIZE 16

#define PART_LEN 1024

/*
 * original system:
 *       G and C --- [n,n]
 *       B       --- [n,m]
 *       L       --- [p,n]
 *       x       --- [n,1]
 * reduced system:
 *       Gr and Cr --- [q,q]
 */
typedef struct{
  int numPts; // time zero is counted
  int n;
  int q;
  int m; // Currently, it should be equal to nVS+nIS.
  int nport;

  int use_cuda_single, use_cuda_double;

  double tstep, tstop;
  double *ut_host; // [m,numPts-1]

  int *ipiv_host;
  double *V_host;
  double *LV_host; // Xp
  double *L_hCG_host;
  double *U_hCG_host;
  double *hC_host;
  double *Br_host;
  double *xr0_host;
  double *x_host;

  float *V_single_host;
  float *LV_single_host; // Xp
  float *L_hCG_single_host;
  float *U_hCG_single_host;
  float *hC_single_host;
  float *Br_single_host;
  float *xr0_single_host;
  float *x_single_host;

  int ldUt; // a multiple of 32;
  double *ut_dev; // time zero is not included
  int *ipiv_dev;
  double *V_dev;
  double *LV_dev; // Xp
  double *L_hCG_dev;
  double *U_hCG_dev;
  double *hC_dev;
  double *Br_dev;
  double *xr_dev;
  double *x_dev;

  float *ut_single_dev; // time zero is not included
  float *V_single_dev;
  float *LV_single_dev; // Xp
  float *L_hCG_single_dev;
  float *U_hCG_single_dev;
  float *hC_single_dev;
  float *Br_single_dev;
  float *xr_single_dev;
  float *x_single_dev;

  int nIS, nVS;
  double *dcVt_host, *dcVt_dev; // value of voltage source, DC only
  float *dcVt_single_host, *dcVt_single_dev; // value of voltage source, DC only
  // source info: currently only support PWL and PULSE,
  //              and only one type is allowed in a netlist.
  //              They should be current sources only.
  int PWLvolExist, PWLcurExist, PULSEvolExist, PULSEcurExist;

  int *PWLnumPts_host, *PWLnumPts_dev;
  double *PWLtime_host, *PWLtime_dev;// maximum length is MAX_PWL_PTS
  double *PWLval_host, *PWLval_dev; // maximum length is MAX_PWL_PTS

  float *PWLtime_single_host, *PWLtime_single_dev;// maximum length is MAX_PWL_PTS
  float *PWLval_single_host, *PWLval_single_dev; // maximum length is MAX_PWL_PTS

  double *PULSEtime_host, *PULSEtime_dev;
  double *PULSEval_host, *PULSEval_dev;

  float *PULSEtime_single_host, *PULSEtime_single_dev;
  float *PULSEval_single_host, *PULSEval_single_dev;
} gpuETBR;


#ifdef __cplusplus
extern "C"
#endif
void cudaTranSim(gpuETBR *myGPUetbr);


#ifdef __cplusplus
extern "C"
#endif
void cudaTranSim_shutdown(gpuETBR *myGPUetbr);


#ifdef __cplusplus
extern "C"
#endif
void gpuRelatedDataInit(gpuETBR *myGPUetbr);

#ifdef __cplusplus
extern "C"
#endif
void gpuRelatedDataFree(gpuETBR *myGPUetbr);


void myMemcpyD2S(float *dst, double *src, int n);
void myMemcpyL2I(int *dst, long int *src, int n);


class ucr_cs_dl{
 public:
  long int nzmax ; /* maximum number of entries */
  long int m ;	    /* number of rows */
  long int n ;	    /* number of columns */
  long int *p ;    /* column pointers (size n+1) or col indlces (size nzmax) */
  long int *i ;    /* row indices, size nzmax */
  double *x ;	    /* numerical values, size nzmax */
  long int nz ;

  void shallowCpy(long int nzmaxIn, long int mIn, long int nIn, long int *pIn, long int *iIn,
                  double *xIn, long int nzIn)
  {
    nzmax = nzmaxIn;
    m = mIn;
    n = nIn;
    p = pIn;
    i = iIn;
    x = xIn;
    nz = nzIn;
  }
};

class ucr_cs_di{ // column based
 public:
  int nzmax ; /* maximum number of entries */
  int m ;	    /* number of rows */
  int n ;	    /* number of columns */
  int *p ;    /* column pointers (size n+1) or col indlces (size nzmax) */
  int *i ;    /* row indices, size nzmax */
  double *x ;	    /* numerical values, size nzmax */
  int nz ;

  void convertFromCS_DL(long int nzmaxIn, long int mIn, long int nIn,
                        long int *pIn, long int *iIn, double *xIn, long int nzIn)
  {
    // ILU++ use the memory allocated here to construct matrix.
    // Since ILU++ uses delete [] to free memory, we use new here.
    nzmax = nzmaxIn;
    m = mIn;
    n = nIn;
    p = new int [n+1];
    for(int j=0; j<n+1; j++)  p[j] = pIn[j];
    i = new int [nzmax];
    x = new double [nzmax];
    for(int j=0; j<nzmax; j++) {
      i[j] = iIn[j];
      x[j] = xIn[j];
    }
    nz = nzIn;
  }

  // ~ucr_cs_di() {
  //   free(p);
  //   free(i);
  //   free(x);
  // }
};

void coo2csr_in(int numRows, int nz, float *a, int *i_idx, int *j_idx);
void coo2csrDouble_in(int numRows, int nz, double *a, int *i_idx, int *j_idx);
void LDcsc2csrMySpMatrix(MySpMatrix *mySpM, ucr_cs_dl *M);
void LDcsc2cscMySpMatrix(MySpMatrix *mySpM, ucr_cs_dl *M);
void LDcsc2csrMySpMatrixDouble(MySpMatrixDouble *mySpM, ucr_cs_dl *M);
void vec2csrMySpMatrix(MySpMatrix *mySpM, int *p, int n);
void writeCSRmySpMatrix(MySpMatrix *M, const char *filename);
void writeCSRmySpMatrixDouble(MySpMatrixDouble *M, const char *filename);

void writeCSR(int m, int n, int nnz,
              int *rowPtr, int *colIdx, float *val,
              const char *filename);

void wrapperGMRESforPG(ucr_cs_dl *left,
                       ucr_cs_dl *right, ucr_cs_dl *G, ucr_cs_dl *B, //double *w,
                       int *invPort, int nport,
                       // ucr_cs_dl *LG, ucr_cs_dl *UG, int *pinvG, int *qinvG,
                       // ucr_cs_dl *LA, ucr_cs_dl *UA, int *pinvA, int *qinvA,
                       gpuETBR *myGPUetbr);

void wrapperGPUforPG(ucr_cs_dl *left,
                       ucr_cs_dl *right, ucr_cs_dl *G, ucr_cs_dl *B, //double *w,
                       int *invPort, int nport,
                       ucr_cs_dl *LG, ucr_cs_dl *UG, int *pinvG, int *qinvG,
                       ucr_cs_dl *LA, ucr_cs_dl *UA, int *pinvA, int *qinvA,
                       gpuETBR *myGPUetbr);



class SpaFmt {
/*--------------------------------------------- 
| C-style CSR format - used internally
| for all matrices in CSR format 
|---------------------------------------------*/
 public:
  int n;
  int *nzcount;  /* length of each row */
  int **ja;      /* pointer-to-pointer to store column indices  */
  float **ma;   /* pointer-to-pointer to store nonzero entries */
};
typedef SpaFmt SparMat;
typedef SpaFmt* csptr;


typedef struct ILUfac {
    int n;
    csptr L;      /* L part elements                            */
    float *D;    /* diagonal elements                          */
    csptr U;      /* U part elements                            */
    int *work;    /* working buffer */
} ILUSpar, LDUmat, *iluptr;


// Added by XXLiu from protos.h.
//int setupILU( iluptr lu, int n );
int mallocRow( iluptr lu, int nrow );
// int nnz_cs (csptr A) ;
int setupCS(csptr amat, int len, int job);
int cleanCS(csptr amat);

void *Malloc( int nbytes, char *msg );
int ilukC( int lofM, csptr csmat, iluptr lu, FILE *fp );
int cleanILU( iluptr lu );

void setGPUdevice();

#endif

/*
  Should be careful about stamp_sub() for the source parser.

*/
