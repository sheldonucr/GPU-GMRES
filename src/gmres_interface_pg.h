#ifndef _GMRES_INTERFACE_PG_H_
#define _GMRES_INTERFACE_PG_H_
#include "SpMV.h"

class gmresInterfacePG {
 public:
  ~gmresInterfacePG();
  
  int matrixSize;

  float *h_val;
  int *h_rowPtr;
  int *h_colIdx;

  // float *d_val;
  // int *d_rowPtr;
  // int *d_colIdx;

  float *x_h;
  float *x_d;
  
  float *xgmres_h;
  float *rhs_h;

  void *Precond;
  
  int max_it; // both input and output
  float tol;

  void setPrecondPG(MySpMatrix *A,
                    MySpMatrixDouble *PrLeft, MySpMatrixDouble *PrRight,
                    MySpMatrix *PrMiddle_mySpM,
                    MySpMatrix *PrPermRow, MySpMatrix *PrPermCol,
                    MySpMatrixDouble *PrLscale, MySpMatrixDouble *PrRscale);
  int GMRES_host_PG();  
};

class gmresInterfacePGfloat {
 public:
  ~gmresInterfacePGfloat();
  
  int matrixSize;
  int nnz;
  float *h_val;
  int *h_rowPtr;
  int *h_colIdx;

  float *d_val;
  int *d_rowPtr;
  int *d_colIdx;

  float *x_h;
  float *x_d;
  
  float *xgmres_h;
  float *rhs_h;

  float *xgmres_d;
  float *rhs_d;

  void *Precond;

  int max_it; // both input and output
  float tol;

  void setPrecondPG(MySpMatrix *A,
                    MySpMatrixDouble *PrLeft, MySpMatrixDouble *PrRight,
                    MySpMatrix *PrMiddle_mySpM,
                    MySpMatrix *PrPermRow, MySpMatrix *PrPermCol,
                    MySpMatrix *PrLscale, MySpMatrix *PrRscale);
  int GMRES_host_PG();
  int GMRES_dev_PG();  
};

#endif
