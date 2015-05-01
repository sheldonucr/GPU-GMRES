#include <stdio.h>

#include "preconditioner.h"
#include "gmres.h"
#include "gmres_interface_pg.h"

#define gmres_tol_global 1e-7

void gmresInterfacePG::setPrecondPG(MySpMatrix *A,
                                    MySpMatrixDouble *PrLeft, MySpMatrixDouble *PrRight,
                                    MySpMatrix *PrMiddle,
                                    MySpMatrix *PrPermRow, MySpMatrix *PrPermCol,
                                    MySpMatrixDouble *PrLscale, MySpMatrixDouble *PrRscale)
{
  matrixSize = A->numRows;
  h_val = A->val;
  h_rowPtr = A->rowIndices;
  h_colIdx = A->indices;
  
  xgmres_h = (float*)malloc(matrixSize*sizeof(float));
  rhs_h = (float*)malloc(matrixSize*sizeof(float));
  
  Precond = (Preconditioner *)new MyILUPP(); // MyNONE;//
  //((MyILUPP *) Precond)->Initilize(*A);
  ((MyILUPP *) Precond)->Initilize(*PrLeft, *PrRight, *PrMiddle, *PrPermRow, *PrPermCol,
                                   *PrLscale, *PrRscale);
  printf("ILU++double has been constructed.\n");
}

void gmresInterfacePGfloat::setPrecondPG(MySpMatrix *A,
                                         MySpMatrixDouble *PrLeft, MySpMatrixDouble *PrRight,
                                         MySpMatrix *PrMiddle,
                                         MySpMatrix *PrPermRow, MySpMatrix *PrPermCol,
                                         MySpMatrix *PrLscale, MySpMatrix *PrRscale)
{
  matrixSize = A->numRows;
  h_val = A->val;
  h_rowPtr = A->rowIndices;
  h_colIdx = A->indices;
  nnz = h_rowPtr[matrixSize];
  
  cudaMalloc((void**)&d_val, nnz*sizeof(float));
  cudaMalloc((void**)&d_rowPtr, (matrixSize+1)*sizeof(int));
  cudaMalloc((void**)&d_colIdx, nnz*sizeof(int));
  cudaMemcpy(d_val, h_val, nnz*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rowPtr, h_rowPtr, (matrixSize+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colIdx, h_colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);

  xgmres_h = (float*)malloc(matrixSize*sizeof(float));
  rhs_h = (float*)malloc(matrixSize*sizeof(float));

  cudaMalloc((void**)&xgmres_d, matrixSize*sizeof(float));
  cudaMalloc((void**)&rhs_d, matrixSize*sizeof(float));

  Precond = (Preconditioner *)new MyILUPPfloat(); // MyNONE;//
  //((MyILUPP *) Precond)->Initilize(*A);
  ((MyILUPPfloat *) Precond)->Initilize(*PrLeft, *PrRight, *PrMiddle, *PrPermRow, *PrPermCol,
                                        *PrLscale, *PrRscale, restart);
  // printf("ILU++float has been constructed.\n");
}

int gmresInterfacePG::GMRES_host_PG()
{
  Preconditioner *precond=(Preconditioner *)Precond;
  
  max_it = 10000;//max_iter;
  tol = gmres_tol_global;//1e-7;//tolerance;
  
  timeval st, et;
  gettimeofday(&st, NULL);
  // solve with preconditioned GMRES on Host
  // for(int i=0; i<N; i++)  xTranGMREShost[i] = 0.0;
  int result = GMRESilu(h_val, h_rowPtr, h_colIdx, xgmres_h, rhs_h, matrixSize,
                        restart, &max_it, &tol, *precond);
  gettimeofday(&et, NULL);
  // float cputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
  // printf("CPU GMRES flag = %d\n", result);
  // printf("  iterations performed: %d\n", max_it);
  // printf("  tolerance achieved: %8.6e\n", tol);
  // printf("  Time: %.2f (ms)\n", cputime);
  if(result != 0)
    printf("Failed to converge.\n");
  return result;
}

int gmresInterfacePGfloat::GMRES_host_PG()
{
  Preconditioner *precond=(Preconditioner *)Precond;
  
  int max_it = max_iter;
  float tol = gmres_tol_global;//1e-6;//tolerance;
  
  timeval st, et;
  gettimeofday(&st, NULL);
  // solve with preconditioned GMRES on Host
  // for(int i=0; i<N; i++)  xTranGMREShost[i] = 0.0;
  int result = GMRESilu(h_val, h_rowPtr, h_colIdx, xgmres_h, rhs_h, matrixSize,
                        restart, &max_it, &tol, *precond);
  gettimeofday(&et, NULL);
  // float cputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
  // printf("CPU GMRES flag = %d\n", result);
  // printf("  iterations performed: %d\n", max_it);
  // printf("  tolerance achieved: %8.6e\n", tol);
  // printf("  Time: %.2f (ms)\n", cputime);
  if(result != 0)
    printf("Failed to converge.\n");
  return result;
}

int gmresInterfacePGfloat::GMRES_dev_PG()
{
  Preconditioner *precond=(Preconditioner *)Precond;
  
  max_it = 10000;//max_iter;
  tol = gmres_tol_global;//1e-7;//tolerance;
  

  cudaMemcpy(xgmres_d, xgmres_h, matrixSize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_d, rhs_h, matrixSize*sizeof(float), cudaMemcpyHostToDevice);
  timeval st, et;
  gettimeofday(&st, NULL);
  // solve with preconditioned GMRES on Host
  // for(int i=0; i<N; i++)  xTranGMREShost[i] = 0.0;
  int result = GMRESilu_GPU(d_val, d_rowPtr, d_colIdx, nnz,
                            xgmres_d, rhs_d, matrixSize,
                            restart, &max_it, &tol, *precond);
  gettimeofday(&et, NULL);

  cudaMemcpy(xgmres_h, xgmres_d, matrixSize*sizeof(float), cudaMemcpyDeviceToHost);

  // float cputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
  // printf("CPU GMRES flag = %d\n", result);
  // printf("  iterations performed: %d\n", max_it);
  // printf("  tolerance achieved: %8.6e\n", tol);
  // printf("  Time: %.2f (ms)\n", cputime);
  if(result != 0)
    printf("Failed to converge.\n");
  return result;
}


gmresInterfacePG::~gmresInterfacePG()
{
  free(xgmres_h);
  free(rhs_h);
  delete (Preconditioner *)Precond;
}


gmresInterfacePGfloat::~gmresInterfacePGfloat()
{
  free(xgmres_h);
  free(rhs_h);

  cudaFree(xgmres_d);
  cudaFree(rhs_d);
  cudaFree(d_val);
  cudaFree(d_rowPtr);
  cudaFree(d_colIdx);
  

  delete (Preconditioner *)Precond;
}
