/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: dc_solver.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:54 $
 *    Authors: Duo Li
 *
 *    Functions: DC solver
 *
 */

#include <iostream>
#include <itpp/base/timing.h>
#include <itpp/base/mat.h>
#include <itpp/base/vec.h>
#include <itpp/base/specmat.h>
#include <itpp/base/algebra/lapack.h>
#include <itpp/base/algebra/ls_solve.h>
#include <itpp/signal/transforms.h>
#include "umfpack.h"
#include "etbr.h"
#include "etbr_dd.h"
#include "cs.h"

using namespace itpp;

void dc_solver(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value)
{
  Real_Timer umfpack_symbolic, umfpack_numeric, umfpack_solve;
  Real_Timer umfpack_run_time, svd_run_time, rmatrix_run_time;
  vec vs(nVS);
  vec is(nIS);
  vec u(nVS+nIS);
  for (int i = 0; i < nVS; i++){
	vs(i) = VS[i].value(0);
  }
  for (int i = 0; i < nIS; i++){
	is(i) = IS[i].value(0);
  }
  u = concat(vs, is);

  umfpack_run_time.start();
  UF_long nDim = B->m;
  UF_long nSDim = B->n;
  UF_long *Ap = G->p; 
  UF_long *Ai = G->i;
  double *Ax = G->x;
  void *Symbolic, *Numeric;
  double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
  umfpack_dl_defaults(Control);
  umfpack_symbolic.start();
  (void) umfpack_dl_symbolic (nDim, nDim, Ap, Ai, Ax, &Symbolic, Control, Info);
  umfpack_symbolic.stop();
  if (Info[0] == -1){
	std::cout << "UMFPACK ERROR: symbolic out of memory" << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);
  }else if (Info[0] < 0){
	std::cout << "Info[0] = " << Info[0] << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);
  }

  umfpack_numeric.start();
  (void) umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
  umfpack_numeric.stop();
  if (Info[0] == -1){
	std::cout << "UMFPACK ERROR: numeric out of memory" << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);	  
  }else if (Info[0] < 0){
	std::cout << "Info[0] = " << Info[0] << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);
  }
  umfpack_dl_free_symbolic (&Symbolic);

  /* solve Gx = Bu  */
  vec b(nDim);
  b.zeros();
  (void) cs_dl_gaxpy(B, u._data(), b._data());
  umfpack_solve.start();
  dc_value.set_size(nDim);
  dc_value.zeros();
  (void) umfpack_dl_solve (UMFPACK_A, Ap, Ai, Ax, dc_value._data(), b._data(), Numeric, Control, Info);	
  umfpack_solve.stop();
  if (Info[0] == -1){
	std::cout << "UMFPACK ERROR: solve out of memory" << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);
  }else if (Info[0] < 0){
	std::cout << "Info[0] = " << Info[0] << std::endl;
	umfpack_dl_report_info(Control, Info);
	exit(-1);
  }
  umfpack_dl_free_numeric (&Numeric); 
  umfpack_run_time.stop();	
}

void dc_dd_solver(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value,
				  int npart, UF_long *part_size, UF_long *node_part, 
				  UF_long *mat_pinv, UF_long *mat_q)
{
  Real_Timer form_dd_run_time, dd_solve_run_time;
  Real_Timer symbolic_runtime, numeric_runtime, solve_runtime;
  vec vs(nVS);
  vec is(nIS);
  vec u(nVS+nIS);
  for (int i = 0; i < nVS; i++){
	vs(i) = VS[i].value(0);
  }
  for (int i = 0; i < nIS; i++){
	is(i) = IS[i].value(0);
  }
  u = concat(vs, is);

  /* solve Az = b using Domain Decompositon */
  UF_long nDim = B->m;
  UF_long nSDim = B->n;
  cs_dl **As = new cs_dl*[npart];
  cs_dl **E = new cs_dl*[npart];
  cs_dl **F = new cs_dl*[npart];
  cs_dl *At;
  double **f = new double*[npart];
  double *g;
  cs_dl *A_dd = cs_dl_permute(G, mat_pinv, mat_q, 1);
  
  /*
  cs_dl *A = G;
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(0);
  mat A_mat(A->m, A->n);
  A_mat.zeros();
  for (UF_long j = 0; j < A->n; j++){
	for (UF_long p = A->p[j]; p < A->p[j+1]; p++){
	  A_mat.set(A->i[p], j, A->x[p]);
	}
  }
  std::cout << "** A = " << std::endl;
  for (int r = 0; r < A_mat.rows(); r++){
	for (int c = 0; c < A_mat.cols(); c++){
	  std::cout << A_mat(r,c) << "   " ;
	}
	std::cout<< std::endl;
  }
  mat A_dd_mat(A_dd->m, A_dd->n);
  A_dd_mat.zeros();
  for (UF_long j = 0; j < A_dd->n; j++){
	for (UF_long p = A_dd->p[j]; p < A_dd->p[j+1]; p++){
	  A_dd_mat.set(A_dd->i[p], j, A_dd->x[p]);
	}
  }
  std::cout << "** A_dd = " << std::endl;
  for (int r = 0; r < A_dd_mat.rows(); r++){
	for (int c = 0; c < A_dd_mat.cols(); c++){
	  std::cout << A_dd_mat(r,c) << "   " ;
	}
	std::cout<< std::endl;
  }
  */

  cs_dl_spfree(G);
  vec b(nDim);
  b.zeros();
  (void) cs_dl_gaxpy(B, u._data(), b._data());
  double *b_dd = new double[nDim];
  cs_dl_pvec(mat_q, b._data(), b_dd, nDim);

  form_dd_run_time.start();
  dd_form(npart, part_size, node_part, A_dd, b_dd, As, E, F, At, f, g);
  form_dd_run_time.stop();
  delete [] b_dd;

  cs_dl_spfree(A_dd);
  double *z_dd = new double[nDim];
  dd_solve_run_time.start();
  dd_solve_ooc(npart, As, E, F, At, f, g, z_dd, symbolic_runtime, numeric_runtime, solve_runtime);
  dd_solve_run_time.stop();

  dc_value.set_size(nDim);
  dc_value.zeros();
  cs_dl_ipvec(mat_q, z_dd, dc_value._data(), nDim);

  delete [] z_dd;
  for (int k = 0; k < npart; k++){
	cs_dl_spfree(As[k]);
  }
  delete [] As;	
  for (int k = 0; k < npart; k++){
	cs_dl_spfree(E[k]);
  }
  delete [] E;
  for (int k = 0; k < npart; k++){
	cs_dl_spfree(F[k]);
  }
  delete [] F;
  cs_dl_spfree(At);
  for (int k = 0; k < npart; k++){
	delete [] f[k];
  }
  delete [] f;
  delete [] g;

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "form_dd         \t: " << form_dd_run_time.get_time() << std::endl;
  std::cout << "dd_solve        \t: " << dd_solve_run_time.get_time() << std::endl;
  std::cout << "symbolic        \t: " << symbolic_runtime.get_time() << std::endl;
  std::cout << "numeric         \t: " << numeric_runtime.get_time() << std::endl;
  std::cout << "solve           \t: " << solve_runtime.get_time() << std::endl;
}

void dc_solver2(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value)
{
  Real_Timer umfpack_symbolic, umfpack_numeric, umfpack_solve;
  Real_Timer umfpack_run_time, svd_run_time, rmatrix_run_time;
  vec vs(nVS);
  vec is(nIS);
  vec u(nVS+nIS);
  for (int i = 0; i < nVS; i++){
	vs(i) = VS[i].value(0);
  }
  for (int i = 0; i < nIS; i++){
	is(i) = IS[i].value(0);
  }
  u = concat(vs, is);

  umfpack_run_time.start();
  UF_long nDim = B->m;
  UF_long nSDim = B->n;
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;
  umfpack_symbolic.start();
  Symbolic = cs_dl_sqr(order, G, 0);
  umfpack_symbolic.stop();

  umfpack_numeric.start();
  Numeric = cs_dl_lu(G, Symbolic, tol);
  umfpack_numeric.stop();
  UF_long *Symbolic_q = new UF_long[G->n];
  for (UF_long j = 0; j < G->n; j++){
	Symbolic_q[j] = Symbolic->q[j];
  }
  cs_dl_sfree(Symbolic);

  /* solve Gx = Bu  */
  vec b(nDim);
  b.zeros();
  vec x(nDim);
  x.zeros();
  (void) cs_dl_gaxpy(B, u._data(), b._data());
  umfpack_solve.start();
  dc_value.set_size(nDim);
  dc_value.zeros();
  cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), G->n);
  cs_dl_lsolve(Numeric->L, x._data());
  cs_dl_usolve(Numeric->U, x._data());
  cs_dl_ipvec(Symbolic_q, x._data(), dc_value._data(), G->n);  

  cs_dl_nfree(Numeric);
  umfpack_run_time.stop();
	
#ifndef UCR_EXTERNAL
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "Symbolic        \t: " << umfpack_symbolic.get_time() << std::endl;
  std::cout << "Numeric         \t: " << umfpack_numeric.get_time() << std::endl;
  std::cout << "Solve           \t: " << umfpack_solve.get_time() << std::endl;
  std::cout << "Total           \t: " << umfpack_run_time.get_time() << std::endl;
#endif

}
