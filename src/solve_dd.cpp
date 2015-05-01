/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: solve_dd.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:01 $
 *    Authors: Duo Li
 *
 *    Functions: solve Az = b with Domain Decomposition
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <itpp/base/timing.h>
#include <itpp/base/mat.h>
#include <itpp/base/vec.h>
#include <itpp/base/algebra/lapack.h>
#include <itpp/base/algebra/ls_solve.h>
#include "cs.h"
#include "umfpack.h"
#include "etbr_dd.h"

void dd_solve(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			  double **f, double *g, double *z,
			  Real_Timer &umfpack_symbolic, Real_Timer &umfpack_numeric, Real_Timer &umfpack_solve)
{
  mat S(At->m, At->n);
  mat FAE(At->m, At->n);
  vec* Ab = new vec[npart];
  S.zeros();
  // initial S = At
  for (UF_long j = 0; j < At->n; j++){
	for (UF_long p = At->p[j]; p < At->p[j+1]; p++){
	  S.set(At->i[p], j, At->x[p]);
	}
  }
  vec gg(g, At->m);
  double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
  umfpack_dl_defaults(Control);
  void *Symbolic;
  void **Numeric = new void* [npart];
  for (int k = 0; k < npart; k++){
	umfpack_symbolic.start();
	(void) umfpack_dl_symbolic (As[k]->m, As[k]->n, As[k]->p, As[k]->i, As[k]->x, &Symbolic, Control, Info);
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
	(void) umfpack_dl_numeric (As[k]->p, As[k]->i, As[k]->x, Symbolic, &Numeric[k], Control, Info);
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
	umfpack_dl_free_symbolic(&Symbolic);
	// compute FAE and S;
	FAE.zeros();
	double *e = new double[E[k]->m];
	double *ae = new double[E[k]->m];
	int counter = 0;
	for (UF_long j = 0; j < At->n; j++){	  
	  for (int i = 0; i < E[k]->m; i++){
		e[i] = 0;
		ae[i] = 0;
	  }
	  bool nonzero = false;
	  for (UF_long p = E[k]->p[j]; p < E[k]->p[j+1]; p++){
		e[E[k]->i[p]] = E[k]->x[p];
		nonzero = true;
	  }
	  if (nonzero){
		umfpack_solve.start();
		(void) umfpack_dl_solve(UMFPACK_A, As[k]->p, As[k]->i, As[k]->x, ae, e, Numeric[k], Control, Info);
		umfpack_solve.stop();
		counter++;
	  }
	  // column j for FAE
	  vec fae(At->m);
	  fae.zeros();
	  if (nonzero){
		(void) cs_dl_gaxpy(F[k], ae, fae._data());
	  }
	  FAE.set_col(j, fae);
	}
	// std::cout << "nnz of E[" << k << "] = " << E[k]->p[E[k]->n] << std::endl;
	// std::cout << "counter = " << counter << std::endl;
	delete [] e;
	delete [] ae;
	S = S - FAE;
	// compute FAb;
	double *ab = new double[As[k]->m];
	umfpack_solve.start();
	(void) umfpack_dl_solve(UMFPACK_A, As[k]->p, As[k]->i, As[k]->x, ab, f[k], Numeric[k], Control, Info);  
	umfpack_solve.stop();
	vec abk(ab, As[k]->m);
	Ab[k] = abk;
	// umfpack_dl_free_numeric(&Numeric[k]);
	// compute right hand side Sy = g - FAb;
	vec FAb(At->m);
	FAb.zeros();
	(void) cs_dl_gaxpy(F[k], ab, FAb._data());
	delete [] ab;
	gg = gg - FAb;	
  }
  // solve Sy = g
  /*
  std::cout << "** S = " << std::endl;
  for (int i = 0; i < S.rows(); i++){
	for (int j = 0; j < S.cols(); j++){
	  std::cout << S(i,j) << " " ;
	}
	std::cout<< std::endl;
  }
  */
  vec y(At->m);
  y = ls_solve(S, gg);
  /*
  std::cout << "** gg = " << std::endl;
  for (int i = 0; i < gg.size(); i++){
	std::cout << gg(i) << std::endl;
  }
  std::cout << "** y = " << std::endl;
  for (int i = 0; i < y.size(); i++){
	std::cout << y(i) << std::endl;
  }
  */
  vec *x = new vec[npart];
  for (int k = 0; k < npart; k++){
	vec ey(E[k]->m);
	ey.zeros(); 
	(void) cs_dl_gaxpy(E[k], y._data(), ey._data());
	double *aey = new double[E[k]->m];
	umfpack_solve.start();
	(void) umfpack_dl_solve(UMFPACK_A, As[k]->p, As[k]->i, As[k]->x, aey, ey._data(), Numeric[k], Control, Info);
	umfpack_solve.stop();
	umfpack_dl_free_numeric(&Numeric[k]);
	vec aey_vec(aey, E[k]->m);
	x[k] = Ab[k] - aey_vec;
  }
  delete [] Ab;
  delete [] Numeric;
  // form z
  int current = 0;
  for (int k = 0; k < npart; k++){
	for (int i = 0; i < As[k]->m; i++){
	  z[current++] = x[k](i);
	}
  }
  delete [] x;
  for (int i = 0; i < At->m; i++){
	z[current++] = y(i);
  }
  // std::cout << "umfpack_symbolic   \t: " << umfpack_symbolic.get_time() << std::endl;
  // std::cout << "umfpack_numeric   \t: " << umfpack_numeric.get_time() << std::endl;
  // std::cout << "umfpack_solve   \t: " << umfpack_solve.get_time() << std::endl;
}

void dd_solve2(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			   double **f, double *g, double *z,
			   Real_Timer &cs_symbolic_runtime, Real_Timer &cs_numeric_runtime, Real_Timer &cs_solve_runtime)
{
  Real_Timer schur_runtime, schur_solve_runtime,  ae_runtime, fae_runtime, runtime;
  cs_dl *TFAE, *FAE, *S;
  vec *Ab = new vec[npart];
  // initial S = At
  S = cs_dl_spalloc(At->m, At->n, At->nzmax, 1, 0);
  for (UF_long i = 0; i < At->n+1; i++){
	S->p[i] = At->p[i];
  }
  for (UF_long i = 0; i < At->nzmax; i++){
	S->i[i] = At->i[i];
	S->x[i] = At->x[i];
  }

  vec gg(g, At->m);
  cs_dls *Symbolic;
  UF_long **Symbolic_q = new UF_long* [npart];
  cs_dln **Numeric = new cs_dln* [npart];
  int order = 2;
  double tol = 1e-14;
  schur_runtime.start();
  for (int k = 0; k < npart; k++){
	//runtime.tic();
	cs_symbolic_runtime.start();
	Symbolic = cs_dl_sqr(order, As[k], 0);
	Symbolic_q[k] = new UF_long[As[k]->n];
	for (UF_long j = 0; j < As[k]->n; j++){
	  Symbolic_q[k][j] = Symbolic->q[j];
	}
	cs_symbolic_runtime.stop();

	cs_numeric_runtime.start();
	Numeric[k] = cs_dl_lu(As[k], Symbolic, tol);
	cs_numeric_runtime.stop();

	cs_dl_sfree(Symbolic);
	// compute FAE and S;
	TFAE = cs_dl_spalloc(At->m, At->n, 1, 1, 1);
	double *e = new double[E[k]->m];
	double *ae = new double[E[k]->m];
	for (int i = 0; i < E[k]->m; i++){
	  e[i] = 0;
	  ae[i] = 0;
	}
	UF_long *xi = new UF_long[2*E[k]->m];
	UF_long top;
	
	int counter = 0;
	//runtime.toc_print();
	schur_solve_runtime.start();
	for (UF_long j = 0; j < At->n; j++){	 
	  ae_runtime.start(); 
	  for (int i = 0; i < E[k]->m; i++){
		e[i] = 0;
		ae[i] = 0;
	  }
	  bool nonzero = false;
	  for (UF_long p = E[k]->p[j]; p < E[k]->p[j+1]; p++){
		ae[E[k]->i[p]] = E[k]->x[p];
		nonzero = true;
	  }
	  ae_runtime.stop();
	  if (nonzero){
		// cs_solve_runtime.start();
		// top = cs_dl_spsolve(Numeric[k]->L, E[k], j, xi, e, Numeric[k]->pinv, 1);
		// top = cs_dl_spsolve(Numeric[k]->U, E[k], j, xi, e, Numeric[k]->pinv, 0);
		cs_solve_runtime.start();
		cs_dl_ipvec(Numeric[k]->pinv, ae, e, As[k]->n);
		cs_dl_lsolve(Numeric[k]->L, e);
		cs_dl_usolve(Numeric[k]->U, e);
		cs_dl_ipvec(Symbolic_q[k], e, ae, As[k]->n);
		cs_solve_runtime.stop();
		// cs_solve_runtime.stop();
		counter++;

	  }
	  // column j for FAE
	  // vec fae(At->m);
	  // fae.zeros();
	  fae_runtime.start();
	  double *fae = new double[At->m];
	  for (int i = 0; i < At->m; i++){
		fae[i] = 0;
	  }
	  if (nonzero){
		(void) cs_dl_gaxpy(F[k], ae, fae);
		for (int i = 0; i < At->m; i++){
		  if (fae[i] != 0){
			cs_dl_entry(TFAE, i, j, fae[i]);
		  }
		}
		// FAE.set_col(j, fae);
	  }	  
	  fae_runtime.stop();
	  delete [] fae;
	  //for (UF_long p = E[k]->p[j]; p < E[k]->p[j+1]; p++){
	  //	ae[E[k]->i[p]] = 0;
	  //}
	}
	schur_solve_runtime.stop();
	//runtime.toc_print();
	FAE = cs_dl_compress(TFAE);
	cs_dl_spfree(TFAE);
	// std::cout << "nnz of E[" << k << "] = " << E[k]->p[E[k]->n] << std::endl;
	// std::cout << "counter = " << counter << std::endl;
	delete [] e;
	delete [] ae;
	// S = S - FAE;
	//runtime.toc_print();
	cs_dl *S1;
	S1 = cs_dl_add(S, FAE, 1, -1);
	cs_dl_spfree(S);
	S = cs_dl_spalloc(S1->m, S1->n, S1->nzmax, 1, 0);
	for (UF_long i = 0; i < S1->n+1; i++){
	  S->p[i] = S1->p[i];
	}
	for (UF_long i = 0; i < S1->nzmax; i++){
	  S->i[i] = S1->i[i];
	  S->x[i] = S1->x[i];
	}
	cs_dl_spfree(FAE);
	cs_dl_spfree(S1);
	//runtime.toc_print();
	// compute FAb;
	double *ab = new double[As[k]->m];
	double *abp = new double[As[k]->m];
	for (int j = 0; j < As[k]->m; j++){
	  ab[j] = f[k][j];
	  abp[j] = 0;
	}
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, ab, abp, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, abp);
	cs_dl_usolve(Numeric[k]->U, abp);	
	cs_dl_ipvec(Symbolic_q[k], abp, ab, As[k]->n);	
	cs_solve_runtime.stop();
	vec abk(ab, As[k]->m);
	Ab[k] = abk;
	// umfpack_dl_free_numeric(&Numeric[k]);
	// compute right hand side Sy = g - FAb;
	vec FAb(At->m);
	FAb.zeros();
	(void) cs_dl_gaxpy(F[k], ab, FAb._data());
	delete [] ab;
	delete [] abp;
	gg = gg - FAb;	
	//runtime.toc_print();
  }
  // solve Sy = g
  /*
  std::cout << "** S = " << std::endl;
  for (int i = 0; i < S.rows(); i++){
	for (int j = 0; j < S.cols(); j++){
	  std::cout << S(i,j) << " " ;
	}
	std::cout<< std::endl;
  }
  */
  /*
  int nnzS = 0;
  for (int i = 0; i < S._datasize(); i++){
	if (*(S._data()+i) != 0)
	  nnzS++;
  }
  std::cout << "nnz of S = " << nnzS << std::endl;
  */
  schur_runtime.stop();
  vec y = gg;
  // ls_solve(S, gg, y);
  cs_dl_lusol(order, S, y._data(), tol);
  cs_dl_spfree(S);
  
  /*
  std::cout << "** gg = " << std::endl;
  for (int i = 0; i < gg.size(); i++){
	std::cout << gg(i) << std::endl;
  }
  std::cout << "** y = " << std::endl;
  for (int i = 0; i < y.size(); i++){
	std::cout << y(i) << std::endl;
  }
  */
  vec *x = new vec[npart];
  for (int k = 0; k < npart; k++){ 
	double *ey = new double[E[k]->m];
	double *aey = new double[E[k]->m];
	for (int j = 0; j < E[k]->m; j++){
	  ey[j] = 0;
	  aey[j] = 0;
	}
	(void) cs_dl_gaxpy(E[k], y._data(), aey);
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, aey, ey, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, ey);
	cs_dl_usolve(Numeric[k]->U, ey);	
	cs_dl_ipvec(Symbolic_q[k], ey, aey, As[k]->n);		
	cs_solve_runtime.stop();
	cs_dl_nfree(Numeric[k]);
	delete [] Symbolic_q[k];
	vec aey_vec(aey, E[k]->m);
	x[k] = Ab[k] - aey_vec;
	delete [] ey;
	delete [] aey;
  }
  delete [] Ab;
  delete [] Numeric;
  delete [] Symbolic_q;
  // form z
  int current = 0;
  for (int k = 0; k < npart; k++){
	for (int i = 0; i < As[k]->m; i++){
	  z[current++] = x[k](i);
	}
  }
  delete [] x;
  for (int i = 0; i < At->m; i++){
	z[current++] = y(i);
  }
  // std::cout << "schur   \t: " << schur_runtime.get_time() << std::endl;
  // std::cout << "ae   \t: " << ae_runtime.get_time() << std::endl;
  // std::cout << "schur_solve   \t: " << schur_solve_runtime.get_time() << std::endl;
  // std::cout << "fae   \t: " << fae_runtime.get_time() << std::endl;

  // std::cout << "cs_symbolic   \t: " << cs_symbolic_runtime.get_time() << std::endl;
  // std::cout << "cs_numeric   \t: " << cs_numeric_runtime.get_time() << std::endl;
  // std::cout << "cs_solve   \t: " << cs_solve_runtime.get_time() << std::endl;
}

void dd_solve3(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			   double **f, double *g, double *z,
			   Real_Timer &cs_symbolic_runtime, Real_Timer &cs_numeric_runtime, Real_Timer &cs_solve_runtime)
{
  Real_Timer schur_runtime, schur_solve_runtime,  ae_runtime, fae_runtime, runtime;
  cs_dl *TFAE, *FAE, *S;
  vec *Ab = new vec[npart];
  // initial S = At
  S = cs_dl_spalloc(At->m, At->n, At->nzmax, 1, 0);
  for (UF_long i = 0; i < At->n+1; i++){
	S->p[i] = At->p[i];
  }
  for (UF_long i = 0; i < At->nzmax; i++){
	S->i[i] = At->i[i];
	S->x[i] = At->x[i];
  }

  vec gg(g, At->m);
  cs_dls *Symbolic;
  UF_long **Symbolic_q = new UF_long* [npart];
  cs_dln **Numeric = new cs_dln* [npart];
  int order = 2;
  double tol = 1e-14;
  schur_runtime.start();
  for (int k = 0; k < npart; k++){
	//runtime.tic();
	cs_symbolic_runtime.start();
	Symbolic = cs_dl_sqr(order, As[k], 0);
	Symbolic_q[k] = new UF_long[As[k]->n];
	for (UF_long j = 0; j < As[k]->n; j++){
	  Symbolic_q[k][j] = Symbolic->q[j];
	}
	cs_symbolic_runtime.stop();

	cs_numeric_runtime.start();
	Numeric[k] = cs_dl_lu(As[k], Symbolic, tol);
	cs_numeric_runtime.stop();

	cs_dl_sfree(Symbolic);
	// compute FAE and S;
	TFAE = cs_dl_spalloc(At->m, At->n, 1, 1, 1);
	double *e = new double[E[k]->m];
	double *ae = new double[E[k]->m];
	for (int i = 0; i < E[k]->m; i++){
	  e[i] = 0;
	  ae[i] = 0;
	}
	UF_long *xi = new UF_long[2*E[k]->m];
	UF_long top;
	
	int counter = 0;
	//runtime.toc_print();
	schur_solve_runtime.start();
	for (UF_long j = 0; j < At->n; j++){	 
	  ae_runtime.start(); 
	  bool nonzero = false;
	  for (UF_long p = E[k]->p[j]; p < E[k]->p[j+1]; p++){
		// ae[E[k]->i[p]] = E[k]->x[p];
		e[Numeric[k]->pinv[E[k]->i[p]]] = E[k]->x[p];
		nonzero = true;
	  }
	  ae_runtime.stop();
	  if (nonzero){
		cs_solve_runtime.start();
		// cs_dl_ipvec(Numeric[k]->pinv, ae, e, As[k]->n);
		my_cs_dl_lsolve(Numeric[k]->L, e);
		my_cs_dl_usolve(Numeric[k]->U, e);
		cs_dl_ipvec(Symbolic_q[k], e, ae, As[k]->n);
		cs_solve_runtime.stop();
		// column j for FAE
		fae_runtime.start();
		double *fae = new double[At->m];
		for (int i = 0; i < At->m; i++){
		  fae[i] = 0;
		}
		(void) cs_dl_gaxpy(F[k], ae, fae);
		for (int i = 0; i < At->m; i++){
		  if (fae[i] != 0){
			cs_dl_entry(TFAE, i, j, fae[i]);
		  }
		}
		fae_runtime.stop();
		delete [] fae;
		ae_runtime.start(); 
		for (int i = 0; i < E[k]->m; i++){
		  e[i] = 0;
		  // ae[i] = 0;
		}
		ae_runtime.stop();
	  }	  

	}
	schur_solve_runtime.stop();
	//runtime.toc_print();
	FAE = cs_dl_compress(TFAE);
	cs_dl_spfree(TFAE);
	// std::cout << "nnz of E[" << k << "] = " << E[k]->p[E[k]->n] << std::endl;
	// std::cout << "counter = " << counter << std::endl;
	delete [] e;
	delete [] ae;
	// S = S - FAE;
	//runtime.toc_print();
	cs_dl *S1;
	S1 = cs_dl_add(S, FAE, 1, -1);
	cs_dl_spfree(S);
	S = cs_dl_spalloc(S1->m, S1->n, S1->nzmax, 1, 0);
	for (UF_long i = 0; i < S1->n+1; i++){
	  S->p[i] = S1->p[i];
	}
	for (UF_long i = 0; i < S1->nzmax; i++){
	  S->i[i] = S1->i[i];
	  S->x[i] = S1->x[i];
	}
	cs_dl_spfree(FAE);
	cs_dl_spfree(S1);
	//runtime.toc_print();
	// compute FAb;
	double *ab = new double[As[k]->m];
	double *abp = new double[As[k]->m];
	for (int j = 0; j < As[k]->m; j++){
	  ab[j] = f[k][j];
	  abp[j] = 0;
	}
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, ab, abp, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, abp);
	cs_dl_usolve(Numeric[k]->U, abp);	
	cs_dl_ipvec(Symbolic_q[k], abp, ab, As[k]->n);	
	cs_solve_runtime.stop();
	vec abk(ab, As[k]->m);
	Ab[k] = abk;
	// umfpack_dl_free_numeric(&Numeric[k]);
	// compute right hand side Sy = g - FAb;
	vec FAb(At->m);
	FAb.zeros();
	(void) cs_dl_gaxpy(F[k], ab, FAb._data());
	delete [] ab;
	delete [] abp;
	gg = gg - FAb;	
	//runtime.toc_print();
  }
  // solve Sy = g
  /*
  std::cout << "** S = " << std::endl;
  for (int i = 0; i < S.rows(); i++){
	for (int j = 0; j < S.cols(); j++){
	  std::cout << S(i,j) << " " ;
	}
	std::cout<< std::endl;
  }
  */
  /*
  int nnzS = 0;
  for (int i = 0; i < S._datasize(); i++){
	if (*(S._data()+i) != 0)
	  nnzS++;
  }
  std::cout << "nnz of S = " << nnzS << std::endl;
  */
  schur_runtime.stop();
  vec y = gg;
  // ls_solve(S, gg, y);
  cs_dl_lusol(order, S, y._data(), tol);
  cs_dl_spfree(S);
  
  /*
  std::cout << "** gg = " << std::endl;
  for (int i = 0; i < gg.size(); i++){
	std::cout << gg(i) << std::endl;
  }
  std::cout << "** y = " << std::endl;
  for (int i = 0; i < y.size(); i++){
	std::cout << y(i) << std::endl;
  }
  */
  vec *x = new vec[npart];
  for (int k = 0; k < npart; k++){ 
	double *ey = new double[E[k]->m];
	double *aey = new double[E[k]->m];
	for (int j = 0; j < E[k]->m; j++){
	  ey[j] = 0;
	  aey[j] = 0;
	}
	(void) cs_dl_gaxpy(E[k], y._data(), aey);
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, aey, ey, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, ey);
	cs_dl_usolve(Numeric[k]->U, ey);	
	cs_dl_ipvec(Symbolic_q[k], ey, aey, As[k]->n);		
	cs_solve_runtime.stop();
	cs_dl_nfree(Numeric[k]);
	delete [] Symbolic_q[k];
	vec aey_vec(aey, E[k]->m);
	x[k] = Ab[k] - aey_vec;
	delete [] ey;
	delete [] aey;
  }
  delete [] Ab;
  delete [] Numeric;
  delete [] Symbolic_q;
  // form z
  int current = 0;
  for (int k = 0; k < npart; k++){
	for (int i = 0; i < As[k]->m; i++){
	  z[current++] = x[k](i);
	}
  }
  delete [] x;
  for (int i = 0; i < At->m; i++){
	z[current++] = y(i);
  }
  // std::cout << "schur   \t: " << schur_runtime.get_time() << std::endl;
  // std::cout << "ae   \t: " << ae_runtime.get_time() << std::endl;
  // std::cout << "schur_solve   \t: " << schur_solve_runtime.get_time() << std::endl;
  // std::cout << "fae   \t: " << fae_runtime.get_time() << std::endl;

  // std::cout << "cs_symbolic   \t: " << cs_symbolic_runtime.get_time() << std::endl;
  // std::cout << "cs_numeric   \t: " << cs_numeric_runtime.get_time() << std::endl;
  // std::cout << "cs_solve   \t: " << cs_solve_runtime.get_time() << std::endl;
}

void dd_solve_ooc(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
				  double **f, double *g, double *z,
				  Real_Timer &cs_symbolic_runtime, Real_Timer &cs_numeric_runtime, Real_Timer &cs_solve_runtime)
{
  Real_Timer schur_runtime, schur_solve_runtime,  ae_runtime, fae_runtime, runtime;
  cs_dl *TFAE, *FAE, *S;
  string *part_file_name = new string [npart];
  for (int k = 0; k < npart; k++){
	part_file_name[k] = "temp/part";
	stringstream ss;
	ss << k;
	part_file_name[k].append(ss.str());
  }
  ifstream *in_part_file = new ifstream [npart];
  ofstream *out_part_file = new ofstream [npart];

  vec *Ab = new vec[npart];
  // initial S = At
  S = cs_dl_spalloc(At->m, At->n, At->nzmax, 1, 0);
  for (UF_long i = 0; i < At->n+1; i++){
	S->p[i] = At->p[i];
  }
  for (UF_long i = 0; i < At->nzmax; i++){
	S->i[i] = At->i[i];
	S->x[i] = At->x[i];
  }

  vec gg(g, At->m);
  cs_dls **Symbolic = new cs_dls* [npart];
  cs_dln **Numeric = new cs_dln* [npart];
  int order = 2;
  double tol = 1e-14;
  schur_runtime.start();
  for (int k = 0; k < npart; k++){
	//runtime.tic();
	cs_symbolic_runtime.start();
	Symbolic[k] = cs_dl_sqr(order, As[k], 0);
	cs_symbolic_runtime.stop();

	cs_numeric_runtime.start();
	Numeric[k] = cs_dl_lu(As[k], Symbolic[k], tol);
	cs_numeric_runtime.stop();

	// compute FAE and S;
	TFAE = cs_dl_spalloc(At->m, At->n, 1, 1, 1);
	// UF_long *xi = new UF_long[2*E[k]->m];
	// UF_long top;
	
	int counter = 0;
	//runtime.toc_print();
	double *e = (double*) calloc(E[k]->m, sizeof(double));
	double *ae = (double*) calloc(E[k]->m, sizeof(double));
	double *fae = (double*) calloc(At->m, sizeof(double));
	schur_solve_runtime.start();
	for (UF_long j = 0; j < At->n; j++){	 
	  ae_runtime.start(); 
	  bool nonzero = false;
	  for (UF_long p = E[k]->p[j]; p < E[k]->p[j+1]; p++){
		// ae[E[k]->i[p]] = E[k]->x[p];
		e[Numeric[k]->pinv[E[k]->i[p]]] = E[k]->x[p];
		nonzero = true;
	  }
	  ae_runtime.stop();
	  if (nonzero){
		cs_solve_runtime.start();
		// cs_dl_ipvec(Numeric[k]->pinv, ae, e, As[k]->n);
		my_cs_dl_lsolve(Numeric[k]->L, e);
		my_cs_dl_usolve(Numeric[k]->U, e);
		cs_dl_ipvec(Symbolic[k]->q, e, ae, As[k]->n);
		cs_solve_runtime.stop();
		// column j for FAE
		fae_runtime.start();		
		(void) cs_dl_gaxpy(F[k], ae, fae);
		for (int i = 0; i < At->m; i++){
		  if (fae[i] != 0){
			cs_dl_entry(TFAE, i, j, fae[i]);
		  }
		}
		memset((void*) fae, 0, sizeof(double)*At->m);
		fae_runtime.stop();		
		ae_runtime.start(); 
		memset((void*) e, 0, sizeof(double)*E[k]->m);
		ae_runtime.stop();
	  }	  
	}
	free(fae);
	free(e);
	free(ae);
	schur_solve_runtime.stop();
	//runtime.toc_print();
	FAE = cs_dl_compress(TFAE);
	cs_dl_spfree(TFAE);
	// std::cout << "nnz of E[" << k << "] = " << E[k]->p[E[k]->n] << std::endl;
	// std::cout << "counter = " << counter << std::endl;
	// S = S - FAE;
	//runtime.toc_print();
	cs_dl *S1 = S;
	S = cs_dl_add(S1, FAE, 1, -1);
	cs_dl_spfree(S1);
	/*
	S = cs_dl_spalloc(S1->m, S1->n, S1->nzmax, 1, 0);
	for (UF_long i = 0; i < S1->n+1; i++){
	  S->p[i] = S1->p[i];
	}
	for (UF_long i = 0; i < S1->nzmax; i++){
	  S->i[i] = S1->i[i];
	  S->x[i] = S1->x[i];
	}
	*/
	cs_dl_spfree(FAE);
	//cs_dl_spfree(S1);
	// runtime.toc_print();
	// compute FAb;
	double *ab = new double[As[k]->m];
	double *abp = new double[As[k]->m];
	for (int j = 0; j < As[k]->m; j++){
	  ab[j] = f[k][j];
	  abp[j] = 0;
	}
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, ab, abp, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, abp);
	cs_dl_usolve(Numeric[k]->U, abp);	
	cs_dl_ipvec(Symbolic[k]->q, abp, ab, As[k]->n);	
	cs_solve_runtime.stop();
	// Out of core
	out_part_file[k].open(part_file_name[k].c_str(), ios::binary);
	numeric_dl_save(out_part_file[k], Numeric[k]);
	out_part_file[k].close();
	cs_dl_nfree(Numeric[k]);
	vec abk(ab, As[k]->m);
	Ab[k] = abk;
	// umfpack_dl_free_numeric(&Numeric[k]);
	// compute right hand side Sy = g - FAb;
	vec FAb(At->m);
	FAb.zeros();
	(void) cs_dl_gaxpy(F[k], ab, FAb._data());
	delete [] ab;
	delete [] abp;
	gg = gg - FAb;	
	//runtime.toc_print();
  }
  // solve Sy = g
  /*
  std::cout << "** S = " << std::endl;
  for (int i = 0; i < S.rows(); i++){
	for (int j = 0; j < S.cols(); j++){
	  std::cout << S(i,j) << " " ;
	}
	std::cout<< std::endl;
  }
  */
  /*
  int nnzS = 0;
  for (int i = 0; i < S._datasize(); i++){
	if (*(S._data()+i) != 0)
	  nnzS++;
  }
  std::cout << "nnz of S = " << nnzS << std::endl;
  */
  schur_runtime.stop();
  vec y = gg;
  // ls_solve(S, gg, y);
  cs_dl_lusol(order, S, y._data(), tol);
  cs_dl_spfree(S);
  std::cout << "Schur complement done" << std::endl;
  
  /*
  std::cout << "** gg = " << std::endl;
  for (int i = 0; i < gg.size(); i++){
	std::cout << gg(i) << std::endl;
  }
  std::cout << "** y = " << std::endl;
  for (int i = 0; i < y.size(); i++){
	std::cout << y(i) << std::endl;
  }
  */
  vec *x = new vec[npart];
  for (int k = 0; k < npart; k++){ 
	double *ey = new double[E[k]->m];
	double *aey = new double[E[k]->m];
	for (int j = 0; j < E[k]->m; j++){
	  ey[j] = 0;
	  aey[j] = 0;
	}
	(void) cs_dl_gaxpy(E[k], y._data(), aey);
	// Out of core
	in_part_file[k].open(part_file_name[k].c_str(), ios::binary);
	numeric_dl_load(in_part_file[k], Numeric[k]);
	in_part_file[k].close();
	cs_solve_runtime.start();
	cs_dl_ipvec(Numeric[k]->pinv, aey, ey, As[k]->n);
	cs_dl_lsolve(Numeric[k]->L, ey);
	cs_dl_usolve(Numeric[k]->U, ey);	
	cs_dl_ipvec(Symbolic[k]->q, ey, aey, As[k]->n);		
	cs_solve_runtime.stop();
	cs_dl_sfree(Symbolic[k]);
	cs_dl_nfree(Numeric[k]);
	vec aey_vec(aey, E[k]->m);
	x[k] = Ab[k] - aey_vec;
	delete [] ey;
	delete [] aey;
  }
  delete [] Ab;
  delete [] Symbolic;
  delete [] Numeric;
  delete [] part_file_name;
  delete [] in_part_file;
  delete [] out_part_file;
  // form z
  int current = 0;
  for (int k = 0; k < npart; k++){
	for (int i = 0; i < As[k]->m; i++){
	  z[current++] = x[k](i);
	}
  }
  delete [] x;
  for (int i = 0; i < At->m; i++){
	z[current++] = y(i);
  }
  // std::cout << "schur         \t: " << schur_runtime.get_time() << std::endl;
  // std::cout << "ae            \t: " << ae_runtime.get_time() << std::endl;
  // std::cout << "schur_solve   \t: " << schur_solve_runtime.get_time() << std::endl;
  // std::cout << "fae           \t: " << fae_runtime.get_time() << std::endl;

  // std::cout << "cs_symbolic   \t: " << cs_symbolic_runtime.get_time() << std::endl;
  // std::cout << "cs_numeric   \t: " << cs_numeric_runtime.get_time() << std::endl;
  // std::cout << "cs_solve   \t: " << cs_solve_runtime.get_time() << std::endl;
}

int my_cs_dl_lsolve (const cs_dl *L, double *x)
{
    UF_long p, j, n, *Lp, *Li;
    double *Lx;
    n = L->n; Lp = L->p; Li = L->i; Lx = L->x;
    for (j = 0; j < n; j++){
	  if (x[j] != 0){
		x[j] /= Lx[Lp[j]];
		for (p = Lp[j]+1 ; p < Lp[j+1]; p++){
		  x[Li[p]] -= Lx[p] * x[j];
		}
	  }
    }
    return (1);
}

int my_cs_dl_usolve (const cs_dl *U, double *x)
{
    UF_long p, j, n, *Up, *Ui ;
    double *Ux ;
    n = U->n; Up = U->p; Ui = U->i; Ux = U->x;
    for (j = n-1; j >= 0; j--){
	  if (x[j] != 0){
		x[j] /= Ux[Up[j+1]-1];
		for (p = Up[j] ; p < Up[j+1]-1; p++){
		  x[Ui[p]] -= Ux[p] * x[j];
		}
	  }
	}
    return (1) ;
}
