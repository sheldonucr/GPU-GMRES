/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr2.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR function with CSparse
 *
 */

#include <iostream>
#include <algorithm>
#include <itpp/base/timing.h>
#include <itpp/base/smat.h>
#include <itpp/base/mat.h>
#include <itpp/base/vec.h>
#include <itpp/base/specmat.h>
#include <itpp/base/algebra/lapack.h>
#include <itpp/base/algebra/lu.h>
#include <itpp/base/algebra/ls_solve.h>
#include <itpp/base/algebra/svd.h>
#include <itpp/signal/transforms.h>
#include <itpp/base/math/elem_math.h>
#include <itpp/base/math/log_exp.h>
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"

using namespace itpp;

void etbr2(cs_dl *G, cs_dl *C, cs_dl *B, 
		   Source *VS, int nVS, Source *IS, int nIS, 
		   double tstep, double tstop, int q, 
		   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value)
{

  Real_Timer interp_run_time, interp2_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer svd_run_time, rmatrix_run_time;
  Real_Timer sim_run_time, etbr2_run_time;

  etbr2_run_time.start();

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  form_vec(ts, 0, tstep, tstop);


  /* FFT */
  int fft_n = 512;
  int L = ts.size();
  int N = floor_i(log2(L))+1;
  
  fft_n = pow2i(N);
  std::cout << "# time steps: "<< L << std::endl;
  std::cout << "# FFT points: "<< fft_n << std::endl;

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 0;
  double f_max = 0.5/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1/tstep/fft_n;
  double f_max = 0.5/tstep;
  vec lin_samples = linspace(std::log10(f_min), std::log10(f_max), q-1);
  vec samples = pow10(lin_samples); 

  samples.ins(0, 0);
  int np = samples.size();
  //cout <<"# samples(1): " << np << endl;


  
  vec f;
  // form_vec(f, 0, 1, fft_n/2);
  f = linspace(0, 1, fft_n/2+1);
  f *= 0.5/tstep;
  cvec spwl_row;
  vec abs_spwl_row;
  mat us(nVS+nIS, np);
  vec us_row(np);
  vec interp_value(ts.size());
  for (int i = 0; i < nVS; i++){
	interp_run_time.start();
	interp1(VS[i].time, VS[i].value, ts, interp_value);
	interp_run_time.stop();
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(i, us_row);
  }
  for (int i = 0; i < nIS; i++){
	interp_run_time.start();
	interp1(IS[i].time, IS[i].value, ts, interp_value);
	interp_run_time.stop();
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(nVS+i, us_row);
  }
  
	
  /* use UMFPACK to solve Ax=b */
  mat Z(nDim, np);
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;

  for (int i = 0; i < np; i++){
	
	sCpG_run_time.start();
	cs_dl *A;
	if (C != NULL)
	  A = cs_dl_add(G, C, 1, samples(i));
	else
	  A = G;
	sCpG_run_time.stop();

	/* LU decomposition */
	UF_long *Ap = A->p; 
	UF_long *Ai = A->i;
	double *Ax = A->x;

	cs_symbolic.start();
	Symbolic = cs_dl_sqr(order, A, 0);
	cs_symbolic.stop();

	cs_numeric.start();
	Numeric = cs_dl_lu(A, Symbolic, tol);
	cs_numeric.stop();

	/* solve Az = b  */
	vec x(nDim);
	x.zeros();
	vec z(nDim);
	z.zeros();
	vec b(nDim);
	b.zeros();
	(void) cs_dl_gaxpy(B, us.get_col(i)._data(), b._data());
	cs_solve.start();
	cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), A->n);
	cs_dl_lsolve(Numeric->L, x._data());
	cs_dl_usolve(Numeric->U, x._data());
	cs_dl_ipvec(Symbolic->q, x._data(), z._data(), A->n);  	
	cs_solve.stop();

	Z.set_col(i, z);	
	cs_dl_sfree(Symbolic);
	cs_dl_nfree(Numeric);
	if (A != G)
	  cs_dl_spfree(A);
  }

  /* SVD */
  svd_run_time.start();
  mat U, V;
  vec S;
  int info;
  info =  svd0(Z, U, S, V);
  X = U.get_cols(0,q-1);
  svd_run_time.stop();

  /* Generate reduced matrices */
  rmatrix_run_time.start();
  // Gr = X.T()*G*X;
  Gr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(G, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Gr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Cr = X.T()*C*X; 
  Cr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(C, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Cr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Br = X.T()*B;
  Br.set_size(q, nSDim);
  cs_dl *BT;
  BT  = cs_dl_transpose(B, 1);
  for (int j = 0; j < q; j++){
	vec v(nSDim);
	v.zeros();
	(void) cs_dl_gaxpy(BT, X.get_col(j)._data(), v._data());
	Br.set_row(j, v);
  }
  cs_dl_spfree(BT);
  rmatrix_run_time.stop();

  sim_run_time.start();
  vec u_col(nVS+nIS);
  vec w(q);
  sim_value.set_size(q, ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vector<int> const_v, const_i, var_v, var_i;
  for(int j = 0; j < nVS; j++){
	if (VS[j].time.size() == 1)
	  const_v.push_back(j);
	else
	  var_v.push_back(j);
  }
  for(int j = 0; j < nIS; j++){
	if (IS[j].time.size() == 1)
	  const_i.push_back(j);
	else
	  var_i.push_back(j);
  }
  /* DC simulation */
  for(vector<int>::iterator it = const_v.begin(); it != const_v.end(); ++it){
	u_col(*it) = VS[*it].value(0);
  }
  for(vector<int>::iterator it = const_i.begin(); it != const_i.end(); ++it){
	u_col(nVS+(*it)) = IS[*it].value(0);
  }
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	w = Br*u_col;
  }
  vec xres;
  xres = ls_solve(Gr, w);
  sim_value.set_col(0, xres);
  /* Transient simulation */
  mat right = 1/tstep*Cr;
  mat left = Gr + right;
  mat l_left, u_left;
  ivec p;
  lu(left, l_left, u_left, p);
  vec xn(q), xn1(q), xn1t(q);
  xn = xres;
  xn1.zeros();
  xn1t.zeros();
  for (int i = 1; i < ts.size(); i++){
	interp2_run_time.start();
	/*
	for(int j = 0; j < nVS; j++){
	  interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
	  u_col(j) = temp;
	}
	for(int j = 0; j < nIS; j++){
	  interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
	  u_col(nVS+j) = temp;
	}
	*/
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	interp2_run_time.stop();
	w = Br*u_col;
	w += right*xn;
	interchange_permutations(w, p);
	forward_substitution(l_left, w, xn1t);
	backward_substitution(u_left, xn1t, xn1);
	sim_value.set_col(i, xn1);
	xn = xn1;
  }
  delete [] cur;
  sim_run_time.stop();

  etbr2_run_time.stop();

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "Interpolation   \t: " << interp_run_time.get_time() << std::endl;
  std::cout << "FFT             \t: " << fft_run_time.get_time() << std::endl;
  std::cout << "sC+G            \t: " << sCpG_run_time.get_time() << std::endl;
  std::cout << "symbolic        \t: " << cs_symbolic.get_time() << std::endl;
  std::cout << "numeric         \t: " << cs_numeric.get_time() << std::endl;
  std::cout << "solve           \t: " << cs_solve.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "Interpolation2   \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "simulation      \t: " << sim_run_time.get_time() << std::endl;
  std::cout << "Total           \t: " << etbr2_run_time.get_time() << std::endl;
}

void etbr2(cs_dl *G, cs_dl *C, cs_dl *B, 
		   Source *VS, int nVS, Source *IS, int nIS, 
		   double tstep, double tstop, int q, 
		   mat &Gr, mat &Cr, mat &Br, mat &X, double &max_i, int &max_i_idx)
{

  Real_Timer interp_run_time, interp2_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer svd_run_time, rmatrix_run_time;
  Real_Timer etbr2_run_time;

  etbr2_run_time.start();

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  form_vec(ts, 0, tstep, tstop);


  /* FFT */
  int fft_n = 512;
  int L = ts.size();
  //int N = floor_i(log2(L))+1;
  int N = 10;  //1024 samples
  
  fft_n = pow2i(N);
  std::cout << "# time steps: "<< L << std::endl;
  std::cout << "# FFT points: "<< fft_n << std::endl;

  /* sampling: uniform in linear scale */
  /*
  double f_min = 0;
  double f_max = 0.5/tstep;
  vec samples = linspace(f_min, f_max, q);
  */

  /* sampling: uniform in log scale */
  double f_min = 1/tstep/fft_n;
  double f_max = 0.5/tstep;
  //cout <<"f_min: " << f_min << endl;
  //cout << "f_max: " << f_max << endl;
  vec lin_samples;
  vec samples;
  if(q > 6){
    lin_samples = linspace(std::log10(f_min), std::log10(f_max), q-6);
    samples = pow10(lin_samples); 
  }

  //samples.ins(0, 1e9);
  //samples.ins(0, 1e8);
  samples.ins(0, 1e7);
  samples.ins(0, 1e6);
  samples.ins(0, 1e5);
  samples.ins(0, 1e1);
  samples.ins(0, 1);
  samples.ins(0, 0);
  int np = samples.size();
  cout <<"# samples:(2) " << np << endl;
  
  vec f;
  // form_vec(f, 0, 1, fft_n/2);
  f = linspace(0, 1, fft_n/2+1);
  f *= 0.5/tstep;
  cvec spwl_row;
  vec abs_spwl_row;
  mat us(nVS+nIS, np);
  vec us_row(np);
  vec interp_value(ts.size());
  for (int i = 0; i < nVS; i++){
	interp_run_time.start();
	interp1(VS[i].time, VS[i].value, ts, interp_value);
	interp_run_time.stop();
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(i, us_row);
  }
  for (int i = 0; i < nIS; i++){
	interp_run_time.start();
	interp1(IS[i].time, IS[i].value, ts, interp_value);
	interp_run_time.stop();
	vec abs_interp_value = abs(interp_value);
	double max_interp = max(abs_interp_value);
	// int max_interp_idx = max_index(abs_interp_value);
	if (max_interp > max_i){
	  max_i = max_interp;
	  max_i_idx = max_index(abs_interp_value);
	}
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(nVS+i, us_row);
  } 
	
  /* use UMFPACK to solve Ax=b */
  mat Z(nDim, np);
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;

  for (int i = 0; i < np; i++){
	
	sCpG_run_time.start();
	cs_dl *A;
	if (C != NULL)
	  A = cs_dl_add(G, C, 1, samples(i));
	else
	  A = G;
	sCpG_run_time.stop();

	/* LU decomposition */
	UF_long *Ap = A->p; 
	UF_long *Ai = A->i;
	double *Ax = A->x;

	cs_symbolic.start();
	Symbolic = cs_dl_sqr(order, A, 0);
	cs_symbolic.stop();

	cs_numeric.start();
	Numeric = cs_dl_lu(A, Symbolic, tol);
	cs_numeric.stop();

	/* solve Az = b  */
	vec x(nDim);
	x.zeros();
	vec z(nDim);
	z.zeros();
	vec b(nDim);
	b.zeros();
	(void) cs_dl_gaxpy(B, us.get_col(i)._data(), b._data());
	cs_solve.start();
	cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), A->n);
	cs_dl_lsolve(Numeric->L, x._data());
	cs_dl_usolve(Numeric->U, x._data());
	cs_dl_ipvec(Symbolic->q, x._data(), z._data(), A->n);  	
	cs_solve.stop();

	Z.set_col(i, z);	
	cs_dl_sfree(Symbolic);
	cs_dl_nfree(Numeric);
	if (A != G)
	  cs_dl_spfree(A);
  }

  /* SVD */
  svd_run_time.start();
  mat U, V;
  vec S;
  int info;
  info =  svd0(Z, U, S, V);
  X = U.get_cols(0,q-1);
  svd_run_time.stop();

  /* Generate reduced matrices */
  rmatrix_run_time.start();
  // Gr = X.T()*G*X;
  Gr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(G, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Gr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Cr = X.T()*C*X; 
  Cr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(C, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Cr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Br = X.T()*B;
  Br.set_size(q, nSDim);
  cs_dl *BT;
  BT  = cs_dl_transpose(B, 1);
  for (int j = 0; j < q; j++){
	vec v(nSDim);
	v.zeros();
	(void) cs_dl_gaxpy(BT, X.get_col(j)._data(), v._data());
	Br.set_row(j, v);
  }
  cs_dl_spfree(BT);
  rmatrix_run_time.stop();

  etbr2_run_time.stop();

#ifndef UCR_EXTERNAL
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation   \t: " << interp_run_time.get_time() << std::endl;
  std::cout << "FFT             \t: " << fft_run_time.get_time() << std::endl;
  std::cout << "sC+G            \t: " << sCpG_run_time.get_time() << std::endl;
  std::cout << "symbolic        \t: " << cs_symbolic.get_time() << std::endl;
  std::cout << "numeric         \t: " << cs_numeric.get_time() << std::endl;
  std::cout << "solve           \t: " << cs_solve.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "total reduction \t: " << etbr2_run_time.get_time() << std::endl;
#endif
}

void etbr2(cs_dl *G, cs_dl *C, cs_dl *B, 
		   Source *VS, int nVS, Source *IS, int nIS, 
		   double tstep, double tstop, int q, 
		   mat &Gr, mat &Cr, mat &Br, mat &X, double &max_i,
		   vec* u_col)
{

  Real_Timer interp_run_time, interp2_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer svd_run_time, rmatrix_run_time;
  Real_Timer etbr2_run_time;

  etbr2_run_time.start();

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  form_vec(ts, 0, tstep, tstop);

  /* FFT */
  int fft_n = 512;
  int L = ts.size();
  int N = floor_i(log2(L))+1;
  fft_n = pow2i(N);
  std::cout << "# time steps: "<< L << std::endl;
  std::cout << "# FFT points: "<< fft_n << std::endl;

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 0;
  double f_max = 0.5/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1/tstep/fft_n;
  double f_max = 0.5/tstep;
  vec lin_samples = linspace(std::log10(f_min), std::log10(f_max), q-1);
  vec samples = pow10(lin_samples); 

  samples.ins(0, 0);
  int np = samples.size();
  //cout <<"# samples:(3) " << np << endl;
  
  vec f;
  // form_vec(f, 0, 1, fft_n/2);
  f = linspace(0, 1, fft_n/2+1);
  f *= 0.5/tstep;
  cvec spwl_row;
  vec abs_spwl_row;
  mat us(nVS+nIS, np);
  vec us_row(np);
  vec interp_value(ts.size());
  for (int i = 0; i < nVS; i++){
	interp_run_time.start();
	interp1(VS[i].time, VS[i].value, ts, interp_value);
	for (int k = 0; k < ts.size(); k++)
		u_col[k](i) = interp_value(k);
	interp_run_time.stop();
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(i, us_row);
  }
  for (int i = 0; i < nIS; i++){
	interp_run_time.start();
	interp1(IS[i].time, IS[i].value, ts, interp_value);
	for (int k = 0; k < ts.size(); k++)
		u_col[k](nVS+i) = interp_value(k);
	interp_run_time.stop();
	vec abs_interp_value = abs(interp_value);
	double max_interp = max(abs_interp_value);
	// int max_interp_idx = max_index(abs_interp_value);
	if (max_interp > max_i){
	  max_i = max_interp;
	}
	fft_run_time.start();
	// spwl_row = fft_real(interp_value, fft_n);
	interp_value.set_size(fft_n, true);
	fft_real(interp_value, spwl_row);
	spwl_row /= L;
	fft_run_time.stop();
	// spwl_row *= (double)1/fft_n;
	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_row);
	us.set_row(nVS+i, us_row);
  } 
	
  /* use UMFPACK to solve Ax=b */
  mat Z(nDim, np);
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;

  for (int i = 0; i < np; i++){
	
	sCpG_run_time.start();
	cs_dl *A;
	if (C != NULL)
	  A = cs_dl_add(G, C, 1, samples(i));
	else
	  A = G;
	sCpG_run_time.stop();

	/* LU decomposition */
	UF_long *Ap = A->p; 
	UF_long *Ai = A->i;
	double *Ax = A->x;

	cs_symbolic.start();
	Symbolic = cs_dl_sqr(order, A, 0);
	cs_symbolic.stop();

	cs_numeric.start();
	Numeric = cs_dl_lu(A, Symbolic, tol);
	cs_numeric.stop();

	/* solve Az = b  */
	vec x(nDim);
	x.zeros();
	vec z(nDim);
	z.zeros();
	vec b(nDim);
	b.zeros();
	(void) cs_dl_gaxpy(B, us.get_col(i)._data(), b._data());
	cs_solve.start();
	cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), A->n);
	cs_dl_lsolve(Numeric->L, x._data());
	cs_dl_usolve(Numeric->U, x._data());
	cs_dl_ipvec(Symbolic->q, x._data(), z._data(), A->n);  	
	cs_solve.stop();

	Z.set_col(i, z);	
	cs_dl_sfree(Symbolic);
	cs_dl_nfree(Numeric);
	if (A != G)
	  cs_dl_spfree(A);
  }

  /* SVD */
  svd_run_time.start();
  mat U, V;
  vec S;
  int info;
  info =  svd0(Z, U, S, V);
  X = U.get_cols(0,q-1);
  svd_run_time.stop();

  /* Generate reduced matrices */
  rmatrix_run_time.start();
  // Gr = X.T()*G*X;
  Gr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(G, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Gr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Cr = X.T()*C*X; 
  Cr.set_size(q, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(C, X.get_col(j)._data(), v._data());
	for (int i = 0; i < q; i++){
	  Cr.set(i, j, dot(X.get_col(i), v));
	}
  }
  //Br = X.T()*B;
  Br.set_size(q, nSDim);
  cs_dl *BT;
  BT  = cs_dl_transpose(B, 1);
  for (int j = 0; j < q; j++){
	vec v(nSDim);
	v.zeros();
	(void) cs_dl_gaxpy(BT, X.get_col(j)._data(), v._data());
	Br.set_row(j, v);
  }
  cs_dl_spfree(BT);
  rmatrix_run_time.stop();

  etbr2_run_time.stop();

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "Interpolation   \t: " << interp_run_time.get_time() << std::endl;
  std::cout << "FFT             \t: " << fft_run_time.get_time() << std::endl;
  std::cout << "sC+G            \t: " << sCpG_run_time.get_time() << std::endl;
  std::cout << "symbolic        \t: " << cs_symbolic.get_time() << std::endl;
  std::cout << "numeric         \t: " << cs_numeric.get_time() << std::endl;
  std::cout << "solve           \t: " << cs_solve.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "Total           \t: " << etbr2_run_time.get_time() << std::endl;
}
