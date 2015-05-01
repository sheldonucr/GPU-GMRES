/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfil$
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR function with CSparse, pthread implementation
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
#include <itpp/base/algebra/ls_solve.h>
#include <itpp/base/algebra/lu.h>
#include <itpp/base/algebra/svd.h>
#include <itpp/signal/transforms.h>
#include <itpp/base/math/elem_math.h>
#include <itpp/base/math/log_exp.h>
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"
#include <pthread.h>

using namespace itpp;

// typedef struct{
//   vec samples;
//   cs_dl *G;
//   cs_dl *C;
//   cs_dl *B;
//   mat *us;
//   vec *zvec;
// }AXBDATA;
// 
AXBDATA pdata;
// 
// // #define NUM_THREADS 20
// // pthread_t threads[NUM_THREADS];
pthread_mutex_t mutexz;

void *solve_axb(void * threadarg)
{
  UF_long nDim = pdata.B->m;
  UF_long nSDim = pdata.B->n;
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;
  int i = *(int*) threadarg;

  cs_dl *A;
  if (pdata.C != NULL)
	A = cs_dl_add(pdata.G, pdata.C, 1, pdata.samples(i));
  else
	A = pdata.G;

  /* LU decomposition */
  UF_long *Ap = A->p; 
  UF_long *Ai = A->i;
  double *Ax = A->x;

  Symbolic = cs_dl_sqr(order, A, 0);

  Numeric = cs_dl_lu(A, Symbolic, tol);

  /* solve Az = b  */
  vec x(nDim);
  x.zeros();
  vec z(nDim);
  z.zeros();
  vec b(nDim);
  b.zeros();
  (void) cs_dl_gaxpy(pdata.B, pdata.us->get_col(i)._data(), b._data());
  cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), A->n);
  cs_dl_lsolve(Numeric->L, x._data());
  cs_dl_usolve(Numeric->U, x._data());
  cs_dl_ipvec(Symbolic->q, x._data(), z._data(), A->n);  	

  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  if (A != pdata.G)
	cs_dl_spfree(A);

  pdata.zvec[i] = z;
  /*
  pthread_mutex_lock(&mutexz);
  pdata.Z.set_col(i, z);	
  pthread_mutex_unlock(&mutexz);
  */

  pthread_exit((void*) 0);
}

#if 0

void etbr_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
				 Source *VS, int nVS, Source *IS, int nIS, 
				 double tstep, double tstop, int q, 
				 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value)
{

  Real_Timer interp_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer cs_run_time, svd_run_time, rmatrix_run_time;
  Real_Timer sim_run_time;

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  form_vec(ts, 0, tstep, tstop);

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 1.0e-2;
  double f_max = 1/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1/tstep/fft_n;
  double f_max = 0.5/tstep;
  vec lin_samples = linspace(std::log10(f_min), std::log10(f_max), q-5);
  vec samples = pow10(lin_samples); 

  samples.ins(0, 1e9);
  samples.ins(0, 1e8);

  samples.ins(0, 1e7);
  samples.ins(0, 1e6);
  pdata.samples = samples;

  int np = samples.size();
  
  /* FFT */
  fft_run_time.start();
  int fft_n = 512;
  vec f;
  form_vec(f, 0, 1, fft_n/2);
  f *= 1/tstep*1/fft_n;
  cvec spwl_row;
  vec abs_spwl_row;
  mat us_v(nVS, np);
  mat us_i(nIS, np);
  vec us_v_row(np);
  vec us_i_row(np);
  vec interp_value(ts.size());
  for (int i = 0; i < nVS; i++){
	interp1(VS[i].time, VS[i].value, ts, interp_value);
	spwl_row = fft_real(interp_value, fft_n);
	spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_v_row);
	us_v.set_row(i, us_v_row);
  }
  for (int i = 0; i < nIS; i++){
	interp1(IS[i].time, IS[i].value, ts, interp_value);
	spwl_row = fft_real(interp_value, fft_n);
	spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_i_row);
	us_i.set_row(i, us_i_row);
  }
  pdata.us.set_size(nVS+nIS, np);
  pdata.us = concat_vertical(us_v, us_i);
  fft_run_time.stop();
	
  /* Solve Ax=b */
  pthread_t* threads = new pthread_t[np];
  int ** thd_idx = new int*[np];
  pthread_attr_t attr;
  void *status;
  pdata.Z.set_size(nDim, np);
  pdata.G = G;
  pdata.C = C;
  pdata.B = B;
  pthread_mutex_init(&mutexz, NULL);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (int i = 0; i < np; i++){
	thd_idx[i] = new int;
	*thd_idx[i] = i;
	pthread_create(&threads[i], &attr, solve_axb, (void *)thd_idx[i]);
  }
  pthread_attr_destroy(&attr);
  for (int i = 0; i < np; i++){
	pthread_join(threads[i], &status);
  }
  pthread_mutex_destroy(&mutexz);

  delete [] threads;
  for (int i = 0; i < np; i++){
	delete thd_idx[i];
  }
  delete [] thd_idx;

  /* SVD */
  svd_run_time.start();
  mat U, V;
  vec S;
  int info;
  info =  svd0(pdata.Z, U, S, V);
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
	cur[i] = -1;
  }
  /* DC simulation */
  for (int i = 0; i < 1; i++){
	for(int j = 0; j < nVS; j++){
	  interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
	  u_col(j) = temp;
	}
	for(int j = 0; j < nIS; j++){
	  interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
	  u_col(nVS+j) = temp;
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
	interp_run_time.start();
	for(int j = 0; j < nVS; j++){
	  interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
	  u_col(j) = temp;
	}
	for(int j = 0; j < nIS; j++){
	  interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
	  u_col(nVS+j) = temp;
	}
	interp_run_time.stop();
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

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "Interpolation   \t: " << interp_run_time.get_time() << std::endl;
  std::cout << "FFT             \t: " << fft_run_time.get_time() << std::endl;
  std::cout << "sC+G            \t: " << sCpG_run_time.get_time() << std::endl;
  std::cout << "Symbolic        \t: " << cs_symbolic.get_time() << std::endl;
  std::cout << "Numeric         \t: " << cs_numeric.get_time() << std::endl;
  std::cout << "Solve           \t: " << cs_solve.get_time() << std::endl;
  std::cout << "Total           \t: " << cs_run_time.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "simulation      \t: " << sim_run_time.get_time() << std::endl;
}
#endif

void etbr2_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
				 Source *VS, int nVS, Source *IS, int nIS, 
				 double tstep, double tstop, int q, 
				 mat &Gr, mat &Cr, mat &Br, mat &X,
				 double &max_i, int &max_i_idx)
{

  Real_Timer interp_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer cs_run_time, svd_run_time, rmatrix_run_time;
  Real_Timer etbr_thread_run_time;

  etbr_thread_run_time.start();

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

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 0;
  double f_max = 0.5/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1/tstep/fft_n;
  double f_max = 0.5/tstep;
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
  pdata.samples = samples;  
  int np = samples.size();
  cout <<"# samples: (t) " << np << "   in etbr2_thread()" <<endl;
  
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
  printf("      nVS=%d, nIS=%d\n",nVS,nIS);
  printf("    Evaluation of voltage sources: %6.4e\n",interp_run_time.get_time() );
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
  //pdata.us.set_size(nVS+nIS, np);
  pdata.us = &us;
	
  /* Solve Ax=b */
  pthread_t* threads = new pthread_t[np];
  int ** thd_idx = new int*[np];
  pthread_attr_t attr;
  void *status;
  pdata.zvec = new vec[np];
  pdata.G = G;
  pdata.C = C;
  pdata.B = B;
  //pthread_mutex_init(&mutexz, NULL);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (int i = 0; i < np; i++){
	thd_idx[i] = new int;
	*thd_idx[i] = i;
	pthread_create(&threads[i], &attr, solve_axb, (void *)thd_idx[i]);
  }
  pthread_attr_destroy(&attr);
  for (int i = 0; i < np; i++){
	pthread_join(threads[i], &status);
  }
  //pthread_mutex_destroy(&mutexz);

  delete [] threads;
  for (int i = 0; i < np; i++){
	delete thd_idx[i];
  }
  delete [] thd_idx;

  /* SVD */
  svd_run_time.start();
  mat Z, U, V;
  vec S;
  int info;
  Z.set_size(nDim, np);
  for (int i = 0; i < np; ++i){
	Z.set_col(i, pdata.zvec[i]);
  }
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

  etbr_thread_run_time.stop();

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
  std::cout << "total reduction \t: " << etbr_thread_run_time.get_time() << std::endl;
#endif
}
