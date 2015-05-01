/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfil$
 *    $Revision: 1.1 $
 *    $Date: 2011/12/06 02:25:42 $
 *    Authors: Xue-Xin Liu, Duo Li
 *
 *    Functions: ETBR function with CSparse, GPU and pthread implementation
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

AXBDATA pdata2011;


void *solve_axb2011(void * threadarg)
{
  UF_long nDim = pdata2011.B->m;
  UF_long nSDim = pdata2011.B->n;
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;
  int i = *(int*) threadarg;

  cs_dl *A;
  if (pdata2011.C != NULL)
	A = cs_dl_add(pdata2011.G, pdata2011.C, 1, pdata2011.samples(i));
  else
	A = pdata2011.G;

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
  (void) cs_dl_gaxpy(pdata2011.B, pdata2011.us->get_col(i)._data(), b._data());
  cs_dl_ipvec(Numeric->pinv, b._data(), x._data(), A->n);
  cs_dl_lsolve(Numeric->L, x._data());
  cs_dl_usolve(Numeric->U, x._data());
  cs_dl_ipvec(Symbolic->q, x._data(), z._data(), A->n);  	

  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  if (A != pdata2011.G)
	cs_dl_spfree(A);

  pdata2011.zvec[i] = z;
  /*
  pthread_mutex_lock(&mutexz);
  pdata2011.Z.set_col(i, z);	
  pthread_mutex_unlock(&mutexz);
  */

  pthread_exit((void*) 0);
}


// XXLiu: this function was etbr2_thread().
void gpu_etbr_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
		     Source *VS, int nVS, Source *IS, int nIS, 
		     double tstep, double tstop, int q, 
		     mat &Gr, mat &Cr, mat &Br, mat &X,
		     double &max_i, int &max_i_idx, gpuETBR *myGPUetbr)
{

  Real_Timer interp_run_time, fft_run_time, sCpG_run_time;
  Real_Timer cs_symbolic, cs_numeric, cs_solve;
  Real_Timer cs_run_time, svd_run_time, rmatrix_run_time;
  Real_Timer etbr_thread_run_time;

  myGPUetbr->q = q;
  myGPUetbr->m = B->n;
  myGPUetbr->tstep = tstep;
  myGPUetbr->tstop = tstop;
  
  if(myGPUetbr->PWLcurExist && myGPUetbr->PULSEcurExist) {
    printf("                    Do not support PWL and PULSE current sources at the same time.\n");
    while( !getchar() ) ;
  }
  if(myGPUetbr->PWLcurExist) {
    if(myGPUetbr->PWLcurExist == nIS)
      printf("       All PWL current sources.\n");
    else {
      printf("       Error: There are non-PWL current sources mingled with PWL.\n");
      while(!getchar()) ;
    }
  }
  if(myGPUetbr->PULSEcurExist) {
    if(myGPUetbr->PULSEcurExist == nIS)
      printf("       All PULSE current sources.\n");
    else {
      printf("       Error: There are non-PULSE current sources mingled with PULSE.\n");
      while(!getchar()) ;
    }
  }
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;

  if(q > MAX_THREADS) {
    printf("                    Reduced order must be smaller than MAX_THREADS.\n");
    while( !getchar() ) ;
  }
  if(myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m) {
    printf("                    myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m\n");
    while( !getchar() ) ;
  }

  if(myGPUetbr->PWLvolExist || myGPUetbr->PULSEvolExist) {
    printf("                    Do not support PWL and PULSE voltage sources.\n");
    while( !getchar() ) ;
  }

  etbr_thread_run_time.start();

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  if(myGPUetbr->PULSEcurExist)
    form_vec(ts, 0, tstep, 2*myGPUetbr->PULSEtime_host[4]);
  else
    form_vec(ts, 0, tstep, tstop);

  int fft_n = 512;
  int L = ts.size(), ldUt=((L+31)/32)*32, nBat=2048;
  int N = 10;  //1024 samples    

  /* FFT */
  //int N = floor_i(log2(L))+1;
  
  fft_n = pow2i(N);
  std::cout << "# time steps: "<< L << " ldUt=" << ldUt << std::endl;
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
  pdata2011.samples = samples;  
  int np = samples.size();
  cout <<"# samples: (t) " << np << "   in gpu_etbr_thread()" <<endl;
  
  vec f;
  // form_vec(f, 0, 1, fft_n/2);
  f = linspace(0, 1, fft_n/2+1);
  f *= 0.5/tstep;
  cvec spwl_row;
  vec abs_spwl_row;
  mat us(nVS+nIS, np);
  vec us_row(np);
  vec interp_value(ts.size());
  printf("      nVS=%d, nIS=%d\n",nVS,nIS);
  if(nVS < 32) {
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
  }
  else {
    printf("      GPU parallel voltage source evaluation.\n");
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
  }
  printf("    Evaluation of voltage sources: %6.4e\n",interp_run_time.get_time() );

  if(nVS < 32) {
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
  }
  else {
    printf("      GPU parallel current source evaluation.\n");
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
  }

  //pdata2011.us.set_size(nVS+nIS, np);
  pdata2011.us = &us;
	
  /* Solve Ax=b */
  pthread_t* threads = new pthread_t[np];
  int ** thd_idx = new int*[np];
  pthread_attr_t attr;
  void *status;
  pdata2011.zvec = new vec[np];
  pdata2011.G = G;
  pdata2011.C = C;
  pdata2011.B = B;
  //pthread_mutex_init(&mutexz, NULL);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (int i = 0; i < np; i++){
	thd_idx[i] = new int;
	*thd_idx[i] = i;
	pthread_create(&threads[i], &attr, solve_axb2011, (void *)thd_idx[i]);
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
	Z.set_col(i, pdata2011.zvec[i]);
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



//  else {
//    form_vec(ts, 0, tstep, tstop);
//    myGPUetbr->numPts = ts.size();
//
//    /* FFT */
//    int fft_n = 512;
//    int L = ts.size(), ldUt=((L+31)/32)*32, nBat=2048;
//    //int N = floor_i(log2(L))+1;
//    int N = 10;  //1024 samples
//    
//    fft_n = pow2i(N);
//    std::cout << "# time steps: "<< L << " ldUt=" << ldUt << std::endl;
//    std::cout << "# FFT points: "<< fft_n << std::endl;
//    
//#if 0
//    /* sampling: uniform in linear scale */
//    double f_min = 0;
//    double f_max = 0.5/tstep;
//    vec samples = linspace(f_min, f_max, q);
//#endif 
//    
//    /* sampling: uniform in log scale */
//    double f_min = 1/tstep/fft_n;
//    double f_max = 0.5/tstep;
//    vec lin_samples;
//    vec samples;
//    if(q > 6){
//      lin_samples = linspace(std::log10(f_min), std::log10(f_max), q-6);
//    samples = pow10(lin_samples); 
//    }
//    
//    //samples.ins(0, 1e9);
//    //samples.ins(0, 1e8);
//    samples.ins(0, 1e7);
//    samples.ins(0, 1e6);
//    samples.ins(0, 1e5);
//    samples.ins(0, 1e1);
//    samples.ins(0, 1);
//    samples.ins(0, 0);
//    pdata2011.samples = samples;  
//    int np = samples.size();
//    cout <<"# samples: (t) " << np << "   in gpu_etbr_thread()" <<endl;
//    
//    vec f;
//    // form_vec(f, 0, 1, fft_n/2);
//    f = linspace(0, 1, fft_n/2+1);
//    f *= 0.5/tstep;
//    cvec spwl_row;
//    vec abs_spwl_row;
//    mat us(nVS+nIS, np);
//    vec us_row(np);
//    vec interp_value(ts.size());
//    printf("      nVS=%d, nIS=%d\n",nVS,nIS);
//    if(nVS < 32) {
//      for (int i = 0; i < nVS; i++){
//	interp_run_time.start();
//	interp1(VS[i].time, VS[i].value, ts, interp_value);
//	interp_run_time.stop();
//	fft_run_time.start();
//	// spwl_row = fft_real(interp_value, fft_n);
//	interp_value.set_size(fft_n, true);
//	fft_real(interp_value, spwl_row);
//	spwl_row /= L;
//	fft_run_time.stop();
//	// spwl_row *= (double)1/fft_n;
//	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
//	abs_spwl_row *= 2;
//	interp1(f, abs_spwl_row, samples, us_row);
//	us.set_row(i, us_row);
//      }
//    }
//    else {
//      printf("      GPU parallel voltage source evaluation.\n");
//      for (int i = 0; i < nVS; i++){
//	interp_run_time.start();
//	interp1(VS[i].time, VS[i].value, ts, interp_value);
//	interp_run_time.stop();
//	fft_run_time.start();
//	// spwl_row = fft_real(interp_value, fft_n);
//	interp_value.set_size(fft_n, true);
//	fft_real(interp_value, spwl_row);
//	spwl_row /= L;
//	fft_run_time.stop();
//	// spwl_row *= (double)1/fft_n;
//	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
//	abs_spwl_row *= 2;
//	interp1(f, abs_spwl_row, samples, us_row);
//	us.set_row(i, us_row);
//      }
//    }
//    printf("    Evaluation of voltage sources: %6.4e\n",interp_run_time.get_time() );
//
//    if(nVS < 32) {
//      for (int i = 0; i < nIS; i++){
//	interp_run_time.start();
//	interp1(IS[i].time, IS[i].value, ts, interp_value);
//	interp_run_time.stop();
//	vec abs_interp_value = abs(interp_value);
//	double max_interp = max(abs_interp_value);
//	// int max_interp_idx = max_index(abs_interp_value);
//	if (max_interp > max_i){
//	  max_i = max_interp;
//	  max_i_idx = max_index(abs_interp_value);
//	}
//	fft_run_time.start();
//	// spwl_row = fft_real(interp_value, fft_n);
//	interp_value.set_size(fft_n, true);
//	fft_real(interp_value, spwl_row);
//	spwl_row /= L;
//	fft_run_time.stop();
//	// spwl_row *= (double)1/fft_n;
//	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
//	abs_spwl_row *= 2;
//	interp1(f, abs_spwl_row, samples, us_row);
//	us.set_row(nVS+i, us_row);
//      }
//    }
//    else {
//      printf("      GPU parallel current source evaluation.\n");
//      for (int i = 0; i < nIS; i++){
//	interp_run_time.start();
//	interp1(IS[i].time, IS[i].value, ts, interp_value);
//	interp_run_time.stop();
//	vec abs_interp_value = abs(interp_value);
//	double max_interp = max(abs_interp_value);
//	// int max_interp_idx = max_index(abs_interp_value);
//	if (max_interp > max_i){
//	  max_i = max_interp;
//	  max_i_idx = max_index(abs_interp_value);
//	}
//	fft_run_time.start();
//	// spwl_row = fft_real(interp_value, fft_n);
//	interp_value.set_size(fft_n, true);
//	fft_real(interp_value, spwl_row);
//	spwl_row /= L;
//	fft_run_time.stop();
//	// spwl_row *= (double)1/fft_n;
//	abs_spwl_row = real(spwl_row(0,floor_i(fft_n/2)));
//	abs_spwl_row *= 2;
//	interp1(f, abs_spwl_row, samples, us_row);
//	us.set_row(nVS+i, us_row);
//      }
//    }
//  }
