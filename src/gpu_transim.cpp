/*
 * Transient simulation function.
 *
 * This file provides an interface to the main function,
 * and GPU based simulation is enabled by "-gpu" option
 * on command line arguments
 *
 * Author: Xue-Xin Liu
 *         2011-Nov-08
 */

#include <iostream>
#include <fstream>

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
#include <itpp/base/math/min_max.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/sort.h>
#include <itpp/stat/misc_stat.h>
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"

#include "gpuData.h"
//extern "C" void cudaTranSim(gpuETBR *myGPUetbr);

extern long interp2_sum;

using namespace itpp;
using namespace std;

void myMemcpyD2S(float *dst, double *src, int n)
{
  if(src==NULL || dst==NULL) exit(-1);
  for(int i=0; i<n; i++)
    dst[i] = (float)src[i];
}

void myMemcpyL2I(int *dst, long int *src, int n)
{
  if(src==NULL || dst==NULL) exit(-1);
  for(int i=0; i<n; i++)
    dst[i] = (int)src[i];
}

// extern "C"
void gpu_transim(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
		 double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
		 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
		 const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
		 int num, int ir_info, char *ir_name,
		 gpuETBR *myGPUetbr)
{
  Real_Timer interp2_run_time, solve_red_lu_time;	
  Real_Timer check_run_time1, check_run_time2, solveLU_run_time, sim_run_time;
  Real_Timer init_time, compute_sol_time, ir_run_time;

  sim_run_time.start();

  init_time.start();
  
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, avg_ir; //  min_ir,
  int max_ir_idx; // , min_ir_idx
  ivec sorted_max_value_idx, sorted_min_value_idx, 
	sorted_avg_value_idx, sorted_ir_value_idx;
  int nNodes = tc_node.size();
  int display_num = num<tc_node.size()?num:tc_node.size();
  max_value.set_size(nNodes);
  min_value.set_size(nNodes);
  avg_value.set_size(nNodes);
  sorted_max_value_idx.set_size(nNodes);
  sorted_min_value_idx.set_size(nNodes);
  sorted_avg_value_idx.set_size(nNodes);
  sorted_ir_value_idx.set_size(nNodes);

  // the initial effective resistance
  // double min_eff_resist = 1e5;
  // double avg_eff_resist = 1e5;  
  // double max_eff_resist = 0;  

  UF_long nDim = B->m;
  // UF_long nSDim = B->n;
  vec u_col(nVS+nIS);
  UF_long n = G->n;
  
  vec w(n);
  w.zeros();
  vec wc(nNodes);
  wc.zeros();
  vec w_prev(n);
  w_prev.zeros();
  vec w_r(q);
  w_r.zeros();
  vec w_r1(q);
  w_r1.zeros();

  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  
  /* // Duo Li's orginial //  // XXLiu's modification DAC12 comment out this part
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vec* slope = new vec[nVS+nIS];
  for(int i = 0; i < nVS; i++){
	int len = VS[i].time.size();
	slope[i].set_size(len-1);
	for(int j = 0; j < len-1; j++){
	  double delta = (VS[i].value(j+1) - VS[i].value(j)) / (VS[i].time(j+1) - VS[i].time(j));
	  slope[i].set(j, delta);
	}
  }
  for(int i = 0; i < nIS; i++){
	int len = IS[i].time.size();
	slope[nVS+i].set_size(len-1);
	for(int j = 0; j < len-1; j++){
	  double delta = (IS[i].value(j+1) - IS[i].value(j)) / (IS[i].time(j+1) - IS[i].time(j));
	  slope[nVS+i].set(j, delta);
	}
  }
  */
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
  /* // Duo Li's original
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)], slope[nVS+*it]);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  */
  for (int i = 0; i < 1; i++){ // XXLiu's modification DAC12
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  temp = VS[*it].value(0);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  temp = IS[*it].value(0);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  
  vec xn(n);
  xn.zeros();
  vec x(n);
  x.zeros();
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;
  Symbolic = cs_dl_sqr(order, G, 0);
  Numeric = cs_dl_lu(G, Symbolic, tol);
  cs_dl_ipvec(Numeric->pinv, w._data(), x._data(), n);
  cs_dl_lsolve(Numeric->L, x._data());
  cs_dl_usolve(Numeric->U, x._data());
  cs_dl_ipvec(Symbolic->q, x._data(), xn._data(), n);  
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xn(port(j)));
  }
  ivec tc_vec(nNodes);
  for (int j = 0; j < nNodes; j++){
	tc_vec(j) = tc_node[j];
  }
  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xn(tc_node[j]);
	  min_value(j) = xn(tc_node[j]);
	  avg_value(j) = xn(tc_node[j]);
	}
	ir_run_time.stop();
  }  
  vec xdcc = xn(tc_vec);

#ifdef _DEBUG
  //  cout << "max_dc_voltage = " << max_dc_voltage << endl;
#endif

  int nport = port.size();
  mat Xp(nport, q);
  if (nport > 0){
	for(int i = 0; i < nport; i++){
	  Xp.set_row(i, X.get_row(port(i)));
	}
  }
  mat Xc(nNodes, q);
  for(int i = 0; i < nNodes; i++){
	Xc.set_row(i, X.get_row(tc_node[i]));
  }

  mat XT = X.T();
  vec xn_r = XT * xn;

  init_time.stop();

  /* Transient simulation */
  /*  // XXLiu comment out
  cs_dl *right = cs_dl_spalloc(C->m, C->n, C->nzmax, 1, 0);
  for (UF_long i = 0; i < C->n+1; i++){
	right->p[i] = C->p[i];
  }
  for (UF_long i = 0; i < C->nzmax; i++){
	right->i[i] = C->i[i];
	right->x[i] = 1/tstep*C->x[i];
  }
  cs_dl *left = cs_dl_add(G, right, 1, 1);
  Symbolic = cs_dl_sqr(order, left, 0);
  Numeric = cs_dl_lu(left, Symbolic, tol);
  
  cs_dl *CG = cs_dl_add(C, G, 1/tstep, 1);
  mat CGX(nDim, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(left, X.get_col(j)._data(), v._data());
	CGX.set_col(j, v);
  }
  cs_dl_spfree(CG);
  cs_dl_spfree(left);
  mat CX(nDim, q);
  for (int j = 0; j < q; j++){
	vec v(nDim);
	v.zeros();
	(void) cs_dl_gaxpy(right, X.get_col(j)._data(), v._data());
	CX.set_col(j, v);
  }
  */
  // mat CGXc = CGX.get_rows(tc_vec); // XXLiu comment out
  // mat CXc = CX.get_rows(tc_vec); // XXLiu comment out

  mat right_r = 1/tstep*Cr;
  mat left_r = Gr + right_r;
  mat l_left_r, u_left_r;
  ivec p_r;
  solveLU_run_time.start();
  lu(left_r, l_left_r, u_left_r, p_r);
  solveLU_run_time.stop();

  // double *tmpA_double=(double*)malloc(q*q*sizeof(double));
  // float *tmpA=(float*)malloc(q*q*sizeof(float));
  // if(myGPUetbr->use_cuda_single) {
  //   //Mat<float> left_r_single, l_left_r_single, u_left_r_single;  
  //   //left_r_single.set_size(q,q,false);
  //   /* The mat class uses column major. */
  //   /*
  //   memcpy(tmpA_double, left_r._data(), q*q*sizeof(double));
  //   myMemcpyD2S(tmpA, tmpA_double, q*q);
  //   for(int i=0; i<q; i++)
  //     for(int j=0; j<q; j++)
  //       left_r_single.set(i,j,tmpA[i+j*q]);
  //   */
  // }


  interp2_run_time.start();
  /********************* GPU preparation ******************************/
  Real_Timer GPUtime;
  printf("*** GPU accelerated transient simulation ***\n");
  myGPUetbr->numPts = ts.size();
  
  myGPUetbr->n = n;
  myGPUetbr->m = Br.cols();  printf("          Br size: [%d,%d]\n",Br.rows(),Br.cols());
  myGPUetbr->ldUt = (((myGPUetbr->numPts-1) +31)/32)*32;
  
  /* The following section need CPU generated source info. */
  // myGPUetbr->ut_host=(double*)malloc((myGPUetbr->m)*(myGPUetbr->ldUt)*sizeof(double));

  myGPUetbr->ipiv_host=(int*)malloc(q*sizeof(int));
  //myGPUetbr->V_host=(double*)malloc(n*q*sizeof(double));
  if(nport > 0)
    myGPUetbr->LV_host=(double*)malloc(nport*q*sizeof(double));
  double *A_hCG=(double*)malloc(q*q*sizeof(double));
  myGPUetbr->L_hCG_host=(double*)malloc(q*q*sizeof(double));
  myGPUetbr->U_hCG_host=(double*)malloc(q*q*sizeof(double));
  myGPUetbr->hC_host=(double*)malloc(q*q*sizeof(double));
  myGPUetbr->Br_host=(double*)malloc(q*(myGPUetbr->m)*sizeof(double));
  myGPUetbr->xr0_host=(double*)malloc(q*sizeof(double));
  if(nport > 0)
    myGPUetbr->x_host=(double*)malloc(nport*(myGPUetbr->numPts)*sizeof(double));

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    if(nport > 0)
      myGPUetbr->LV_single_host=(float*)malloc(nport*q*sizeof(float));
    myGPUetbr->L_hCG_single_host=(float*)malloc(q*q*sizeof(float));
    myGPUetbr->U_hCG_single_host=(float*)malloc(q*q*sizeof(float));
    myGPUetbr->hC_single_host=(float*)malloc(q*q*sizeof(float));
    myGPUetbr->Br_single_host=(float*)malloc(q*(myGPUetbr->m)*sizeof(float));
    myGPUetbr->xr0_single_host=(float*)malloc(q*sizeof(float));
    if(nport > 0)
      myGPUetbr->x_single_host=(float*)malloc(nport*(myGPUetbr->numPts)*sizeof(float));
  }

  int *Pvec=(int*)malloc(q*sizeof(int));
  memcpy(Pvec, p_r._data(), q*sizeof(int));  
  for(int i=0; i<q; i++) myGPUetbr->ipiv_host[i] = i;
  for(int i=0; i<q-1; i++) {
    int tmp = myGPUetbr->ipiv_host[Pvec[i]];
    myGPUetbr->ipiv_host[Pvec[i]] = myGPUetbr->ipiv_host[i];
    myGPUetbr->ipiv_host[i] = tmp;
  }
  free(Pvec);

  // DOUBLE PRECISION, but also needed for SINGLE PRECISION.
  // memcpy(myGPUetbr->V_host, X._data(), n*q*sizeof(double));
  memcpy(myGPUetbr->LV_host, Xp._data(), nport*q*sizeof(double));
  memcpy(A_hCG, left_r._data(), q*q*sizeof(double));
  memcpy(myGPUetbr->L_hCG_host, l_left_r._data(), q*q*sizeof(double));
  memcpy(myGPUetbr->U_hCG_host, u_left_r._data(), q*q*sizeof(double));
  memcpy(myGPUetbr->hC_host, right_r._data(), q*q*sizeof(double));
  memcpy(myGPUetbr->Br_host, Br._data(), q*(myGPUetbr->m)*sizeof(double));
  //xn_r = XT * xn;
  memcpy(myGPUetbr->xr0_host, xn_r._data(), q*sizeof(double));

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    //myMemcpyD2S(myGPUetbr->V_single_host, myGPUetbr->V_host, n*q);
    myMemcpyD2S(myGPUetbr->LV_single_host, myGPUetbr->LV_host, nport*q);
    // myMemcpyD2S(myGPUetbr->L_hCG_single_host, myGPUetbr->L_hCG_host, q*q);//***
    // myMemcpyD2S(myGPUetbr->U_hCG_single_host, myGPUetbr->U_hCG_host, q*q);//***   
    myMemcpyD2S(myGPUetbr->hC_single_host, myGPUetbr->hC_host, q*q);
    myMemcpyD2S(myGPUetbr->Br_single_host, myGPUetbr->Br_host, q*(myGPUetbr->m));
    //xn_r = XT * xn;
    myMemcpyD2S(myGPUetbr->xr0_single_host, myGPUetbr->xr0_host, q);
  }

  // for(int i=0; i<nIS; i++) {
  //   printf("  IS[%d] length: %d\n", i, IS[i].time.length());
  // }

  myGPUetbr->nport = nport;  printf("          Lr size: [%d,%d]\n",nport,q);

  /* The following section uses CPU to prepare the source info. */
  /*
  Real_Timer myTime;
  myTime.start();
  for (int i = 1; i < ts.size(); i++){
    for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
      interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
      u_col(*it) = temp;
    }
    interp_next_step2(ts[i], IS, var_i, cur, slope, nVS, u_col);
    memcpy(myGPUetbr->ut_host+(i-1)*(myGPUetbr->m), u_col._data(), (myGPUetbr->m)*sizeof(double));
  }
  myTime.stop();
  printf("                Source Evaluation: %6.4e\n",myTime.get_time());
  */
  /************ CPU generated source info transfered. ************/

  if(BLK_SIZE_UTGEN < MAX_PWL_PTS) {
    printf("       Error: BLK_SIZE_UTGEN should be no less than MAX_PWL_PTS\n");
    while(!getchar()) ;
  }

  /* The following section prepares the source info in order to use GPU evaluation. */
  // store DC voltage source info.
  myGPUetbr->dcVt_host=(double*)malloc(nVS*sizeof(double));
  for(int i=0; i<nVS; i++)
    myGPUetbr->dcVt_host[i] = VS[i].value(0);

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    myGPUetbr->dcVt_single_host=(float*)malloc(nVS*sizeof(float));
    myMemcpyD2S(myGPUetbr->dcVt_single_host,  myGPUetbr->dcVt_host, nVS);
  }

  if(myGPUetbr->PWLcurExist) {
    myGPUetbr->PWLnumPts_host=(int*)malloc(nIS*sizeof(int));
    myGPUetbr->PWLtime_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    myGPUetbr->PWLval_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    for(int i=0; i<nIS; i++) {
      int herePWLnumPts=IS[i].time.size();
      //printf(" size: %d\n",herePWLnumPts);
      if(herePWLnumPts > MAX_PWL_PTS) {
	printf("       Error: More PWL points than allowed. %d > %d at source-%d\n",herePWLnumPts, MAX_PWL_PTS, i);
	while(!getchar()) ;
      }
      myGPUetbr->PWLnumPts_host[i] = herePWLnumPts;
      for(int j=0; j<herePWLnumPts; j++) {
	myGPUetbr->PWLtime_host[i*MAX_PWL_PTS+j] = IS[i].time(j);
	myGPUetbr->PWLval_host[i*MAX_PWL_PTS+j] = IS[i].value(j);
      }
    }

    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      myGPUetbr->PWLtime_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myGPUetbr->PWLval_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myMemcpyD2S(myGPUetbr->PWLtime_single_host, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS);
      myMemcpyD2S(myGPUetbr->PWLval_single_host, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS);
    }
  }

  interp2_run_time.stop();

  GPUtime.start();
  if(nport>0)
    cudaTranSim(myGPUetbr);
  GPUtime.stop();
  printf("                Clock based timer: %6.4e\n",GPUtime.get_time());

  cudaTranSim_shutdown(myGPUetbr);
  /***************************************************/

//  vec xn1(n), xn1t(n), xnr(n);;
//  xn1.zeros();
//  xn1t.zeros();
//  xnr.zeros();
//  vec xn1_r(q), xn1t_r(q), b(q);
//  xn1_r.zeros();
//  xn1t_r.zeros();
//  b.zeros();
//  vec xn1p(nport), xn1c(nNodes);
//  xn1p.zeros();
//  xn1c.zeros();
//  vec w_res(n), w_res1(n);
//  w_res.zeros();
//  w_res1.zeros();
//  vec wc_res(nNodes), wc_res1(nNodes);
//  wc_res.zeros();
//  wc_res1.zeros();
//  vec xn1c_appx(nNodes);
//  xn1c_appx.zeros();
//  vec resi(n);
//  resi.zeros();
//  vec resic(nNodes);
//  resic.zeros();
//  vec x_diff(nNodes);
//  x_diff.zeros();
//  int solveLU_num = 0;
//  // double max_voltage_drop = 0;
//  // double allow_cur_resid = 0;
//  // double allow_vol_drop = 0;
//  double xn1c_max = max(xdcc)>0?max(xdcc):max(abs(xdcc));
//  double xn1c_min = min(xdcc)>0?min(xdcc):0;
//  // double resi_over_u = 2;
//
//  // double init_allow_vol_drop = xn1c_min*1e-4;
//  // allow_vol_drop = init_allow_vol_drop;
//
//  int *var_ii = new int[var_i.size()];
//  int nvar_ii = var_i.size();
//  for (int i = 0; i < var_i.size(); i++){
//	var_ii[i] = var_i[i];
//  }
//
//  init_time.stop();
//
//  for (int i = 1; i < ts.size(); i++){
//    for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
//      interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
//      u_col(*it) = temp;
//    }
//    interp_next_step2(ts[i], IS, var_i, cur, slope, nVS, u_col);
//    // printf("               u_col size: [%d,1]\n", u_col.length());
//
//
//    solve_red_lu_time.start();	
//    multiply(Br, u_col._data(), w_r._data());
//    w_r1 = right_r * xn_r;
//    w_r = w_r + w_r1;
//    interchange_permutations(w_r, p_r);
//    forward_substitution(l_left_r, w_r, xn1t_r);
//    backward_substitution(u_left_r, xn1t_r, xn1_r);
//    solve_red_lu_time.stop();
//    compute_sol_time.start();
//		
//    //xn1 = X * xn1_r;
//    if(nport >0){
//      xn1p = Xp * xn1_r;
//      sim_port_value.set_col(i, xn1p);
//    }
//    xn_r = xn1_r;
//    compute_sol_time.stop();
//    if(ir_info){
//      xn1c = Xc * xn1_r;
//      ir_run_time.start();
//      for (int j = 0; j < nNodes; j++){
//	if (max_value(j) < xn1c(j)){
//	  max_value(j) = xn1c(j);
//	}
//	if (xn1c(j) < min_value(j)){
//	  min_value(j) = xn1c(j);
//	}
//	avg_value(j) += xn1c(j);
//      }
//      ir_run_time.stop();
//    }	
//  }
//  delete [] cur; // keep the two variables for GPU calculation
//  delete [] slope;
//  cs_dl_spfree(right);
//  cs_dl_sfree(Symbolic);
//  cs_dl_nfree(Numeric);
//
//  if (ir_info){
//	ir_run_time.start();
//	avg_value /= ts.size();
//	sorted_max_value_idx = sort_index(max_value);
//	sorted_avg_value_idx = sort_index(avg_value);
//	vec sgn_value = sgn(max_value) - sgn(min_value);
//	ir_value.set_size(max_value.size());
//	for (int i = 0; i < sgn_value.size(); i++){
//	  if(sgn_value(i) == 0){
//		ir_value(i) = max_value(i) - min_value(i);
//	  }
//	  else{
//		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
//	  }
//	}
//	//ir_value = max_value - min_value;
//	max_ir = max(ir_value);
//	max_ir_idx = max_index(ir_value);
//	avg_ir = sum(ir_value)/ir_value.size(); 
//	sorted_ir_value_idx = sort_index(ir_value);
//	std::cout.precision(6);
//	printf("****** Node Voltage Info ******  \n");
//	printf("#Tap Currents: %d\n", tc_node.size() );
//	printf("******\n");
//	// fprintf(stdout,"%-30s %15s\n", "Max", "Node Voltage");
//	// fprintf(stdout,"%-30s %15s\n", "----", "------------");
//	printf("Max %d   Node Voltage: \n",display_num);
//	for(int i = 0; i < display_num; i++)
//	  fprintf(stdout,"-30s : %15g\n",
//		  tc_name[sorted_max_value_idx(nNodes-1-i)].c_str(),
//		  max_value(sorted_max_value_idx(nNodes-1-i)) );
//	printf("******\n");
//	cout << "Avg " << display_num << " Node Voltage: " << endl;
//	for (int i = 0; i < display_num; i++){
//	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
//		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
//	}
//	cout << "****** IR Drop Info ******  " << endl;
//	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
//	cout << "Avg IR:     " << avg_ir << endl;
//	cout << "******" << endl;
//	cout << "Max " << display_num << " IR: " << endl;
//	for (int i = 0; i < display_num; i++){
//	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
//		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
//	}
//	cout << "******" << endl;
//
//	ofstream out_ir;
//	out_ir.open(ir_name);
//	if (!out_ir){
//	  cout << "couldn't open " << ir_name << endl;
//	  exit(-1);
//	}
//	for (int i = 0; i < tc_node.size(); i++){
//	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
//			 << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
//	}
//	out_ir.close();
//	cout << "** " << ir_name << " dumped" << endl;
//	ir_run_time.stop();
//  }

  sim_run_time.stop();
 
#ifndef UCR_EXTERNAL 
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "initial time     \t: " << init_time.get_time() << std::endl;
  std::cout << "interpolation2   \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "error check1     \t: " << check_run_time1.get_time() << std::endl;
  std::cout << "error check2     \t: " << check_run_time2.get_time() << std::endl;
  std::cout << "solve reduced LU \t: " << solve_red_lu_time.get_time() << std::endl;  
  printf("CPU LU Factor.   \t: %6.4e\n", solveLU_run_time.get_time());// << std::endl;//<< " \t#" << solveLU_num << std::endl;
  std::cout << "comp sol time    \t: " << compute_sol_time.get_time() << std::endl;
  printf("total simulation \t: %6.4e\n", sim_run_time.get_time() );
  //std::cout << "total simulation \t: " << sim_run_time.get_time() << std::endl;
  std::cout << "IR analysis      \t: " << ir_run_time.get_time() << std::endl;
#endif

  /***************************************************/
  /*
  char filename[] = "testSaveETBR.bin";
  FILE *fp;
  fp = fopen(filename, "wb");
    fwrite(&q, sizeof(int), 1, fp);
    fwrite(myGPUetbr->ipiv_host, sizeof(int), q, fp);
    fwrite(A_hCG, sizeof(double), q*q, fp);
    fwrite(myGPUetbr->L_hCG_host, sizeof(double), q*q, fp);
    fwrite(myGPUetbr->U_hCG_host, sizeof(double), q*q, fp);
  fclose(fp);
  printf("        >>> >>> Binary data file saved in testSaveETBR.bin\n");
  */
  free(A_hCG);

  if(myGPUetbr->use_cuda_double) {
    for(int i=0; i<nport; i++)
      for(int j=0; j<ts.size(); j++)
	sim_port_value.set(i, j, myGPUetbr->x_host[i+j*nport]);
  }
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    for(int i=0; i<nport; i++)
      for(int j=0; j<ts.size(); j++)
	sim_port_value.set(i, j, (double)myGPUetbr->x_single_host[i+j*nport]);
  }
  for(int i=0; i<nport; i++)
    sim_port_value.set(i, 0, xn(port(i)));
  
  free(myGPUetbr->ipiv_host);
  free(myGPUetbr->L_hCG_host);
  free(myGPUetbr->U_hCG_host);
}
/*
  slope = new vec[nVS+nIS];
  for(int i = 0; i < nVS; i++){
    int len = VS[i].time.size();
    slope[i].set_size(len-1);
    for(int j = 0; j < len-1; j++){
      double delta = (VS[i].value(j+1) - VS[i].value(j)) / (VS[i].time(j+1) - VS[i].time(j));
      slope[i].set(j, delta);
    }
  }
  for(int i = 0; i < nIS; i++){
    int len = IS[i].time.size();
    slope[nVS+i].set_size(len-1);
    for(int j = 0; j < len-1; j++){
      double delta = (IS[i].value(j+1) - IS[i].value(j)) / (IS[i].time(j+1) - IS[i].time(j));
      slope[nVS+i].set(j, delta);
    }
  }
  delete [] slope;



    //for(int j=0; j<nIS; j++) {
    //  int k;
    //  for(k=0; k<IS[j].time.size(); k++) {
    //	if( ts(i) <= IS[j].time(k) ) break;
    //  }
    //  if(k==IS[j].time.size()-1) u_col(j+nVS) = IS[j].value(k);
    //  else
    //	u_col(j+nVS) = IS[j].value(k-1)+
    //	  (ts(i)-IS[j].time(k-1))*(IS[j].value(k)-IS[j].value(k-1))/(IS[j].time(k)-IS[j].time(k-1));
    //}

*/
