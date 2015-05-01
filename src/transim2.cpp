/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: transim2.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:01 $
 *    Authors: Duo Li
 *
 *    Functions: Mixed ETBR and LU based transient simulation
 *
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

// #define _DEBUG

extern long interp2_sum;

using namespace itpp;
using namespace std;

void mixed_transim2(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
					double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
					mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
					const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
					int num, int ir_info, char *ir_name)
{
  Real_Timer interp2_run_time, solve_red_lu_time;	
  Real_Timer check_run_time1, check_run_time2, solveLU_run_time, sim_run_time;
  Real_Timer init_time, compute_sol_time, ir_run_time;

  sim_run_time.start();

  init_time.start();
  
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, min_ir, avg_ir;
  int max_ir_idx, min_ir_idx;
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
  double min_eff_resist = 1e5;
  double avg_eff_resist = 1e5;  
double max_eff_resist = 0;  

  UF_long nDim = B->m;
  UF_long nSDim = B->n;
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
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)], slope[nVS+*it]);
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
  //double max_dc_voltage = max(abs(xn.mid(0, n-VS)));
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

  /* Transient simulation */
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

  mat CGXc = CGX.get_rows(tc_vec);
  mat CXc = CX.get_rows(tc_vec);

  mat right_r = 1/tstep*Cr;
  mat left_r = Gr + right_r;
  mat l_left_r, u_left_r;
  ivec p_r;
  lu(left_r, l_left_r, u_left_r, p_r);

  vec xn1(n), xn1t(n), xnr(n);;
  xn1.zeros();
  xn1t.zeros();
  xnr.zeros();
  vec xn1_r(q), xn1t_r(q), b(q);
  xn1_r.zeros();
  xn1t_r.zeros();
  b.zeros();
  vec xn1p(nport), xn1c(nNodes);
  xn1p.zeros();
  xn1c.zeros();
  vec w_res(n), w_res1(n);
  w_res.zeros();
  w_res1.zeros();
  vec wc_res(nNodes), wc_res1(nNodes);
  wc_res.zeros();
  wc_res1.zeros();
  vec xn1c_appx(nNodes);
  xn1c_appx.zeros();
  vec resi(n);
  resi.zeros();
  vec resic(nNodes);
  resic.zeros();
  vec x_diff(nNodes);
  x_diff.zeros();
  int solveLU_num = 0;
  double max_voltage_drop = 0;
  double allow_cur_resid = 0;
  double allow_vol_drop = 0;
  double xn1c_max = max(xdcc)>0?max(xdcc):max(abs(xdcc));
  double xn1c_min = min(xdcc)>0?min(xdcc):0;
  double resi_over_u = 2;

  double init_allow_vol_drop = xn1c_min*1e-4;
  allow_vol_drop = init_allow_vol_drop;
#ifdef _DEBUG
  cout << "init allowed voltage drop: = "  << init_allow_vol_drop << endl;
#endif
  int *var_ii = new int[var_i.size()];
  int nvar_ii = var_i.size();
  for (int i = 0; i < var_i.size(); i++){
	var_ii[i] = var_i[i];
  }
  /*
  vec xn_r(q);
  xn_r.zeros();
  vec x_r(q);
  x_r.zeros();
  multiply(Br, u_col._data(), w_r._data());
  interchange_permutations(w_r, p_r);
  forward_substitution(l_left_r, w_r, x_r);
  backward_substitution(u_left_r, x_r, xn_r);
  */
  
  init_time.stop();
  /*
  int* cur0 = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur0[i] = cur[i];
  }
  for (int i = max_i_idx; i < max_i_idx+1; i++){

	++solveLU_num;

	interp2_run_time.start();
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
	  u_col(*it) = temp;
	}

	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  int len = IS[*it].time.size();
	  if (cur[nVS+(*it)] == len-1){
		temp = IS[*it].value[len-1];
	  }else{
		interp1(len, IS[*it], ts[i], temp, cur[nVS+(*it)], slope[nVS+*it]._data());
	  }
	  u_col[nVS+*it] = temp;
	}
	
	interp2_run_time.stop();

	solve_red_lu_time.start();	
	// w_r = Br * u_col;
	multiply(Br, u_col._data(), w_r._data());
	w_r1 = right_r * xn_r;
	w_r = w_r + w_r1;
	interchange_permutations(w_r, p_r);
	forward_substitution(l_left_r, w_r, xn1t_r);
	backward_substitution(u_left_r, xn1t_r, xn1_r);
	solve_red_lu_time.stop();

	check_run_time1.start();

	multiply(Xc, xn1_r._data(), xn1c._data());
    xn1c_appx = xn1c;
	double max_xn1c_appx = max(abs(xn1c - xdcc));
	check_run_time1.stop();

	check_run_time2.start();
	w.zeros();
	cs_dl_gaxpy(B, u_col._data(), w._data());
	multiply(CGX, xn1_r._data(), w_res._data());
	multiply(CX, xn_r._data(), w_res1._data());
	w_res = w_res - w_res1;
	resi = abs(w - w_res);
	double max_resid = max(resi.mid(0, n-nVS));
	double max_u_value = max(abs(u_col.mid(nVS, nVS+nIS-1)));
	//resi_over_u = max_resid / max_u_value;
	check_run_time2.stop();

	solveLU_run_time.start();
	xnr.zeros();
	xn = X * xn_r;
	cs_dl_gaxpy(right, xn._data(), xnr._data());
	b = w + xnr;
	cs_dl_ipvec(Numeric->pinv, b._data(), xn1t._data(), n);
	cs_dl_lsolve(Numeric->L, xn1t._data());
	cs_dl_usolve(Numeric->U, xn1t._data());
	cs_dl_ipvec(Symbolic->q, xn1t._data(), xn1._data(), n);   
	solveLU_run_time.stop();			
	xn1c = xn1(tc_vec);
	x_diff = abs(xn1c - xn1c_appx);
	double max_x_diff = max(x_diff);

	double eff_resist_tmp = max_x_diff / max_resid;
	//avg_eff_resist = (avg_eff_resist*(solveLU_num-1) + eff_resist_tmp)/solveLU_num;	
	if (eff_resist_tmp > avg_eff_resist)
	  avg_eff_resist = eff_resist_tmp;

#ifdef _DEBUG
	cout << "max_xn1c_appx = " << max_xn1c_appx << endl;
	cout << "max_x_diff = " << max_x_diff << endl;
	cout << "max_resid = " << max_resid << endl;
	cout << "max_u_value = " << max_u_value << endl;
	cout << "effective resistance = " << eff_resist_tmp << endl;
#endif
  }
  cout << "resi_over_u = " << resi_over_u << endl;
 
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = cur0[i];
  } 
  delete [] cur0;
  */
  for (int i = 1; i < ts.size(); i++){

	interp2_run_time.start();
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
	  u_col(*it) = temp;
	}
	/*
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  int len = IS[*it].time.size();
	  if (cur[nVS+(*it)] == len-1){
		temp = IS[*it].value[len-1];
	  }else{
		interp1(len, IS[*it], ts[i], temp, cur[nVS+(*it)], slope[nVS+*it]._data());
	  }	  
	  u_col[nVS+*it] = temp;
	}
	*/
	//interp_next_step3(ts[i], IS, var_ii, nvar_ii, cur, slope, nVS, u_col);
	interp_next_step2(ts[i], IS, var_i, cur, slope, nVS, u_col);
	
	interp2_run_time.stop();

	solve_red_lu_time.start();	
	// w_r = Br * u_col;
	multiply(Br, u_col._data(), w_r._data());
	w_r1 = right_r * xn_r;
	w_r = w_r + w_r1;
	interchange_permutations(w_r, p_r);
	forward_substitution(l_left_r, w_r, xn1t_r);
	backward_substitution(u_left_r, xn1t_r, xn1_r);
	solve_red_lu_time.stop();

	check_run_time1.start();
#ifdef _DEBUG
	cout << "time step: " << i << endl;
#endif

	// compute the all the voltage and maximum voltage	
	//xn1c = Xc * xn1_r;
	multiply(Xc, xn1_r._data(), xn1c._data());
    xn1c_appx = xn1c;
	double max_xn1c_appx = max(abs(xn1c - xdcc));
#ifdef _DEBUG
	cout << "max apprx voltage: = "  << max(abs(xn1c_appx)) << endl;
	cout << "max apprx voltage drop: = "  << max_xn1c_appx << endl;
#endif
	if(max_xn1c_appx >= max_voltage_drop)
	  {		
		max_voltage_drop = max_xn1c_appx;
		// compute the allowed voltage, which is percentage given by user
		// the default percentage is 5%, --Sheldon Tan
		allow_vol_drop = max_voltage_drop * threshold_percentage;
		
		// compute the allowed current residuals,
		if (abs(avg_eff_resist) > 1e-10)
		  allow_cur_resid = allow_vol_drop / avg_eff_resist;	
		else 
		  allow_cur_resid = 0;

#ifdef _DEBUG
		cout << "allowed voltage drop: = "  << allow_vol_drop << endl;
		cout << "allowed current residual = " << allow_cur_resid << endl;	
#endif
	  }	

	double max_u_value = max(abs(u_col.mid(nVS, nVS+nIS-1)));
#ifdef _DEBUG
    cout <<"max abs input current = " << max_u_value << endl;	
#endif	
	check_run_time1.stop();

	if(max_u_value*resi_over_u > allow_cur_resid)
	//if (1)
  	{
		check_run_time2.start();
		w.zeros();
		cs_dl_gaxpy(B, u_col._data(), w._data());
		wc = w(tc_vec);
		// vec w_res = CGX * xn1_r;
		multiply(CGXc, xn1_r._data(), wc_res._data());
		// w_res -= CX * xn_r;
		multiply(CXc, xn_r._data(), wc_res1._data());
		wc_res = wc_res - wc_res1;
	    resic = abs(wc - wc_res);
		//double max_resid_idx = max_index(resic);
		double max_resid = max(resic);
#ifdef _DEBUG
	    cout << "max residual current: " << max_resid << endl;
		cout << "max_u_value = " << max_u_value << endl;
#endif	
		check_run_time2.stop();
		if(max_resid > allow_cur_resid) 	
		  {		
			++solveLU_num;
#ifdef _DEBUG
			cout << "time step: " << i << endl;
			cout << "solveLU_num = " << solveLU_num << endl;
#endif
			solveLU_run_time.start();
			xnr.zeros();
			xn = X * xn_r;
			cs_dl_gaxpy(right, xn._data(), xnr._data());
			b = w + xnr;
			cs_dl_ipvec(Numeric->pinv, b._data(), xn1t._data(), n);
			cs_dl_lsolve(Numeric->L, xn1t._data());
			cs_dl_usolve(Numeric->U, xn1t._data());
			cs_dl_ipvec(Symbolic->q, xn1t._data(), xn1._data(), n);
			solveLU_run_time.stop();
#ifdef _DEBUG
			// compute the maximum exact (debugging staff)
			double max_xn1 = max(xn1.mid(0, n-nVS-1));
			cout << "max exact voltage: = "  << max_xn1 << endl;
#endif
			// compute the maximum voltage different
			xn1c = xn1(tc_vec);
			x_diff = abs(xn1c - xn1c_appx);
			//double max_x_diff_idx = max_index(x_diff);
			double max_x_diff = max(x_diff);
			double max_xn1c = max(abs(xn1c-xdcc));
#ifdef _DEBUG
			cout << "allowed current residual = " << allow_cur_resid << endl;
			cout << "** max residual current: " << max_resid << endl;	
			cout << "allowed voltage drop: = "  << max_voltage_drop * threshold_percentage << endl;
			cout << "** max voltage difference = " << max_x_diff << endl;	  
#endif
			// now compute the effetive resistance
			double eff_resist_tmp = max_x_diff/max_resid;
			// double eff_resist_tmp = max(elem_div(x_diff, resic));
#ifdef _DEBUG
			cout << "effective resistance = " << eff_resist_tmp << endl;
#endif	  	  
			// compute the effective resistance by taking average
			// if(eff_resist_tmp < min_eff_resist)
			// 	min_eff_resist = eff_resist_tmp;
			
			avg_eff_resist = (avg_eff_resist*(solveLU_num-1) + eff_resist_tmp)/solveLU_num;		   

#ifdef _DEBUG
			cout << "avg effective resistance = " << avg_eff_resist  << endl;
			//cout << "min effective resistance = " << min_eff_resist << endl;
#endif	  
			// recompute the maximum voltage
			if (max_eff_resist < eff_resist_tmp)
				max_eff_resist = eff_resist_tmp;
			double max_v_by_i = max_i * avg_eff_resist;
			//double max_v_by_i = max_i * max_eff_resist;
			
			if(max_v_by_i > max_voltage_drop)
			  {
				max_voltage_drop = max_v_by_i;
				allow_vol_drop = max_voltage_drop * threshold_percentage;
			  }
			
			if (max_voltage_drop < max_xn1c){
			  max_voltage_drop = max_xn1c;
			  allow_vol_drop = max_voltage_drop * threshold_percentage;
			}
			if (allow_vol_drop/avg_eff_resist > max_i*threshold_percentage)
			  if (solveLU_num > 10)
				allow_cur_resid = allow_vol_drop / avg_eff_resist;
		 	if (abs(allow_cur_resid) < 1e-10)
			  allow_cur_resid = max_i*threshold_percentage;
			if (allow_cur_resid > max_i*threshold_percentage)
				allow_cur_resid = max_i*threshold_percentage;

			// compute the allowed current residuals,
			// allow_current = allow_voltage/eff_resist;					
#ifdef _DEBUG
			cout << "max exact voltage: = "  << max(abs(xn1c)) << endl;
			cout << "max exact voltage drop: = "  << max_xn1c << endl;
			cout << "allowed voltage drop (2nd place) = "  << allow_vol_drop << endl;
			cout << "allowed current residual (2nd place) = " << allow_cur_resid << endl;	
#endif  

			for (int j = 0; j < port.size(); j++){
			  sim_port_value.set(j, i, xn1(port(j)));
			}
			if (ir_info){
			  ir_run_time.start();
			  for (int j = 0; j < nNodes; j++){
				if (max_value(j) < xn1(tc_node[j])){
				  max_value(j) = xn1(tc_node[j]);
				}
				if (xn1(tc_node[j]) < min_value(j)){
				  min_value(j) = xn1(tc_node[j]);
				}
				avg_value(j) += xn1(tc_node[j]);
			  }
			  ir_run_time.stop();
			}
			xn_r = XT * xn1;
	  
		  } 
		else
		  {
			compute_sol_time.start();
		
			//xn1 = X * xn1_r;
			if (nport >0){
			  xn1p = Xp * xn1_r;
			  sim_port_value.set_col(i, xn1p);
			}
			if (ir_info){
			  // xn1c = Xc * xn1_r;
			  ir_run_time.start();
			  for (int j = 0; j < nNodes; j++){
				if (max_value(j) < xn1c(j)){
				  max_value(j) = xn1c(j);
				}
				if (xn1c(j) < min_value(j)){
				  min_value(j) = xn1c(j);
				}
				avg_value(j) += xn1c(j);
			  }
			  ir_run_time.stop();
			}
			xn_r = xn1_r;
			compute_sol_time.stop();	
		  }	
	  }
	else
	  {
		compute_sol_time.start();
		
		//xn1 = X * xn1_r;
		if (nport >0){
		  xn1p = Xp * xn1_r;
		  sim_port_value.set_col(i, xn1p);
		}
		if (ir_info){
		  // xn1c = Xc * xn1_r;
		  ir_run_time.start();
		  for (int j = 0; j < nNodes; j++){
			if (max_value(j) < xn1c(j)){
			  max_value(j) = xn1c(j);
			}
			if (xn1c(j) < min_value(j)){
			  min_value(j) = xn1c(j);
			}
			avg_value(j) += xn1c(j);
		  }
		  ir_run_time.stop();
		}
		xn_r = xn1_r;
		compute_sol_time.stop();		
	  }
  }
  delete [] cur;
  delete [] slope;
  cs_dl_spfree(right);
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	//ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
			 << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	cout << "** " << ir_name << " dumped" << endl;
	ir_run_time.stop();
  }

  sim_run_time.stop();

#ifndef UCR_EXTERNAL  
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "initial time     \t: " << init_time.get_time() << std::endl;
  std::cout << "interpolation2   \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "error check1     \t: " << check_run_time1.get_time() << std::endl;
  std::cout << "error check2     \t: " << check_run_time2.get_time() << std::endl;
  std::cout << "solve reduced LU \t: " << solve_red_lu_time.get_time() << std::endl;  
  std::cout << "solve LU         \t: " << solveLU_run_time.get_time() << " \t#" << solveLU_num << std::endl;
  std::cout << "comp sol time    \t: " << compute_sol_time.get_time() << std::endl;
  std::cout << "total simulation \t: " << sim_run_time.get_time() << std::endl;
  std::cout << "IR analysis      \t: " << ir_run_time.get_time() << std::endl;
#endif
}


void reduced_transim2(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
					double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
					mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
					const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
					int num, int ir_info, char *ir_name)
{
  Real_Timer interp2_run_time, solve_red_lu_time;	
  Real_Timer check_run_time1, check_run_time2, solveLU_run_time, sim_run_time;
  Real_Timer init_time, compute_sol_time, ir_run_time;

  sim_run_time.start();

  init_time.start();
  
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, min_ir, avg_ir;
  int max_ir_idx, min_ir_idx;
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
  double min_eff_resist = 1e5;
  double avg_eff_resist = 1e5;  
  double max_eff_resist = 0;  

  UF_long nDim = B->m;
  UF_long nSDim = B->n;
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
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)], slope[nVS+*it]);
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

  double *tmp_xnr=xn_r._data();
  // for(int j=0; j<q; j++)  printf("  x[%d]=%6.4e  cpu\n",j,tmp_xnr[j]); // XXLiu

  /* Transient simulation */
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

  mat CGXc = CGX.get_rows(tc_vec);
  mat CXc = CX.get_rows(tc_vec);

  mat right_r = 1/tstep*Cr;
  mat left_r = Gr + right_r;
  mat l_left_r, u_left_r;
  ivec p_r;
  lu(left_r, l_left_r, u_left_r, p_r);

  vec xn1(n), xn1t(n), xnr(n);;
  xn1.zeros();
  xn1t.zeros();
  xnr.zeros();
  vec xn1_r(q), xn1t_r(q), b(q);
  xn1_r.zeros();
  xn1t_r.zeros();
  b.zeros();
  vec xn1p(nport), xn1c(nNodes);
  xn1p.zeros();
  xn1c.zeros();
  vec w_res(n), w_res1(n);
  w_res.zeros();
  w_res1.zeros();
  vec wc_res(nNodes), wc_res1(nNodes);
  wc_res.zeros();
  wc_res1.zeros();
  vec xn1c_appx(nNodes);
  xn1c_appx.zeros();
  vec resi(n);
  resi.zeros();
  vec resic(nNodes);
  resic.zeros();
  vec x_diff(nNodes);
  x_diff.zeros();
  int solveLU_num = 0;
  double max_voltage_drop = 0;
  double allow_cur_resid = 0;
  double allow_vol_drop = 0;
  double xn1c_max = max(xdcc)>0?max(xdcc):max(abs(xdcc));
  double xn1c_min = min(xdcc)>0?min(xdcc):0;
  double resi_over_u = 2;

  double init_allow_vol_drop = xn1c_min*1e-4;
  allow_vol_drop = init_allow_vol_drop;

  int *var_ii = new int[var_i.size()];
  int nvar_ii = var_i.size();
  for (int i = 0; i < var_i.size(); i++){
	var_ii[i] = var_i[i];
  }

  init_time.stop();

  for (int i = 1; i < ts.size(); i++){
    for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
      interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it], slope[*it]);
      u_col(*it) = temp;
    }
    interp_next_step2(ts[i], IS, var_i, cur, slope, nVS, u_col);
	
    interp2_run_time.stop();

    solve_red_lu_time.start();	
    multiply(Br, u_col._data(), w_r._data());
    w_r1 = right_r * xn_r;
    w_r = w_r + w_r1;
    interchange_permutations(w_r, p_r);
    forward_substitution(l_left_r, w_r, xn1t_r);
    backward_substitution(u_left_r, xn1t_r, xn1_r);
    solve_red_lu_time.stop();
    compute_sol_time.start();
		
    //xn1 = X * xn1_r;
    if (nport >0){
      xn1p = Xp * xn1_r;
      sim_port_value.set_col(i, xn1p);
    }
    xn_r = xn1_r;
    compute_sol_time.stop();
    if (ir_info){
      xn1c = Xc * xn1_r;
      ir_run_time.start();
      for (int j = 0; j < nNodes; j++){
	if (max_value(j) < xn1c(j)){
	  max_value(j) = xn1c(j);
	}
	if (xn1c(j) < min_value(j)){
	  min_value(j) = xn1c(j);
	}
	avg_value(j) += xn1c(j);
      }
      ir_run_time.stop();
    }		
  }
  delete [] cur;
  delete [] slope;
  cs_dl_spfree(right);
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	//ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
			 << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	cout << "** " << ir_name << " dumped" << endl;
	ir_run_time.stop();
  }

  sim_run_time.stop();
 
#ifndef UCR_EXTERNAL 
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "initial time     \t: " << init_time.get_time() << std::endl;
  std::cout << "interpolation2   \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "error check1     \t: " << check_run_time1.get_time() << std::endl;
  std::cout << "error check2     \t: " << check_run_time2.get_time() << std::endl;
  std::cout << "solve reduced LU \t: " << solve_red_lu_time.get_time() << std::endl;  
  std::cout << "solve LU         \t: " << solveLU_run_time.get_time() << " \t#" << solveLU_num << std::endl;
  std::cout << "comp sol time    \t: " << compute_sol_time.get_time() << std::endl;
  std::cout << "total simulation \t: " << sim_run_time.get_time() << std::endl;
  std::cout << "IR analysis      \t: " << ir_run_time.get_time() << std::endl;
#endif
}

