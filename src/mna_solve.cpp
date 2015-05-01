/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: mna_solve.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Duo Li
 *
 *    Functions: MNA direct solver using CXSparse
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
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"
#include <vector>
#include <itpp/base/math/min_max.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/sort.h>

using namespace itpp;
using namespace std;

void mna_solve(cs_dl *G, cs_dl *C, cs_dl *B, 
			   Source *VS, int nVS, Source *IS, int nIS, 
			   double tstep, double tstop, const ivec &port, 
			   mat &sim_port_value)
{
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
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
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
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
  cs_dl_ipvec(Symbolic->q, x._data(), xres._data(), n);  
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xres(port(j)));
  }
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
  cs_dl_spfree(left);
  vec xn(n), xnr(n), xn1(n), xn1t(n);
  xn = xres;
  xn1.zeros();
  xn1t.zeros();
  for (int i = 1; i < ts.size(); i++){
	for(int j = 0; j < nVS; j++){
	  interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
	  u_col(j) = temp;
	}
	for(int j = 0; j < nIS; j++){
	  interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
	  u_col(nVS+j) = temp;
	}
	w.zeros();
	cs_dl_gaxpy(B, u_col._data(), w._data());
	xnr.zeros();
	// cs_dl_gaxpy(C, xn._data(), xnr._data());
	// w += 1/tstep*xnr;
	cs_dl_gaxpy(right, xn._data(), xnr._data());
	w += xnr;
	cs_dl_ipvec(Numeric->pinv, w._data(), xn1t._data(), n);
	cs_dl_lsolve(Numeric->L, xn1t._data());
	cs_dl_usolve(Numeric->U, xn1t._data());
	cs_dl_ipvec(Symbolic->q, xn1t._data(), xn1._data(), n);   
	for (int j = 0; j < port.size(); j++){
	  sim_port_value.set(j, i, xn1(port(j)));
	}
	xn = xn1;
  }
  cs_dl_spfree(right);
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  delete [] cur;
}

void mna_solve(cs_dl *G, cs_dl *C, cs_dl *B, 
			   Source *VS, int nVS, Source *IS, int nIS, 
			   double tstep, double tstop, const ivec &port, mat &sim_port_value, 
			   vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info,
			   char *ir_name)
{
  Real_Timer interp2_run_time;
  Real_Timer ir_run_time;

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
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
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
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
  vec x(n);
  x.zeros();
  cs_dls *Symbolic;
  cs_dln *Numeric;
  int order = 2;
  double tol = 1e-14;
  Symbolic = cs_dl_sqr(order, G, 0);
  Real_Timer lufact_time, lusol_time;
  lufact_time.start();
  Numeric = cs_dl_lu(G, Symbolic, tol);
  lufact_time.stop();
  lusol_time.start();
  cs_dl_ipvec(Numeric->pinv, w._data(), x._data(), n);
  cs_dl_lsolve(Numeric->L, x._data());
  cs_dl_usolve(Numeric->U, x._data());
  cs_dl_ipvec(Symbolic->q, x._data(), xres._data(), n);  
  lusol_time.stop();
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xres(port(j)));
  }
  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xres(tc_node[j]);
	  min_value(j) = xres(tc_node[j]);
	  avg_value(j) = xres(tc_node[j]);
	}
	ir_run_time.stop();
  }
  printf("Matrix size: %d\n",n);
  printf("LU factorization time:\t%.2f\n",lufact_time.get_time());
  printf("LU solve time:        \t%.2f\n",lusol_time.get_time());

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
  cs_dl_spfree(left);

  vec xn(n), xnr(n), xn1(n), xn1t(n);
  xn = xres;
  xn1.zeros();
  xn1t.zeros();
  for (int i = 1; i < ts.size(); i++){
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
	interp2_run_time.start();
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	interp2_run_time.stop();
	w.zeros();
	cs_dl_gaxpy(B, u_col._data(), w._data());
	xnr.zeros();
	// cs_dl_gaxpy(C, xn._data(), xnr._data());
	// w += 1/tstep*xnr;
	cs_dl_gaxpy(right, xn._data(), xnr._data());
	w += xnr;
	cs_dl_ipvec(Numeric->pinv, w._data(), xn1t._data(), n);
	cs_dl_lsolve(Numeric->L, xn1t._data());
	cs_dl_usolve(Numeric->U, xn1t._data());
	cs_dl_ipvec(Symbolic->q, xn1t._data(), xn1._data(), n);   
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
	xn = xn1;
  }
  cs_dl_spfree(right);
  cs_dl_sfree(Symbolic);
  cs_dl_nfree(Numeric);
  delete [] cur;

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	/*
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
	*/
	ir_value = max_value - min_value;
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
	ir_run_time.stop();
  }

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation2  \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "IR analysis     \t: " << ir_run_time.get_time() << std::endl;
}
