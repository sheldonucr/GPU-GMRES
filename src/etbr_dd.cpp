/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_dd.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR function with Domain Decomposition solver
 *
 */

#include <iostream>
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
#include "etbr_dd.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"

using namespace itpp;
/*
void etbr_dd(cs_dl *G, cs_dl *C, cs_dl *B, 
			 Source *VS, int nVS, Source *IS, int nIS, 
			 double tstep, double tstop, int q, 
			 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value,
			 int npart, UF_long *part_size, UF_long *node_part, 
			 UF_long *mat_pinv, UF_long *mat_q)
*/
void etbr_dd(cs_dl *B, 
			 Source *VS, int nVS, Source *IS, int nIS, 
			 double tstep, double tstop, int q, 
			 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value,
			 int npart, UF_long *part_size, UF_long *node_part, 
			 UF_long *mat_pinv, UF_long *mat_q)
{

  Real_Timer interp_run_time, fft_run_time, sCpG_run_time;
  Real_Timer form_dd_run_time, dd_solve_run_time;
  Real_Timer umfpack_symbolic, umfpack_numeric, umfpack_solve;
  Real_Timer symbolic_runtime, numeric_runtime, solve_runtime;
  Real_Timer umfpack_run_time, svd_run_time, rmatrix_run_time;
  Real_Timer sim_run_time;

  UF_long nDim = B->m;
  UF_long nSDim = B->n;

  vec ts;
  form_vec(ts, 0, tstep, tstop);
  
  /* Intepolation on sources */
  /*
  interp_run_time.start();
  mat vs(nVS,ts.size());
  mat is(nIS,ts.size());
  vec interp_value(ts.size());

  for (int i = 0; i < nVS; i++){
	interp1(VS[i].time, VS[i].value, ts, interp_value);
	vs.set_row(i, interp_value);
  }
  for (int i = 0; i < nIS; i++){
	interp1(IS[i].time, IS[i].value, ts, interp_value);
	is.set_row(i, interp_value);
  }
  interp_run_time.stop();
  */

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 1.0e-2;
  double f_max = 1/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1.0e4;
  double f_max = 0.5/tstep;
  vec lin_samples = linspace(std::log10(f_min), std::log10(f_max), q);
  vec samples = pow10(lin_samples);

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
  mat us(nVS+nIS, np);
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
  us = concat_vertical(us_v, us_i);
  fft_run_time.stop();
	
  /* use UMFPACK to solve Ax=b */
  mat Z(nDim, np);
  
  double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
  umfpack_dl_defaults(Control);
  // Control[UMFPACK_ALLOC_INIT] = 0.5;
  // Control[UMFPACK_PRL] = 2;
  ifstream in_GC_file;
  ofstream out_GC_file;
  string GC_file_name = "temp/GC_file";
  /*
  out_GC_file.open(GC_file_name.c_str(), ios::binary);
  cs_dl_save(out_GC_file, G);
  cs_dl_save(out_GC_file, C);
  cs_dl_spfree(G);
  cs_dl_spfree(C);
  out_GC_file.close();
  */
  cs_dl *G, *C;
  for (int i = 0; i < np; i++){
	
	sCpG_run_time.start();
	cs_dl *A;
	in_GC_file.open(GC_file_name.c_str(), ios::binary);
	cs_dl_load(in_GC_file, G);
	cs_dl_load(in_GC_file, C);
	in_GC_file.close();
	A = cs_dl_add(G, C, 1, samples(i));
	cs_dl_spfree(G);
	cs_dl_spfree(C);
	sCpG_run_time.stop();

	/* solve Az = b using Domain Decompositon */
	cs_dl **As = new cs_dl*[npart];
	cs_dl **E = new cs_dl*[npart];
	cs_dl **F = new cs_dl*[npart];
	cs_dl *At;
	double **f = new double*[npart];
	double *g;
	cs_dl *A_dd = cs_dl_permute(A, mat_pinv, mat_q, 1);
  
	/*
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
		std::cout << A_mat(r,c) << " " ;
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
		std::cout << A_dd_mat(r,c) << " " ;
	  }
	  std::cout<< std::endl;
	}
	*/

	cs_dl_spfree(A);
	vec b(nDim);
	b.zeros();
	(void) cs_dl_gaxpy(B, us.get_col(i)._data(), b._data());
	double *b_dd = new double[nDim];
	cs_dl_pvec(mat_q, b._data(), b_dd, nDim);
 
	form_dd_run_time.start();
	dd_form(npart, part_size, node_part, A_dd, b_dd, As, E, F, At, f, g);
	form_dd_run_time.stop();

	delete [] b_dd;

	/*
	mat At_mat(At->m, At->n);
	At_mat.zeros();
	for (UF_long j = 0; j < At->n; j++){
	  for (UF_long p = At->p[j]; p < At->p[j+1]; p++){
		At_mat.set(At->i[p], j, At->x[p]);
	  }
	}
	std::cout << "** At = " << std::endl;
	for (int r = 0; r < At_mat.rows(); r++){
	  for (int c = 0; c < At_mat.cols(); c++){
		std::cout << At_mat(r,c) << " " ;
	  }
	  std::cout<< std::endl;
	}
	*/

	cs_dl_spfree(A_dd);
	double *z_dd = new double[nDim];
	dd_solve_run_time.start();
	dd_solve_ooc(npart, As, E, F, At, f, g, z_dd, symbolic_runtime, numeric_runtime, solve_runtime);
	dd_solve_run_time.stop();

	double *z = new double[nDim];
	cs_dl_ipvec(mat_q, z_dd, z, nDim);
	vec zz(z, nDim);
	Z.set_col(i, zz);

	delete [] z;
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
  }

  /* SVD */
  svd_run_time.start();
  mat U, V;
  vec S;
  int info;
  info = svd0(Z, U, S, V);
  X = U.get_cols(0,q-1);
  svd_run_time.stop();

  /* Generate reduced matrices */
  rmatrix_run_time.start();
  in_GC_file.open(GC_file_name.c_str(), ios::binary);
  cs_dl_load(in_GC_file, G);
  cs_dl_load(in_GC_file, C);
  in_GC_file.close();
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
  cs_dl_spfree(G);
  cs_dl_spfree(C);
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
	interp_run_time.start();
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
  std::cout << "form_dd         \t: " << form_dd_run_time.get_time() << std::endl;
  std::cout << "dd_solve        \t: " << dd_solve_run_time.get_time() << std::endl;
  std::cout << "symbolic        \t: " << symbolic_runtime.get_time() << std::endl;
  std::cout << "numeric         \t: " << numeric_runtime.get_time() << std::endl;
  std::cout << "solve           \t: " << solve_runtime.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "simulation      \t: " << sim_run_time.get_time() << std::endl;
}
