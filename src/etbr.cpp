/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR function
 *
 */

#include <iostream>
#include <itpp/base/timing.h>
#include <itpp/base/smat.h>
#include <itpp/base/mat.h>
#include <itpp/base/vec.h>
#include <itpp/base/specmat.h>
#include <itpp/base/algebra/lapack.h>
#include <itpp/base/algebra/ls_solve.h>
#include <itpp/base/algebra/svd.h>
#include <itpp/signal/transforms.h>
#include <itpp/base/math/elem_math.h>
#include <itpp/base/math/log_exp.h>
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"

using namespace itpp;

void etbr(sparse_mat &G, sparse_mat &C, sparse_mat &B, 
		  Source *VS, int nVS, Source *IS, int nIS, 
		  double tstep, double tstop, int q, 
		  mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value)
{

  Real_Timer interp_run_time, fft_run_time, sCpG_run_time, col_run_time;
  Real_Timer umfpack_run_time, svd_run_time, rmatrix_run_time;
  Real_Timer sim_run_time;

  int nDim = B.rows();
  int nSDim = B.cols();

  vec ts;
  form_vec(ts, 0, tstep, tstop);
  
  /* Intepolation on sources */
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
  /*
  std::cout << "** for debug ** " << std::endl;
  for (int i = 0; i < ts.size(); i++){
	std::cout << i << "  ";
	for (int j = 0; j < nIS/2; j++){
	  std::cout << is.get_row(j).get(i) << " \t";
	}
	std::cout << std::endl;
  }
  */
  interp_run_time.stop();

#if 0
  /* sampling: uniform in linear scale */
  double f_min = 1.0e-2;
  double f_max = 1/tstep;
  vec samples = linspace(f_min, f_max, q);
#endif 

  /* sampling: uniform in log scale */
  double f_min = 1.0e-2;
  double f_max = 0.5/tstep;
  vec lin_samples = linspace(std::log10(f_min), std::log10(f_max), q);
  vec samples = pow10(lin_samples);

  int np = samples.size();
  //cout<< "# samples: " << np << endl;
  
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
  for (int i = 0; i < nVS; i++){
	spwl_row = fft_real(vs.get_row(i), fft_n);
	spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_v_row);
	us_v.set_row(i, us_v_row);
  }
  for (int i = 0; i < nIS; i++){
	spwl_row = fft_real(is.get_row(i), fft_n);
	spwl_row *= (double)1/fft_n;
	abs_spwl_row = abs(spwl_row(0,floor_i(fft_n/2)));
	abs_spwl_row *= 2;
	interp1(f, abs_spwl_row, samples, us_i_row);
	us_i.set_row(i, us_i_row);
  }
  us = concat_vertical(us_v, us_i);
  fft_run_time.stop();
  // std::cout << "**** FFT finished ****" << std::endl; 
	
  /* use UMFPACK to solve Ax=b */
  // std::cout << "**** Forming Z " << " ****" << std::endl; 
  mat Z(nDim, np);
  sparse_mat A;
  for (int i = 0; i < np; i++){
	
	sCpG_run_time.start();
	A = C;
	A *= samples(i);
	A += G;
	sCpG_run_time.stop();

	/* triple to column-compressed */
	// std::cout << "**** dumping triplet for sample " << i << " ****" << std::endl;
	col_run_time.start();
	int Annz = A.nnz();
	int status;
	double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
	umfpack_di_defaults(Control);
	int *ATi, *ATj, *AAp, *AAi;
	double *ATx, *AAx;
	ATi = new int[Annz];
	ATj = new int[Annz];
	ATx = new double[Annz];
	AAp = new int[nDim+1];
	AAi = new int[Annz];
	AAx = new double[Annz];
	int Aindex = 0;
	for (int c=0; c < nDim; c++){
	  sparse_vec svec = A.get_col(c);
	  for (int p=0; p < svec.nnz(); p++){
		ATi[Aindex] = svec.get_nz_index(p);
		ATj[Aindex] = c;
		ATx[Aindex] = svec.get_nz_data(p);
		Aindex++;
	  }
	}
	
	// std::cout << "**** triplet to col for sample " << i << " ****" << std::endl;
	status = umfpack_di_triplet_to_col (nDim, nDim, Annz, ATi, ATj, ATx, AAp, AAi, AAx, NULL);
	delete [] ATi;
	delete [] ATj;
	delete [] ATx;

	col_run_time.stop();
	// std::cout << "**** sovling sample " << i << " by UMFPACK ****" << std::endl;

	/* LU decomposition */
	umfpack_run_time.start();
	void *Symbolic, *Numeric;
	if (i == 0){
		(void) umfpack_di_symbolic (nDim, nDim, AAp, AAi, AAx, &Symbolic, Control, Info);
	}
	(void) umfpack_di_numeric (AAp, AAi, AAx, Symbolic, &Numeric, Control, Info);
	if (i == np-1){
		umfpack_di_free_symbolic (&Symbolic);
	}

	/* solve Az = b  */
	double* z = new double[nDim];
	vec b = B*us.get_col(i);
	(void) umfpack_di_solve (UMFPACK_A, AAp, AAi, AAx, z, b._data(), Numeric, Control, Info);			
	vec zz(z, nDim);
	delete [] z;
	Z.set_col(i, zz);	
	delete [] AAp;
    delete [] AAi;
  	delete [] AAx;
	umfpack_di_free_numeric (&Numeric); 
	umfpack_run_time.stop();
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
  Gr.set_size(q, nDim, true);
  for (int i = 0; i < X.cols(); i++){
	Gr.set_row(i, X.get_col(i)*G);
  }
  Gr *= X;
  //Cr = X.T()*C*X; 
  Cr.set_size(q, nDim, true);
  for (int i = 0; i < X.cols(); i++){
	Cr.set_row(i, X.get_col(i)*C);
  }
  Cr *= X;
  //Br = X.T()*B;
  Br.set_size(q, nSDim, true);
  for (int i = 0; i < X.cols(); i++){
	Br.set_row(i, X.get_col(i)*B);
  }
  rmatrix_run_time.stop();

  sim_run_time.start();
  /* Get w */
  mat u = concat_vertical(vs, is);
  mat w(q,ts.size());
  for (int i = 0; i < u.cols();i++){
	w.set_col(i, Br*u.get_col(i));
  }
  /* DC simulation */
  vec xres;
  xres = ls_solve(Gr, w.get_col(0));
  sim_value.set_size(q, ts.size());
  sim_value.set_col(0, xres);

  /* Transient simulation */
  mat right = 1/tstep*Cr;
  mat left = Gr + right;
  vec xn(q), xn1(q);
  xn.zeros();
  xn1.zeros();
  for(int i = 1; i < ts.size(); i++){
	xn1 = ls_solve(left, right*xn + w.get_col(i));
	sim_value.set_col(i, xn1);
	xn = xn1;
  }
  sim_run_time.stop();
  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "Interpolation   \t: " << interp_run_time.get_time() << std::endl;
  std::cout << "FFT             \t: " << fft_run_time.get_time() << std::endl;
  std::cout << "sC+G            \t: " << sCpG_run_time.get_time() << std::endl;
  std::cout << "col compressed  \t: " << col_run_time.get_time() << std::endl;
  std::cout << "umfpack_solve   \t: " << umfpack_run_time.get_time() << std::endl;
  std::cout << "SVD             \t: " << svd_run_time.get_time() << std::endl;
  std::cout << "reduce matrices \t: " << rmatrix_run_time.get_time() << std::endl;
  std::cout << "simulation      \t: " << sim_run_time.get_time() << std::endl;
}
