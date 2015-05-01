/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: simulation.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:01 $
 *    Authors: Duo Li
 *
 *    Functions: Simulation
 *
 */


using namespace itpp;

void simulate(sparse_mat &G, sparse_mat &C, sparse_mat &B, 
			  Source *VS, int nVS, Source *IS, int nIS, 
			  double tstep, double tstop, mat &sim_value)
{

  int nDim = B.rows();
  int nSDim = B.cols();
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
  u = concat_vertical(vs, is);
  
  mat w(nDim,ts.size());
  for (int i = 0; i < u.cols();i++){
	w.set_col(i, B*u.get_col(i));
  }

  /* DC simulation */
  vec xres;
  xres = ls_solve(G, w.get_col(0));
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
}
