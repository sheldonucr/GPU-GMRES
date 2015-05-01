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
