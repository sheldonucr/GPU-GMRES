/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: itpp_operations.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:58 $
 *    Authors: Duo Li
 *
 *    Functions: ITPP Operations
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
#include "umfpack.h"
#include "etbr.h"
#include "interp.h"
#include "svd0.h"
#include "cs.h"

using namespace itpp;
using namespace std;


void multiply(mat& a, vec& x, vec& b)
{
  int i, k, a_pos, m, n;
  double *a_data = a._data();
  double *b_data = b._data();
  double *x_data = x._data();
  m = a.rows();
  n = a.cols();
  //b.set_size(m);
  for (i = 0; i < m; i++) {
	b_data[i] = 0;
	a_pos = 0;
	for (k = 0; k < n; k++) {
	  b_data[i] += a_data[a_pos+i] * x_data[k];
	  a_pos += m;
	}
  }
}

void multiply(mat& a, double* x, double* b)
{
  int i, k, a_pos, m, n;
  double *a_data = a._data();
  m = a.rows();
  n = a.cols();
  for (i = 0; i < m; i++) {
	b[i] = 0;
	a_pos = 0;
	for (k = 0; k < n; k++) {
	  b[i] += a_data[a_pos+i] * x[k];
	  a_pos += m;
	}
  }
}
