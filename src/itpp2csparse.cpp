/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: itpp2csparse.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:58 $
 *    Authors: Duo Li
 *
 *    Functions: IT++ format matrix to CSparse format
 *
 */

#include <itpp/base/smat.h>
#include "cs.h"

using namespace itpp;

cs *itpp2csparse(sparse_mat &M)
{
	int nnz = M.nnz();
	int nDim = M.cols();
	cs *T = cs_spalloc(nDim, nDim, nnz, 1, 1);

	for (int c=0; c < nDim; c++){
	  sparse_vec svec = M.get_col(c);
	  for (int p=0; p < svec.nnz(); p++){
		cs_entry(T, svec.get_nz_index(p), c, svec.get_nz_data(p));
	  }
	}
	cs *A = cs_compress(T);
	cs_spfree(T);
	return A;
}
