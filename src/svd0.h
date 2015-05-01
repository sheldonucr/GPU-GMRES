/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: svd0.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:01 $
 *    Authors: Duo Li
 *
 *    Functions: Economic SVD header
 *
 */

#ifndef SVD0_H
#define SVD0_H

#include <itpp/base/vec.h>
#include <itpp/base/mat.h>
#include <itpp/base/algebra/lapack.h>

using namespace itpp;

bool svd0(const mat &A, mat &U, vec &S, mat &V);
	
#endif
