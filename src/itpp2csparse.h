/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: itpp2csparse.h,v $
 *    $Revision: 1.2 $
 *    $Date: 2011/12/06 02:25:43 $
 *    Authors: Duo Li
 *
 *    Functions: IT++ format matrix to CSparse format header
 *
 */

#ifndef ITPP2CSPARSE_H
#define ITPP2CSPARSE_H

#include <itpp/base/smat.h>
#include "cs.h"

using namespace itpp;
cs *itpp2csparse(sparse_mat &M);

#endif
