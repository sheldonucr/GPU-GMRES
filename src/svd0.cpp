/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: svd0.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:01 $
 *    Authors: Duo Li
 *
 *    Functions: Economic SVD 
 *
 */

#include "svd0.h"

bool svd0(const mat &A, mat &U, vec &S, mat &V)
{
	char jobu = 'S', jobvt = 'S';
	int m, n, lda, ldu, ldvt, lwork, info;
	m = lda = ldu = A.rows();
	n = ldvt = A.cols();
	lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));

	U.set_size(m, n, false);
	V.set_size(n, n, false);
	S.set_size(std::min(m, n), false);
	vec work(lwork);
	mat B(A);

	// The theoretical calculation of lwork above results in the minimum size
	// needed for dgesvd_ to run to completion without having memory errors.
	// For speed improvement it is best to set lwork=-1 and have dgesvd_
	// calculate the best workspace requirement.
	int lwork_tmp = -1;
	dgesvd_(&jobu, &jobvt, &m, &n, B._data(), &lda, S._data(), U._data(), &ldu,
					V._data(), &ldvt, work._data(), &lwork_tmp, &info);
	if (info == 0) {
			lwork = static_cast<int>(work(0));
			work.set_size(lwork, false);
	}

	dgesvd_(&jobu, &jobvt, &m, &n, B._data(), &lda, S._data(), U._data(), &ldu,
					V._data(), &ldvt, work._data(), &lwork, &info);

	V = V.T(); // This is probably slow!!!

	return (info == 0);
}
