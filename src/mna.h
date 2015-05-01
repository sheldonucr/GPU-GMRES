/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: mna.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:59 $
 *    Authors: Ning Mi	 
 *
 *    Functions: stamp function
 *
 */


#include "circuit.h"
#include "matrix.h"

/***********************************************
*  class MNA
***********************************************/
class MNA 
{
	circuit* netlist;
	matrix *G, *C, *B;
	int *uIndex;

	int size_G, size_C, col_B, row_B, col_u, row_u;
 
public:
	MNA(circuit* ckt);
	~MNA();
	void stamp();
	void stampG();
	void stampC();
	void stampB();
	void deleteG(){delete G;}
	void deleteC(){delete C;}
	void deleteB(){delete B;}
	// void transform(sparse_mat* sG, sparse_mat* sC, sparse_mat* sB);
	cs* G2cs();
	cs_dl* G2csdl();
	cs* C2cs();
	cs_dl* C2csdl();
	cs* B2cs();
	cs_dl* B2csdl();
};

