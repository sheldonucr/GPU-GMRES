/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: matrix.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:59 $
 *    Authors: Ning Mi 
 *
 *    Functions: Header file for matrix class
 *
 */


#ifndef __MAT_H
#define __MAT_H

/**************************************
* class matrix
**************************************/
#include <vector>
#include <itpp/base/smat.h>
#include "cs.h"

#define ROW_NUM 5

using namespace std;
//using namespace itpp;

	struct Entry
	{
		//int i,j;
		int i;
		double value;
		//Entry* rowNext;
		//Entry* colNext;
	  /*bool operator < (const Entry& a, const Entry& b){
	    return a.i<b.i;
	    }*/
	};
class matrix
{
public:

private:
	Entry* compact_data;
	Entry* uncompact_data;

public:
	int colsize,rowsize;
	int nnz;
	
	//Entry** rowIndex;
	Entry** colIndex;
	int* num_per_col;

	matrix(int m, int n);
	//matrix(matrix *A);
	~matrix();

	 
	void pushEntry(int i, int j, double value);
	//void pushEntry(int i, int j, double value, Entry** temp_row, Entry** temp_col);

        void sort();
	//	void compact();
	//void trans(sparse_mat* smatrix); //transform to sparse matrix in itpp
	cs* mat2cs(); //transform matrix to csparse form
	cs_dl* mat2csdl();

	void printmatrix();
	void printmatrix(FILE* fid);

};

bool entry_comp (Entry a, Entry b);
	
#endif
