/*!	\file
	\brief implement the functions of the class of MySpMatrix
*/

//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>

#include "SpMV.h"

void MySpMatrix::Initilize(SpMatrix &M){
	this->numRows = M.numRows;
	this->numCols = M.numCols;
	this->numNZEntries = M.numNZEntries;

	val = new float[M.numNZEntries];
	indices = new int[M.numNZEntries];
	rowIndices = new int[M.numRows + 1];

	//genCSRFormat(&M, val, rowIndices, indices);
	for (int i = 0; i < numNZEntries; i++) {
		val[i] = (M.nzentries)[i].val;
		indices[i] = (M.nzentries)[i].colNum;
	}
	for (int i = 0; i < numRows; i++) {
		rowIndices[i] = M.rowPtrs[i];
	}
	rowIndices[numRows] = M.numNZEntries;


	cudaMalloc((void**)&d_val, M.numNZEntries * sizeof(float));
	cudaMalloc((void**)&d_indices, M.numNZEntries * sizeof(int));
	cudaMalloc((void**)&d_rowIndices, (M.numRows+1) * sizeof(int));

	cudaMemcpy(d_val, val, M.numNZEntries * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, M.numNZEntries * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIndices, rowIndices, (M.numRows+1) * sizeof(int), cudaMemcpyHostToDevice);

}

// MySpMatrix::~MySpMatrix(){
// 	cudaFree(d_val);
// 	cudaFree(d_indices);
// 	cudaFree(d_rowIndices);
// }


