/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#include "SpMV_kernel.h"


////////////////////////////////////////////////////////////////////////////////
// SpMV Kernel Device Code
////////////////////////////////////////////////////////////////////////////////
	__global__ void 
SpMV(float *x,
		const float *val, const  int *rowIndices, const  int *indices,
		const float *y,
		const  int numRows, const  int numCols, const  int numNonZeroElements)
{
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;

	// +++++++++++++++++++++++++++++++++++++++++++++
	// not run
#if C_GLOBAL_OPT
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int ub,lb;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	t=0;
	int rowStartNZ = rowInd[ind2Dy];
	int rowStartRes = rowStartNZ%HALFWARP;
	int rowStartAlignNZ = rowStartNZ + HALFWARP - rowStartRes;
	lb = rowStartAlignNZ+ind2Dx;
	ub = rowInd[ind2Dy+1];
	int j ;
	if (myi < numRows) {
		j = rowStartAlignNZ - HALFWARP + ind2Dx;
		if ( (j >= rowStartNZ) && (j<ub) ) { 
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		for (j=lb; j<ub; j+=HALFWARP) {
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		tempProd[ind2Dy][ind2Dx] = t;
	}
	__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	__syncthreads();
#endif

	// no C_GLOBAL_OPT

#endif // C_GLOBAL_OPT
	// +++++++++++++++++++++++++++++++++++++++++++++

	// ---------------------------------------------
	// not run
#if NEW_GLOBAL_OPT
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int ub,lb;
	int blockStart = bid * (BLOCKSIZE/HALFWARP);
	int blockEnd = min(blockStart+(BLOCKSIZE/HALFWARP)-1,numRows-1);
	int myi = blockStart + ind2Dy;

	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && ((blockStart+tid) <= numRows))
		rowInd[tid] = rowIndices[blockStart+tid];
	__syncthreads();

	t=0;
	lb = rowInd[0]+tid;
	ub = rowInd[blockEnd-blockStart+1];
	int curr_i=0;
	for (int p=0;p<(BLOCKSIZE/HALFWARP);p++)
		tempProd[p][ind2Dy][ind2Dx] = 0;
	__syncthreads();
	for ( int j=lb; j<ub; j+=NUMTHREADS) {
		for (int p=curr_i;p<(blockEnd-blockStart+1);p++) {
			if ( (j >= rowInd[p]) && (j < rowInd[p+1]) ) {
				curr_i = p;
				break;
			}
		}
		int ind = indices[j];
#if CACHE
		float yval = tex1Dfetch(tex_y_float, ind);
#else
		float yval = y[ind];
#endif
		t = val[j] * yval;
		tempProd[curr_i][ind2Dy][ind2Dx] += t;
	}
	__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<(BLOCKSIZE/HALFWARP); k++) {
			for ( int l = 0; l<HALFWARP; l++) 
				t += tempProd[ind2Dy][k][l];
		}
		x[myi] = t;
	}
#endif
#if 1
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<(BLOCKSIZE/HALFWARP); k++) {
			t += tempProd[ind2Dy][k][0] + tempProd[ind2Dy][k][1] + tempProd[ind2Dy][k][2] + tempProd[ind2Dy][k][3] +\
				 tempProd[ind2Dy][k][4] + tempProd[ind2Dy][k][5] + tempProd[ind2Dy][k][6] + tempProd[ind2Dy][k][7] +\
				 tempProd[ind2Dy][k][8] + tempProd[ind2Dy][k][9] + tempProd[ind2Dy][k][10]+ tempProd[ind2Dy][k][11]+\
				 tempProd[ind2Dy][k][12]+ tempProd[ind2Dy][k][13]+ tempProd[ind2Dy][k][14]+ tempProd[ind2Dy][k][15];
		}
		x[myi] = t;
	}
#endif

	__syncthreads();

#endif// end of NEW_GLOBAL_OPT


	// ---------------------------------------------

	// *********************************************
	// will run
#if GLOBAL_OPT
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int ub,lb;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows)) 
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	t=0;
	lb = rowInd[ind2Dy]+ind2Dx;
	ub = rowInd[ind2Dy+1];
	if (myi < numRows) {
		for ( int j=lb; j<ub; j+=HALFWARP) {
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		tempProd[ind2Dy][ind2Dx] = t;
	}
	//__syncthreads();
#if 0 
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=8
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7];
		x[myi] = t;
	}
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	//__syncthreads();
#endif
	// XXLiu: finished
#if 0
	// Works for HALFWARP=16/32
	if (myi < numRows) {
		//if (ind2Dx < 16) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+16];
		if (ind2Dx < 8) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+8];
		if (ind2Dx < 4) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+4];
		if (ind2Dx < 2) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+2];
		if (ind2Dx < 1) x[myi]= tempProd[ind2Dy][ind2Dx] +tempProd[ind2Dy][ind2Dx+1];
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=16 & 32
	if (!(ind2Dx % 4) && (myi < numRows)) {
		tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+1] + tempProd[ind2Dy][ind2Dx+2] + tempProd[ind2Dy][ind2Dx+3];
	}
	__syncthreads();
	if ((ind2Dx == 0) && (myi < numRows)) {
#if 1 // for halfwarp 16
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12];
#else // for halfwarp 32
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12]+\
			+tempProd[ind2Dy][16] + tempProd[ind2Dy][20] + tempProd[ind2Dy][24] + tempProd[ind2Dy][28];
#endif
		x[myi] = t;
	}
	__syncthreads();
#endif
#endif

	// *********************************************

	// #############################################
	// not run
#if GLOBAL
	int myi = bid * BLOCKSIZE + tid;
	if (myi < numRows) {
		int lb = rowIndices[myi];
		int ub = rowIndices[myi+1];
		for ( int j=lb; j<ub; j++) {
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
	// #############################################

	// +++++++++++++++++++++++++++++++++++++++++++++
	// not run
#if SHARED_RI
	int myi = bid * BLOCKSIZE + tid;
	__shared__ int rowInd[NUMTHREADS+1];
	if (myi < numRows) {
		rowInd[tid] = rowIndices[myi];
	}
	__syncthreads();
	if ( (myi < numRows) && ((tid == NUMTHREADS-1) || (myi == numRows-1) ) ) {
		rowInd[tid+1] = rowIndices[myi+1];
	}
	__syncthreads();
	if (myi < numRows) {
		int lb = rowInd[tid];
		int ub = rowInd[tid+1];
		for ( int j=lb; j<ub; j++) {
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
	// +++++++++++++++++++++++++++++++++++++++++++++

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	// not run
#if GLOBAL_SHARED_OPT
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[NUMTHREADS];
	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	t=0;
	int ind, lb, ub, j;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		if (j<ub)	ind = indices[j];
		else ind = numCols+BLOCKSIZE;
	}
	__syncthreads();
	for ( int k=0; k<numCols; k+=BLOCKSIZE) {
		__syncthreads();
		if ( (k+tid) < numCols)
			ys[tid] = y[k+tid];
		__syncthreads();

		if (myi < numRows) {
#if 0		
			while ( ((j+HALFWARP) < ub) && ( ind < (k+BLOCKSIZE) )  ) {
				t += val[j] * ys[ind-k];
				j+=HALFWARP;
				ind = indices[j];
			}
#endif
#if 1
			while ( ind < (k+BLOCKSIZE) ) {
				t += val[j] * ys[ind-k];
				j+=HALFWARP;
				if (j < ub) ind = indices[j];
				else { ind = numCols+BLOCKSIZE; break; }
			}
#endif
		}
	}
#if 0
	if ( (myi < numRows) && (j<ub) )
		tempProd[ind2Dy][ind2Dx] = t + val[j] * y[ind];
	else
		tempProd[ind2Dy][ind2Dx] = t;
#endif
#if 1
	tempProd[ind2Dy][ind2Dx] = t;
#endif
	__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16 & 32
	if (!(ind2Dx % 4) && (myi < numRows)) {
		tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+1] + tempProd[ind2Dy][ind2Dx+2] + tempProd[ind2Dy][ind2Dx+3];
	}
	__syncthreads();
	if ((ind2Dx == 0) && (myi < numRows)) {
#if 1 // for halfwarp 16
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12];
#else // for halfwarp 32
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12]+\
			+tempProd[ind2Dy][16] + tempProd[ind2Dy][20] + tempProd[ind2Dy][24] + tempProd[ind2Dy][28];
#endif
		x[myi] = t;
	}
	__syncthreads();
#endif
#endif
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}



////////////////////////////////////////////////////////////////////////////////
// SpMV Kernel Device Code
// simplified version by zky, deleted the useless compile branches
// for the easier of reading
////////////////////////////////////////////////////////////////////////////////
	__global__ void
SpMV_s(float *x,
		const float *val, const  int *rowIndices, const  int *indices,
		const float *y,
		const  int numRows, const  int numCols, const  int numNonZeroElements)
{
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;

	// *********************************************
	// will run
#if GLOBAL_OPT
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int ub,lb;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;

	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows)) 
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();

	t=0;
	lb = rowInd[ind2Dy]+ind2Dx;
	ub = rowInd[ind2Dy+1];
	if (myi < numRows) {
		for ( int j=lb; j<ub; j+=HALFWARP) {
			int ind = indices[j];
#if CACHE
			float yval = tex1Dfetch(tex_y_float, ind);
#else
			float yval = y[ind];
#endif
			t += val[j] * yval;
		}
		tempProd[ind2Dy][ind2Dx] = t;
	}
	//__syncthreads();
#if 0 
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=8
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7];
		x[myi] = t;
	}
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	//__syncthreads();
#endif
	// XXLiu: finished
#if 0
	// Works for HALFWARP=16/32
	if (myi < numRows) {
		//if (ind2Dx < 16) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+16];
		if (ind2Dx < 8) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+8];
		if (ind2Dx < 4) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+4];
		if (ind2Dx < 2) tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+2];
		if (ind2Dx < 1) x[myi]= tempProd[ind2Dy][ind2Dx] +tempProd[ind2Dy][ind2Dx+1];
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=16 & 32
	if (!(ind2Dx % 4) && (myi < numRows)) {
		tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+1] + tempProd[ind2Dy][ind2Dx+2] + tempProd[ind2Dy][ind2Dx+3];
	}
	__syncthreads();
	if ((ind2Dx == 0) && (myi < numRows)) {
#if 1 // for halfwarp 16
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12];
#else // for halfwarp 32
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12]+\
			+tempProd[ind2Dy][16] + tempProd[ind2Dy][20] + tempProd[ind2Dy][24] + tempProd[ind2Dy][28];
#endif
		x[myi] = t;
	}
	__syncthreads();
#endif
#endif

	// *********************************************

}




/*-------------------------------------------------------
  SpMV with Inspect Input
  -------------------------------------------------------*/
	__global__ void
SpMV_withInspectInput(float *x, const float *val, const  int *rowIndices,
		const  int *indices, const float *y, const  int numRows,
		const  int numCols, const  int numNonZeroElements,
		const  int *ins_rowIndices, const  int *ins_indices, const  int *ins_inputList)
{
#if CACHE
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_INPUT_MAX];
	__shared__ int ins_rowInd[2];
	__shared__ int ins_Ind[BLOCKSIZE]; // Have to fix
	__shared__ int ins_inpStartVal[BLOCKSIZE]; // Have to fix

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows))
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();

	t=0;
	int lb = rowInd[ind2Dy]+ind2Dx;
	int ub = rowInd[ind2Dy+1];
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	if (tid <= (ktub-ktlb)) ins_Ind[tid]=ins_indices[tid+ktlb];
	__syncthreads();
	if (tid < (ktub-ktlb)) {
		int is = ins_Ind[tid];
		ins_inpStartVal[tid]=ins_inputList[is];
	}
	__syncthreads();
	int kt=ktlb;
	int j=lb;
	for (;kt<ktub;kt++) {
		int startVal = ins_inpStartVal[kt-ktlb];
		if (startVal != numCols) {

			int is = ins_Ind[kt-ktlb];
			int ie = ins_Ind[kt-ktlb+1];
			for ( int iL=is+tid;iL<ie;iL+=NUMTHREADS) {
				int currInd = ins_inputList[iL];
				ys[currInd-startVal] = tex1Dfetch(tex_y_float, currInd);	
			}
			__syncthreads();
			if (myi < numRows && j<ub) {
				int ind = indices[j];
				t += val[j] * ys[ind-startVal];
				j+=HALFWARP;
			}
		}
		else {
			if (myi < numRows && j<ub) {
				int ind = indices[j];
				float yval = tex1Dfetch(tex_y_float, ind);
				t += val[j] * yval;
				j+=HALFWARP;
			}
		}
	}
	tempProd[ind2Dy][ind2Dx] = t;
	// __syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	// __syncthreads();
#endif
#endif
}


/*-------------------------------------------------------
  SpMV with Inspect
  -------------------------------------------------------*/
	__global__ void
SpMV_withInspect(float *x, const float *val, const  int *rowIndices,
		const  int *indices, const float *y, const  int numRows,
		const  int numCols, const  int numNonZeroElements,
		const  int *ins_rowIndices, const  int *ins_indices)
{
#if C_GLOBAL_OPT
#if 0 
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_BLOCK_c];
	__shared__ int ins_rowInd[2];
	__shared__ int ins_Ind;

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows))
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();

	t=0;
	int ind, lb, ub, j;
	float valS;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		ind = indices[j];
		valS = val[j]; 
	}
	__syncthreads();
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	for ( int kt=ktlb; kt<ktub; kt++) {
		__syncthreads();
		if (tid==0) ins_Ind=ins_indices[kt];
		__syncthreads();
#if VAR_BLOCK
		int k = ins_Ind; // In case of var_block, ins_indices 'll have original column index
#else
		int k = ins_Ind*INSPECT_BLOCK_c;
#endif
		if ( tid < min(INSPECT_BLOCK_c, numCols-k) )
#if CACHE
			ys[tid] = tex1Dfetch(tex_y_float, k+tid);
#else
		ys[tid] = y[k+tid];
#endif
		__syncthreads();
		if (myi < numRows) {
			while (ind < (k+INSPECT_BLOCK_c)) {
				t += valS * ys[ind-k];
				j+=HALFWARP;
				if (j < ub) { ind = indices[j]; valS = val[j]; }
				else { ind = 2*numCols;  }
			}
		}
	}
	tempProd[ind2Dy][ind2Dx] = t;
	__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	__syncthreads();
#endif
#endif
#if 1 
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_BLOCK_c];
	__shared__ int ins_rowInd[2];
	//__shared__ int ins_Ind;
	__shared__ int ins_Ind[BLOCKSIZE];

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	//__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows))
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();

	t=0;
	int ind, lb, ub, j;
	float valS;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		ind = indices[j];
		valS = val[j]; 
	}
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	if (tid < (ktub-ktlb)) ins_Ind[tid]=ins_indices[tid+ktlb];
	__syncthreads();

	for ( int kt=ktlb; kt<ktub; kt++) {
#if VAR_BLOCK
		int k = ins_Ind[kt-ktlb]; // In case of var_block, ins_indices 'll have original column index
#else
		int k = ins_Ind[kt-ktlb]*INSPECT_BLOCK_c;
#endif
		if ( tid < min(INSPECT_BLOCK_c, numCols-k) )
#if CACHE
			ys[tid] = tex1Dfetch(tex_y_float, k+tid);
#else
		ys[tid] = y[k+tid];
#endif
		__syncthreads();
		if (myi < numRows) {
			while (ind < (k+INSPECT_BLOCK_c)) {
				t += valS * ys[ind-k];
				j+=HALFWARP;
				if (j < ub) { ind = indices[j]; valS = val[j]; }
				else { ind = 2*numCols;  }
			}
		}
		//__syncthreads();
	}

	tempProd[ind2Dy][ind2Dx] = t;
	//__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	//__syncthreads();
#endif
#if 0
	// Works for HALFWARP=8
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7];
		x[myi] = t;
	}
	//__syncthreads();
#endif
#endif
#if 0 
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_BLOCK_c];
	__shared__ int ins_rowInd[2];
	//__shared__ int ins_Ind;
	__shared__ int ins_Ind[BLOCKSIZE];

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows))
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();

	t=0;
	int ind, lb, ub, j;
	float valS;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		ind = indices[j];
		valS = val[j]; 
	}
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	if (tid < (ktub-ktlb)) ins_Ind[tid]=ins_indices[tid+ktlb];
	__syncthreads();

	int kt=ktlb;
	if((ktub-ktlb)%2) {
#if VAR_BLOCK
		int k1 = ins_Ind[kt-ktlb]; // In case of var_block, ins_indices 'll have original column index
#else
		int k1 = ins_Ind[kt-ktlb]*INSPECT_BLOCK_c;
#endif
		if ( tid < min(INSPECT_BLOCK_c, numCols-k1) )
			ys[tid] = y[k1+tid];
		__syncthreads();
		if (myi < numRows) {
			while (ind < (k1+INSPECT_BLOCK_c)) {
				t += valS * ys[ind-k1];
				j+=HALFWARP;
				if (j < ub) { ind = indices[j]; valS = val[j]; }
				else { ind = 2*numCols;  }
			}
		}
		//__syncthreads();
		kt++;
	}
	for (; kt<ktub; kt+=2) {
#if VAR_BLOCK
		int k1 = ins_Ind[kt-ktlb]; // In case of var_block, ins_indices 'll have original column index
		int k2 = ins_Ind[kt-ktlb+1]; // In case of var_block, ins_indices 'll have original column index
#else
		int k1 = ins_Ind[kt-ktlb]*INSPECT_BLOCK_c;
		int k2 = ins_Ind[kt-ktlb+1]*INSPECT_BLOCK_c;
#endif
		if ( tid < min(INSPECT_BLOCK_c, numCols-k1) )
			ys[tid] = y[k1+tid];
		__syncthreads();
		if (myi < numRows) {
			while (ind < (k1+INSPECT_BLOCK_c)) {
				t += valS * ys[ind-k1];
				j+=HALFWARP;
				if (j < ub) { ind = indices[j]; valS = val[j]; }
				else { ind = 2*numCols;  }
			}
		}
		//__syncthreads();
		if ( tid < min(INSPECT_BLOCK_c, numCols-k2) )
			ys[tid] = y[k2+tid];
		__syncthreads();
		if (myi < numRows) {
			while (ind < (k2+INSPECT_BLOCK_c)) {
				t += valS * ys[ind-k2];
				j+=HALFWARP;
				if (j < ub) { ind = indices[j]; valS = val[j]; }
				else { ind = 2*numCols;  }
			}
		}
		//__syncthreads();
	}

	tempProd[ind2Dy][ind2Dx] = t;
	//__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	//__syncthreads();
#endif
#if 0
	// Works for HALFWARP=8
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7];
		x[myi] = t;
	}
	//__syncthreads();
#endif
#endif
#if 0
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_BLOCK_c];
	__shared__ int ins_rowInd[2];
	__shared__ int ins_Ind;

	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	//__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows))
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();

	t=0;
	int ind, lb, ub, j;
	float valS;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		//ind = indices[j];
		//valS = val[j];
	}
	__syncthreads();
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	for ( int kt=ktlb; kt<ktub; kt++) {
		//__syncthreads();
		if (tid==0) ins_Ind=ins_indices[kt];
		__syncthreads();
#if VAR_BLOCK
		int k = ins_Ind; // In case of var_block, ins_indices 'll have original column index
#else
		int k = ins_Ind*INSPECT_BLOCK_c;
#endif
		if ( tid < min(INSPECT_BLOCK_c, numCols-k) )
			ys[tid] = y[k+tid];
		__syncthreads();
		if (myi < numRows) {
			//while (ind < (k+INSPECT_BLOCK_c)) {
			ind = indices[j]; valS = val[j];
			t += valS * ys[ind-k];
			j+=HALFWARP;
			//if (j < ub) { ind = indices[j]; valS = val[j]; }
			//else { ind = 2*numCols;  }
			//}
		}
	}
	tempProd[ind2Dy][ind2Dx] = t;
	//__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
#endif
#if 0
	// Works for HALFWARP=8
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7];
		x[myi] = t;
	}
	//__syncthreads();
#endif
#endif
#else
	int tid = threadIdx.y;
	int bid = blockIdx.y;
	float t=0;
	int ind2Dx = tid%HALFWARP;
	int ind2Dy = tid/HALFWARP;
	int myblock = bid * (BLOCKSIZE/HALFWARP);
	int myi = myblock + ind2Dy;
	__shared__ int rowInd[(BLOCKSIZE/HALFWARP)+1];
	__shared__ float tempProd[(BLOCKSIZE/HALFWARP)][HALFWARP+PAD];
	__shared__ float ys[INSPECT_BLOCK_c];
	__shared__ int ins_rowInd[2];
	__shared__ int ins_Ind;
	if ((tid <= ((BLOCKSIZE/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock+tid];
	__syncthreads();
	if ((ind2Dx < 2) && (ind2Dy == 0) && (myi < numRows)) 
		ins_rowInd[ind2Dx]=ins_rowIndices[bid+ind2Dx];
	__syncthreads();
	t=0;
	int ind, lb, ub, j;
	if (myi < numRows) {
		lb = rowInd[ind2Dy]+ind2Dx;
		ub = rowInd[ind2Dy+1];
		j = lb;
		if (j<ub) ind = indices[j]; 
		else ind = numCols+INSPECT_BLOCK_c;
	}
	__syncthreads();
	int ktlb = ins_rowInd[0];
	int ktub = ins_rowInd[1];
	for ( int kt=ktlb; kt<ktub; kt++) {
		__syncthreads();
		if (tid==0) ins_Ind=ins_indices[kt];
		__syncthreads();
#if VAR_BLOCK
		int k = ins_Ind; // In case of var_block, ins_indices 'll have original column index
#else
		int k = ins_Ind*INSPECT_BLOCK_c;
#endif
		if ( (tid < INSPECT_BLOCK_c) && ((k+tid) < numCols) )
			ys[tid] = y[k+tid];
		__syncthreads();
		if (myi < numRows) {
			while ( ind < (k+INSPECT_BLOCK_c) ) {
				t += val[j] * ys[ind-k];
				j+=HALFWARP;
				if (j < ub) ind = indices[j];
				else { ind = numCols+INSPECT_BLOCK_c; break; }
			}
		}
	}
	tempProd[ind2Dy][ind2Dx] = t;
	__syncthreads();
#if 0
	if ((ind2Dx == 0) && (myi < numRows)) {
		t=0;
		for ( int k = 0; k<HALFWARP; k++) {
			t += tempProd[ind2Dy][k];
		}
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows)) {
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] + tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +\
			tempProd[ind2Dy][4] + tempProd[ind2Dy][5] + tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +\
			tempProd[ind2Dy][8] + tempProd[ind2Dy][9] + tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+\
			tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+ tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
		x[myi] = t;
	}
	__syncthreads();
#endif
#if 0
	// Works for HALFWARP=16 & 32
	if (!(ind2Dx % 4) && (myi < numRows)) {
		tempProd[ind2Dy][ind2Dx] += tempProd[ind2Dy][ind2Dx+1] + tempProd[ind2Dy][ind2Dx+2] + tempProd[ind2Dy][ind2Dx+3];
	}
	__syncthreads();
	if ((ind2Dx == 0) && (myi < numRows)) {
#if 1 // for halfwarp 16
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12];
#else // for halfwarp 32
		t = tempProd[ind2Dy][0] + tempProd[ind2Dy][4] + tempProd[ind2Dy][8] + tempProd[ind2Dy][12]+\
			+tempProd[ind2Dy][16] + tempProd[ind2Dy][20] + tempProd[ind2Dy][24] + tempProd[ind2Dy][28];
#endif
		x[myi] = t;
	}
	__syncthreads();
#endif
#endif
}


