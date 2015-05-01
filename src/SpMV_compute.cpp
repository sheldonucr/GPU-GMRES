/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 *  (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include "SpMV.h"
#include "defs.h"

using namespace std;

// CPU version --- Naive SpMV code
// XXLiu: modified a little (Refer to Fig.3 in the paper for original version)
void computeSpMV(float *x, const float *val,
		const int *rowIndices, const int *indices, 
		const float *y, const int numRows)
{
	float t;
	int i, j, lb, ub, ind;
	for (i=0; i<numRows; i++) {
		t = 0.0;
		lb = rowIndices[i]; 
		ub = rowIndices[i+1];
		//printf("i=%d\n",i);
		for (j=lb; j<ub; j++) {
			ind = indices[j];
			t += val[j] * y[ind];
			//printf("val[%2d]=%f  ind=%d\n",j,val[j],ind);//XXLiu
		}
		x[i] = t;
	}
}

void addTwoVec2(const float *v1, float *v2, const int num){
	for(int i=0; i<num; ++i){
		v2[i] += v1[i];
	}
}

// solve the equation of L*U*x = y, where the L and U matrix are stored in (val, rowIndices, indices)
void LUSolve(float *x, 
		const float *l_val, const int *l_rowIndices, const int *l_indices,
		const float *u_val, const int *u_rowIndices, const int *u_indices,
		const float *y, const int numRows)
{
	float* v = (float*)malloc(numRows*sizeof(float));
	memcpy(v, y, numRows*sizeof(float));
	// solve Lv = y, forward substitution
	for(int i=0; i<numRows; ++i){
		int lb = l_rowIndices[i];
		int ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
		int j=lb;
		for(; j<ub; ++j){
			if(l_indices[j] >= i){
				break;
			}
			else{
				v[i] -= l_val[j] * v[l_indices[j]];
			}
		}
		assert(l_indices[j] == i);// int the L matrix, the location of the diagonal element
		assert(!Equal(l_val[j], 0));
		x[i] /= l_val[j];
	}

	// sovle Ux = v, backward substitution
	memcpy(x, v, numRows*sizeof(float));
	for(int i=numRows-1; i>=0; --i){
		int lb = u_rowIndices[i];// lb is the low bound for the L, not U
		int ub = u_rowIndices[i+1];
		int j=ub - 1;
		for(; j>=lb; --j){
			if(u_indices[j] <= i){// search to the L matrix
				break;
			}
			else{
				x[i] -= u_val[j] * x[u_indices[j]];
			}
		}
		assert(u_indices[j] == i);// in the U matrix, the element in the diagonal should not be zero
		x[i] /= u_val[j];
	}
}


// solve the equation of L*U*x = y, where the L and U matrix are stored in (val, rowIndices, indices)
void LUSolve_ignoreZero(float *x, 
		const float *l_val, const int *l_rowIndices, const int *l_indices,
		const float *u_val, const int *u_rowIndices, const int *u_indices,
		const float *y, const int numRows)
{
	float* v = (float*)malloc(numRows*sizeof(float));
	memcpy(v, y, numRows*sizeof(float));
	// solve Lv = y, forward substitution
	for(int i=0; i<numRows; ++i){
		int lb = l_rowIndices[i];
		int ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
		int j=lb;
		for(; j<ub; ++j){
			if(l_indices[j] >= i){
				break;
			}
			else{
				v[i] -= l_val[j] * v[l_indices[j]];
			}
		}
		assert(l_indices[j] == i);// int the L matrix, the location of the diagonal element
		assert(!Equal(l_val[j], 0));
		x[i] /= l_val[j];
	}

	// sovle Ux = v, backward substitution
	memcpy(x, v, numRows*sizeof(float));
	for(int i=numRows-1; i>=0; --i){
		int lb = u_rowIndices[i];// lb is the low bound for the L, not U
		int ub = u_rowIndices[i+1];
		int j=ub - 1;
		for(; j>=lb; --j){
			if(u_indices[j] <= i){// search to the L matrix
				break;
			}
			else{
				x[i] -= u_val[j] * x[u_indices[j]];
			}
		}
		//assert(u_indices[j] == i);// if the element on the diagnoal of U matrix, just ignore it without update the orginal value of that row
		if(u_indices[j] == i && !Equal(u_val[j], 0)){
			x[i] /= u_val[j];
		}
	}
}


void computeSpMV_BCSR(float *x, const float *val, const int *rowIndices,
		const int *indices, const float *y, const int numRows,
		const int numCols, const int bsx, const int bsy)
{
	float *t = (float *)malloc(sizeof(float)*bsx);

	for (int i=0; i< ceild(numRows,bsx); i++) {
		memset(t,0,sizeof(float)*bsx);
		int lb = rowIndices[i];
		int ub = rowIndices[i+1];
		int r = i*bsx;
		for (int j=lb; j<ub; j++) {
			int ind = indices[j];
			int c = ind*bsy;
			int commonInd = j*bsx*bsy;

			if (((c+bsy) > numCols) || ((r+bsx) > numRows)) {
				for (int bi=0; (r+bi) < min(numRows,r+bsx); bi++) {
					float tb=0;
					for (int bj=0; (c+bj) < min(numCols,c+bsy); bj++)
						tb += val[commonInd+bi*bsy+bj] * y[c+bj];
					t[bi]+=tb;
				}
			}
			else {
#if BCSR_c8 
				// Assuming bsy as fixed (8)
				float y0 = y[c];
				float y1 = y[c+1];
				float y2 = y[c+2];
				float y3 = y[c+3];
				float y4 = y[c+4];
				float y5 = y[c+5];
				float y6 = y[c+6];
				float y7 = y[c+7];
#if BCSR_r2
				t[0]   +=   val[commonInd] * y0 + val[commonInd+1] * y1 +\
							val[commonInd+2] * y2 + val[commonInd+3] * y3 +\
							val[commonInd+4] * y4 + val[commonInd+5] * y5 +\
							val[commonInd+6] * y6 + val[commonInd+7] * y7;
				t[1] +=   val[commonInd+bsy] * y0 + val[commonInd+bsy+1] * y1 +\
						  val[commonInd+bsy+2] * y2 + val[commonInd+bsy+3] * y3 +\
						  val[commonInd+bsy+4] * y4 + val[commonInd+bsy+5] * y5 +\
						  val[commonInd+bsy+6] * y6 + val[commonInd+bsy+7] * y7;
#else
				for (int bi=0; (r+bi) < min(numRows,r+bsx); bi++) {
					t[bi] += val[commonInd+bi*bsy] * y0 + val[commonInd+bi*bsy+1] * y1 +\
							 val[commonInd+bi*bsy+2] * y2 + val[commonInd+bi*bsy+3] * y3 +\
							 val[commonInd+bi*bsy+4] * y4 + val[commonInd+bi*bsy+5] * y5 +\
							 val[commonInd+bi*bsy+6] * y6 + val[commonInd+bi*bsy+7] * y7;
				}
#endif
#endif
#if BCSR_c4 
				// Assuming bsy as fixed (4)
				float y0 = y[c];
				float y1 = y[c+1];
				float y2 = y[c+2];
				float y3 = y[c+3];
#if BCSR_r2
				t[0]   +=   val[commonInd] * y0 + val[commonInd+1] * y1 +\
							val[commonInd+2] * y2 + val[commonInd+3] * y3;
				t[1] +=   val[commonInd+bsy] * y0 + val[commonInd+bsy+1] * y1 +\
						  val[commonInd+bsy+2] * y2 + val[commonInd+bsy+3] * y3;
#else
				for (int bi=0; (r+bi) < min(numRows,r+bsx); bi++) {
					t[bi] += val[commonInd+bi*bsy] * y0 + val[commonInd+bi*bsy+1] * y1 +\
							 val[commonInd+bi*bsy+2] * y2 + val[commonInd+bi*bsy+3] * y3;
				}
#endif
#endif
#if BCSR_c2
				// Assuming bsy as fixed (2)
				float y0 = y[c];
				float y1 = y[c+1];
#if BCSR_r2
				t[0]   +=   val[commonInd] * y0 + val[commonInd+1] * y1;
				t[1] +=   val[commonInd+bsy] * y0 + val[commonInd+bsy+1] * y1; 
#else
				for (int bi=0; (r+bi) < min(numRows,r+bsx); bi++) {
					t[bi] += val[commonInd+bi*bsy] * y0 + val[commonInd+bi*bsy+1] * y1;
				}
#endif
#endif
			}
		}
		for (int bi=0; (r+bi) < min(numRows,r+bsx); bi++) 
			x[r+bi] = t[bi];
	}
	free(t);
}

