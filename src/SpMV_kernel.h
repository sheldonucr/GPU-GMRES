/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#ifndef _SPMV_KERNEL_H_
#define _SPMV_KERNEL_H_

#if CACHE
texture<float,1> tex_y_float;
#endif

#include "config.h"


////////////////////////////////////////////////////////////////////////////////
// SpMV Kernel Device Code
////////////////////////////////////////////////////////////////////////////////
__global__ void
SpMV(float *x,
		const float *val, const  int *rowIndices, const  int *indices,
		const float *y,
		const  int numRows, const  int numCols, const  int numNonZeroElements);

////////////////////////////////////////////////////////////////////////////////
// SpMV Kernel Device Code
// simplified version by zky, deleted the useless compile branches
// for the easier of reading
////////////////////////////////////////////////////////////////////////////////
__global__ void
SpMV_s(float *x,
		const float *val, const  int *rowIndices, const  int *indices,
		const float *y,
		const  int numRows, const  int numCols, const  int numNonZeroElements);


/*-------------------------------------------------------
  SpMV with Inspect Input
  -------------------------------------------------------*/
__global__ void
SpMV_withInspectInput(float *x, const float *val, const  int *rowIndices,
		const  int *indices, const float *y, const  int numRows,
		const  int numCols, const  int numNonZeroElements,
		const  int *ins_rowIndices, const  int *ins_indices, const  int *ins_inputList);


/*-------------------------------------------------------
  SpMV with Inspect
  -------------------------------------------------------*/
__global__ void
SpMV_withInspect(float *x, const float *val, const  int *rowIndices,
		const  int *indices, const float *y, const  int numRows,
		const  int numCols, const  int numNonZeroElements,
		const  int *ins_rowIndices, const  int *ins_indices);

#endif

