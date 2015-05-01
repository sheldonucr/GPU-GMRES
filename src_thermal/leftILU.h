/*! \file 
  	\brief This file contains the functions used for the generation of the ILU0 preconditioner (both GPU and CPU version are included)
 */

#ifndef LEFTILU_H_
#define LEFTILU_H_

#include <iostream>
#include <assert.h>
#include <vector>
#include <exception>
#include <sys/time.h>
#include "SpMV.h"
#include "defs.h"

using namespace std;

//! generate the left ILU0 preconditioner
void leftILU(const int numRows, 
		float* h_val, int* h_rowIndices, int* h_indices, 
		float*& l_val, int*& l_rowIndices, int*& l_indices, 
		float*& u_val, int*& u_rowIndices, int*& u_indices);

//! get the level distribution for a matrix
void generateLevel(const int numRows, 
		const float* h_val, const int* h_rowIndices, const int* h_indices, 
		vector<int>& node, vector<int>& level);

//! transfer a matrix presented with csr format into csc format
void csr2csc(const int numRows, 
		const float* csr_val, const int* csr_rowIndices, const int* csr_indices, 
		float*& csc_val, int*& csc_colIndices, int*& csc_indices);

//! Generate the non-zero partern of the L and U format. All the input and output data are in CSC format
void splitLU_csc(const int numRows, 
		const float* csc_val, const int* csc_colIndices, const int* csc_indices, 
		float*& L_val, int*& L_colIndices, int*& L_indices, 
		float*& U_val, int*& U_colIndices, int*& U_indices);

//! input a matrix, get the L and U part of the matrix. The input and output data are presented in CSR format
void splitLU_csr(const int numRows, 
		const float* csr_val, const int* csr_rowIndices, const int* csr_indices, 
		float*& L_val, int*& L_rowIndices, int*& L_indices, 
		float*& U_val, int*& U_rowIndices, int*& U_indices);

//! version 1 of factorizing a spase matrix. use shared memory
__global__ 
void sparseTriSolve_V1(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		float* d_L_val, int* d_L_colIndices, int* d_L_indices, 
		float* d_U_val, int* d_U_colIndices, int* d_U_indices, 
		int* cols);

//! version 2 of factorizing a spase matrix. use global memory for sparse vector
__global__ 
void sparseTriSolve_V2(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		int* cols);

//! version 3 of factorizing a spase matrix. use global memory for full vector
__global__ 
void sparseTriSolve_V3(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		int* cols, float* d_full_vector);

//! solve triangle matrix with cpu sequentially
void cpuSequentialTriSolve(const int numRows, 
		float* csc_val, int* csc_colIndices, int* csc_indices, 
		const int* cols, const int numNodes);

//! search for a value in an array linearly
template<typename T> 
__device__ __host__ 
int liSearchLowerBound(const T target, const int lpos, const int upos, const T* array);

//! search for a value in an array with binary search method
template<typename T> 
__device__ 
int biSearchLowerBound(const T target, const int lpos, const int upos, const T* array);

//! factorize a matrix with left looking method on CPU
void leftLookingILU0Cpu(const int numRows, 
		float* csc_val, int* csc_colIndices, int* csc_indices);


#endif
