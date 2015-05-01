/*!	\file
 	\brief declare the functions and class used to sparse matrix operations
 */

/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#ifndef __SPMV_H__
#define __SPMV_H__

#include <fstream>

using namespace std;

#define ceild(n,d)  ceil(((float)(n))/((float)(d)))
#define floord(n,d) floor(((float)(n))/((float)(d)))
#define Max(x,y)    ((x) > (y)? (x) : (y))
#define Min(x,y)    ((x) < (y)? (x) : (y))

struct nzInfo
{
	int rowNum;
	int colNum;
	float val;
};

typedef struct nzInfo NZEntry;

struct SpM
{
	int numRows;
	int numCols;
	int numNZEntries;
	NZEntry *nzentries;
	int *rowPtrs;
	int *colPtrs;
};

typedef struct SpM SpMatrix;

struct SpMGPU
{
	float* d_val;
	int* d_indices;
	int* d_rowIndices;

	int* d_ins_indices;
	int* d_ins_rowIndices;
	int* d_ins_inputList;
};

typedef struct SpMGPU SpMatrixGPU;

//! The class is used to manage the pointers to the matrix data, including both host and device data. CSR format
class MySpMatrix{
	public:
                int isCSR; // 1 or 0
		int numRows;
		int numCols;
		int numNZEntries;

		float *d_val;
		int *d_indices;
		int *d_rowIndices;

		float *val;
		int *indices;
		int *rowIndices;

		//! Initilize the attributes with a SpMatrix object
		/*!
		  \brief the data in host has been allocated. This function will allocate the storage for Device
		  \param M the SpMatrix object containing the host data and matrix dimension information
		 */
                // MySpMatrix() : isCSR(1) {};
		void Initilize(SpMatrix &M);
                
		// //! Destruction function, release the Device data
		// ~MySpMatrix();
};
void mySpMatrixFree(MySpMatrix *);

class MySpMatrixDouble{
	public:
                int isCSR;

		int numRows;
		int numCols;
		int numNZEntries;

		double *d_val;
		int *d_indices;
		int *d_rowIndices;

		double *val;
		int *indices;
		int *rowIndices;
};

void mySpMatrixDoubleFree(MySpMatrixDouble *);


//! add \p v1 and \p v2. store the result into v2.
void addTwoVec2(const float *v1, float *v2, const int num);


//! load \p num elements from \p fin and add them to the array of \p vec
/*!
  \brief this function is mainly used to read ascii data for u_vec. Since u_vec is dense, each time only a column of data are loaded
 */
void loadVectorFromFile(ifstream &fin, int num, float *vec);
void readInputVector(float *y, const char *filename, int numCols);

//! write the result to a file on disk
void writeOutputVector(float *x, const char *filename, int numRows);
//! write the result to a file on disk
void writeOutputVector(float *x, FILE *f, int numRows);

void readSparseMatrix(SpMatrix *m, const char *filename, int format);
void readSparseMatrixBinaryD2S(SpMatrix *m, const char *filename, int format);
void genCSRFormat(SpMatrix *m, float *val,  int *rowIndices,  int *indices);
void genCSCFormat(SpMatrix *m, float *val,  int *colIndices,  int *indices);
void genBCSRFormat(SpMatrix *m, float **val,  int **rowIndices,  int **indices,
		int *numblocks,  int bsx,  int bsy);
void genPaddedCSRFormat(SpMatrix *m, float **val,  int **rowIndices,  int **indices);
void allocateSparseMatrixGPU(SpMatrixGPU *spm, SpMatrix *m, float *h_val,  int *h_rowIndices,
		int *h_indices, const  int numRows, const  int numCols);

// 2012-11-14 pg-gmres-gpu
void gpuMallocCpyCSR(SpMatrixGPU *spm, float *h_val,  int *h_rowIndices,
                     int *h_indices, const int numRows, const int numCols);
void gpuMallocCpyCSRmySpM(SpMatrixGPU *spm, MySpMatrix *mySpM);

void freeSparseMatrixGPU(SpMatrixGPU *spm);
void SpMV_cuda(float *x, SpMatrixGPU *spm, const float *y, const  int numRows,
		const  int numCols, const  int numNonZeroElements);
void computeSpMV(float *x, const float *val, const  int *rowIndices, const  int *indices,
		const float *y, const  int numRows);

//! solve 
void LUSolve(float *x, const float *l_val, const int *l_rowIndices, const int *l_indices, const float *u_val, const int *u_rowIndices, const int *u_indices, const float *y, const  int numRows);

//! solve
void LUSolve_ignoreZero(float *x, const float *l_val, const int *l_rowIndices, const int *l_indices, const float *u_val, const int *u_rowIndices, const int *u_indices, const float *y, const  int numRows);

void computeSpMV_BCSR(float *x, const float *val, const  int *rowIndices,
		const  int *indices, const float *y, const  int numRows,
		const  int numCols, const  int bsx, const  int bsy);

// added by XXLiu for matrix display
void printSparseMatrix(SpMatrix *m, int Data, int Header);

#endif /* __SPMV_H__ */
