/*!	\file
  	\brief the declarations for the functions used for gmres routine
 */

//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the 
// Generalized Minimum Residual method
//
// GMRES follows the algorithm described on p. 20 of the 
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//  
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//  
//*****************************************************************
#ifndef _GMRES_H_
#define _GMRES_H_

#include <math.h> 
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include <cublas.h>
#include <cusparse_v2.h>

#include <sys/time.h>
#include <assert.h>


#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include "SpMV.h"
#include "config.h"
#include "SpMV_kernel.h"

#include "defs.h"

#include "preconditioner.h"

#define REAL float

using namespace std;

// to use the sparse Matrix-Vector function of cusparse
//#define CUSPARSE_FLAG

// to get the ratio of time for different operations of GPU

#if SUB_TIMER
	static timeval st, et;
	static float time_trisolve = 0.0f, time_spmv = 0.0f, time_cublas = 0.0f;

	void summary_time(){
		printf("********** The time used for cublas operation is %f\n", time_cublas);
		printf("********** The time used for tri-matrix solving is %f\n", time_trisolve);
		printf("********** The time used for sparse matrix-vector multiply is %f\n", time_spmv);
	}

#define timeSentence(a, b) gettimeofday(&st, NULL); (a); gettimeofday(&et, NULL); b += difftime(st, et)

#else

#define timeSentence(a, b) a

#endif

class GMRES_GPU_Data{
	public:
		int numRows;
		float *s, *cs, *sn, *H;
		float *d_r, *d_rr, *d_bb, *d_temp;
		float *d_v, *d_w, *d_ww;
	
		void Initilize(const int m, const int n){
			numRows = n;

			s = (float*) malloc((m+1)*sizeof(float));
			cs = (float*) malloc((m+1)*sizeof(float));
			sn = (float*) malloc((m+1)*sizeof(float));
			H = (float*) malloc(m*(m+1)*sizeof(float));

			cudaMalloc((void**) &d_r, n*sizeof(float));
			cudaMalloc((void**) &d_rr, n*sizeof(float));
			cudaMalloc((void**)& d_bb, n*sizeof(float));
			cudaMalloc((void**)& d_temp, n*sizeof(float));

			cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
			cudaMalloc((void**) &d_w, n*sizeof(float));
			cudaMalloc((void**) &d_ww, n*sizeof(float));
		}

		~GMRES_GPU_Data(){
			free(s); free(cs); free(sn); free(H);  
			cudaFree(d_r); cudaFree(d_rr); cudaFree(d_bb); cudaFree(d_temp); 
			cudaFree(d_v);  cudaFree(d_w); cudaFree(d_ww);
		}
};

// zky
float difftime(timeval &st, timeval &et);


// Note the transpose storage in array
float mat_get(const float *A, const int row, const int col,
		const  int numRows, const  int numCols);

// Note the transpose storage in array
void mat_set(float *A, const float alpha, const int row, const int col,
		const  int numRows, const  int numCols);

// set vector to a value
void vec_initial(float *v, const float alpha, const  int n);

void copy(float *x, float *y, const int n);

// v = alpha*x
void sscal(float *v, const float *x, const float alpha, const  int n);

// y = alpha*x + y
void sapxy(float *y, const float *x, const float alpha, const  int n);

float norm2(const float *v, const  int n);

float dot(const float *x, const float *y, const  int n);

// y = alpha*A*x + beta*y
void sgemv(float *v,
		const float *val, const  int *rowIndices, const  int *indices,
		const float alpha, const float *x, const float beta, const float *y,
		const  int numRows, const  int numCols);


//! original update operations
	void 
Update(float *x, const int k, const float *H, const int m,
		const float *s, const float *v,
		const int n);



//! this function is for right DIAG precondition
	void 
Update_precondition(float *x, const int k, const float *H, const int m,
		const float *s, const float *v,
		const int n, 
		const float* m_val, const  int* m_rowIndices, const  int* m_indices);



	void 
Update_GPU(float *d_x, const int k, const float *H, const int m,
		const float *s, const float *d_v, const int n);


void ApplyPlaneRotation(float *dx, float *dy, float cs, float sn);


void GeneratePlaneRotation(float dx, float dy, float *cs, float *sn);


//! the GMRES with left diagonal preconditioner on CPU side
int 
GMRES_leftDiag(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *m_val, const  int *m_rowIndices, const  int *m_indices);// n: rowNum, m: restart threshold, with m is a inverse matrix forum


//! the GMRES method with left ILU0 preconditioner on CPU side
int 
GMRES_leftILU0(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *l_val, const int *l_rowIndices, const int *l_indices, 
		const float *u_val, const int *u_rowIndices, const int *u_indices);


//! the original GMRES method without preconditioner on CPU side
int 
GMRES(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol);// n: rowNum, m: restart



//! the right diag preconditioned GMRES on CPU side
/*!
	\m_val	the value array of the right diagonal matrix
	\m_rowIndices	the row ptr of the right diagnoal matrix
	\m_indices	the row indices of the right diagonal matrix
*/
int 
GMRES_right_diag(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *m_val, const  int *m_rowIndices, const  int *m_indices);// n: rowNum, m: restart threshold


//==============================================
// v = alpha*A*x + beta*y
void sgemv_GPU(float *v,
		const SpMatrixGPU *Sparse, const SpMatrix *spm,
		dim3 *grid, dim3 *block,
		const float alpha, const float *x,
		const float beta, const float *y,
		const  int numRows, const  int numCols);

//! pre-process the matrix to get the level of the triangle matrix, useless after the use of cusparse library
void getLevel(bool isLmatrix, const float* val, const  int* rowIndices, const  int* indices, int* level, int n);


//! the original solving of GMRES on GPU without preconditioner
int 
GMRES_GPU(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol);


int 
GMRES_GPU_leftDiag(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol, 
		const float* m_val, const int* m_rowIndices, const int* m_indices);


// solve the equation of L*U*x = y on GPU, where the L and U matrix are stored in (val, rowIndices, indices)
/*!
	\brief	this function employs cusparse library to solve the sparse triangle system, the set up and analisis part were carried out out of this function to save time, because these parts only need to carried out one time
	\L_des	the description of the L matrix
	\U_des	the description of the U matrix
	\handle	the handle for the use of cusparse library
	\status	the status of cusparse execution
	\L_info	the information of the L matrix
	\U_info the information of the U matrix
*/
void LUSolve_gpu(float *x, 
		const float *l_val,	const int *l_rowIndices, const int *l_indices, 
		const float *u_val,	const int *u_rowIndices, const int *u_indices, 
		const float *y, const int numRows, const float* alpha, 
		cusparseMatDescr_t& L_des, cusparseMatDescr_t& U_des, cusparseHandle_t& handle, cusparseStatus_t& status, 
		cusparseSolveAnalysisInfo_t& L_info, cusparseSolveAnalysisInfo_t& U_info,  
		float * v);// the intermidate vector to save the malloc time


//! this function employs left ILU0 preconditioner too solve linear system with GMRES, the precondition part is implemented by zky
/*!
	\l_val	the value array of L matrix part of preconditioner
	\l_rowIndices	the row ptr of L matrix part of preconditioner
	\l_indices	the column array of L matrix part of preconditioner
	\return 0:	if GMRES succed to convergence within a reasonable number of iterations, 1:	otherwise
*/
int 
GMRES_GPU_leftILU0(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, int *max_iter, float *tol, 
		const float* l_val, const int* l_rowIndices, const int* l_indices, 
		const float* u_val, const int* u_rowIndices, const int* u_indices);

int
GMRES_GPU_ainv(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, int *max_iter, float *tol, 
		cusp::precond::nonsym_bridson_ainv<float, cusp::device_memory> &ainv_M);

int
GMRES_cpu_AINV(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		const  int m, int *max_iter,
		float *tol, 
		const MyAINV &myAinv);


//! solve single liner equation
/*!
	\param Sparse matrix data for \p A in CSR format
	\param spm matrix for \p A containing the dimension and triplet format
	\param grid size of grid
	\param block size of block
	\param d_x the unknown variables of the equation
	\param d_b the rhs of the equation
	\param n the dimension of the matrix
	\param m restart value
	\param max_iter the number of maximum iteration
	\param tol the tolrance of error
	\param preconditioner the preconditioner for the equation
	\return 0 for success, 1 for failure
*/
int 
GMRES(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		const  int m, int *max_iter,
		float *tol, 
		Preconditioner &preconditioner);


//! solve the transient problem with Gmres on CPU
/*!
	\param Sparse matrix data for \p A in CSR format
	\param spm matrix for \p A containing the dimension and triplet format
	\param grid size of grid
	\param block size of block
	\param d_x the unknown variables of the equation
	\param d_b the rhs of the equation
	\param n the dimension of the matrix
	\param m restart value
	\param max_iter the number of maximum iteration
	\param tol the tolrance of error
	\param preconditioner the preconditioner for the equation
	\return 0 for success, 1 for failure
*/
int 
GMRES_tran(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		const  int m, const int max_iter,
		const float tol, 
		Preconditioner &preconditioner);



//! solve the single liner equation on gpu
/*!
	\param Sparse matrix data for \p A in CSR format
	\param spm matrix for \p A containing the dimension and triplet format
	\param grid size of grid
	\param block size of block
	\param d_x the unknown variables of the equation
	\param d_b the rhs of the equation
	\param n the dimension of the matrix
	\param m restart value
	\param max_iter the number of maximum iteration
	\param tol the tolrance of error
	\param preconditioner the preconditioner for the equation
	\return 0 for success, 1 for failure
*/
int 
GMRES_GPU(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, int *max_iter, float *tol, 
		Preconditioner &preconditioner);


//! solve the transient problem with Gmres on GPU
/*!
	\param Sparse matrix data for \p A in CSR format
	\param spm matrix for \p A containing the dimension and triplet format
	\param grid size of grid
	\param block size of block
	\param d_x the unknown variables of the equation
	\param d_b the rhs of the equation
	\param n the dimension of the matrix
	\param m restart value
	\param max_iter the number of maximum iteration
	\param tol the tolrance of error
	\param preconditioner the preconditioner for the equation
	\return 0 for success, 1 for failure
*/
int 
GMRES_GPU_tran(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, const int max_iter, 
		const float tol, 
		Preconditioner &preconditioner, 
		GMRES_GPU_Data &gmres_gpu_data);


#endif
