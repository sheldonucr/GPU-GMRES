/*!	\file
 	\brief this file contains the configuration for the program
 */

/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 *  (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#ifndef __CONFIG_H__
#define __CONFIG_H__

#define INSPECT 0
#define INSPECT_BLOCK_c (BLOCKSIZE)  
#define VAR_BLOCK 1
#define VAR_COLUMN 32 

#define INSPECT_INPUT 0
#define INSPECT_INPUT_MAX 512

#define NUMTHREADS 512
#define BLOCKSIZE 512
#define HALFWARP 16
#define WARPSIZE 32

//#define MAX_NUM_BLOCK 16384
#define MAX_NUM_BLOCK 1024

#define VERIFY 1
#define DEBUG_R 0
#define EXEC_CPU 1
#define NUM_ITER 10

#define BCSR 0
#define BCSR_r 8
#define BCSR_c 8
#define BCSR_c8 1
#define BCSR_c4 0
#define BCSR_c2 0
#define BCSR_r2 0 

// padded the data can bring some speedup
#define PADDED_CSR 0

#define C_GLOBAL_OPT 0
#define NEW_GLOBAL_OPT 0 
#define GLOBAL_OPT 1
#define GLOBAL_SHARED_OPT 0
#define GLOBAL 0
#define SHARED_RI 0
#define PAD 1

#define CACHE 0

#define TIMER 1

// following defines added by zky
// preconditioner selection, if none of the following is defined, will run GMRES without preconditioner, only one preconditioner should be selected

/*!\def employ ILU0 preconditioner */
//#define ILU0 0

/*!\def employ jacobi preconditioner(diag preconditioner), the diag preconditioner is only useful for the diagonal dominant problem */
//#define DIAG 0

//#define AINV 0

/*!\def summarize the time used for different operation for GPU GMRES iteration */
#define SUB_TIMER 0

/*!\def dump the preconditioner matrix to the .txt file to the disk, so as to visulize with matlab, for debug only */
#define DUMP_MATRIX 0

/*!\def use the Matrix-Vector method implemented in cusparse for the operation within the GMRES iterations normally, the efficency is not better than the SpMV method implemented by XXLiu */
#define CUSPARSE_FLAG 0


#endif /* __CONFIG_H__ */
