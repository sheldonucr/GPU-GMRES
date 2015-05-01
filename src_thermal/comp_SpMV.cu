/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "config.h"
#include "SpMV.h"
#include "SpMV_inspect.h"
#include "SpMV_kernel.h"

// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    /*--------------------------------
     * Step 0: check the input parameters and input files
     *--------------------------------*/
    // read Sparse Matrix from file or generate
    if (argc < 2 || argc > 4) {
      printf("Correct Usage:\n"
	     "  SpMV <input matrix file> [<input vector file> [<output vector file>]]\n");
      exit(-1);
    }

    // Device Initialization
    // Support for one GPU as of now
    int deviceCount; 
    cudaGetDeviceCount(&deviceCount);
    printf("cuda device count: %d\n",deviceCount);
    if (deviceCount == 0) { 
        printf("No device supporting CUDA\n"); 
	exit(-1);
    }
    cudaSetDevice(0);         

  #if TIMER
    struct timeval st, et;
  #endif

    /*--------------------------------
     * Step 1: read in the sparse matrix from file
     *--------------------------------*/
    float gputime=0.0, cputime=0.0;
    char spmfileName[256], ivfileName[256], ovfileName[256];

    // input matrix file, just a check, will read it later
    strcpy(spmfileName,argv[1]);
    FILE* f;
    if ((f = fopen(spmfileName, "r")) == NULL) {
	printf("Non-existent input matrix file\n");
        exit(-1);
    }
    else { fclose(f); }

    // read the input matrix
    SpMatrix m;
    readSparseMatrix(&m, spmfileName, 0);
    // m is in CPU's memory now

    // Added by XXLiu
    //printSparseMatrix(&m,0,1);

    /*--------------------------------
     * Step 2: read in the input vector from file, or generate a random one
     *--------------------------------*/
    unsigned numRows = m.numRows;
    unsigned numCols = m.numCols;
    unsigned numNonZeroElements = m.numNZEntries;

    unsigned memSize_row = sizeof(float) * numRows;
    unsigned memSize_col = sizeof(float) * numCols;

    // allocate host memory
    // 1) h_y is the input vector
    // 2) h_x is the output vector
    float* h_y = (float*) malloc(memSize_col);
    float* h_x = (float*) malloc(memSize_row); 

    // if input vector file specified, read from it
    // else initalize with random values
    if (argc >=3) { 
        strcpy(ivfileName,argv[2]);
        if ((f = fopen(ivfileName, "r")) == NULL) {
            printf("Non-existent input vector file\n");
            exit(-1);
        }
        else { fclose(f); }

	// read input vector
	readInputVector(h_y, ivfileName, numCols);
    }
    else { // generate a radom vector
    	for (unsigned i = 0; i < numCols; i++)
            h_y[i] = rand() / (float)RAND_MAX;
    }

    /*--------------------------------
     * Step 3: format the sparse matrix, here it is in Padded CSR format.
     *    All information in m will be saved in h_val, h_rowIndices, and h_indices.
     *--------------------------------*/
  #if PADDED_CSR
    float *h_val;
    unsigned *h_indices, *h_rowIndices;
    printf("padded csr\n"); // XXLiu
    genPaddedCSRFormat(&m, &h_val, &h_rowIndices, &h_indices);
  #else
    float* h_val = (float*) malloc(sizeof(float)*numNonZeroElements);
    unsigned* h_indices = (unsigned*) malloc(sizeof(int)*numNonZeroElements);
    unsigned* h_rowIndices = (unsigned*) malloc(sizeof(int)*(numRows+1));
    genCSRFormat(&m, h_val, h_rowIndices, h_indices);
  #endif

    // Debugging use. Added by XXLiu.
    // for (int i=0; i < numRows+1; i++)
    //   printf("   %d\t",h_rowIndices[i]);
    // printf("\n");

    // allocate device memory
    SpMatrixGPU Sparse;
    allocateSparseMatrixGPU(&Sparse, &m, h_val, h_rowIndices, h_indices, numRows, numCols);

    float *d_x, *d_y;
    cudaMalloc((void**) &d_x, memSize_row);
    cudaMalloc((void**) &d_y, memSize_col);
    cudaMemcpy(d_y, h_y, memSize_col, cudaMemcpyHostToDevice); 

  #if CACHE
    // XXLiu: bind_y() defined in cache.h. Why do not use it?
    cudaBindTexture(NULL, tex_y_float, d_y); 
  #endif
  
    unsigned gridParam;
    gridParam = (unsigned) floor((float)numRows/(BLOCKSIZE/HALFWARP));
    if ((gridParam * (BLOCKSIZE/HALFWARP)) < numRows) gridParam++;
    // XXLiu: Why not use ceil()?
    dim3 grid(1, gridParam);
    dim3 block(1, NUMTHREADS);
  
    // XXLiu: Do one time SpMV on GPU for test
  #if INSPECT
    SpMV_withInspect <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices);
  #elif INSPECT_INPUT
    SpMV_withInspectInput <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices, Sparse.d_ins_inputList);
  #else
    printf("GPU calculating x=A*y...\n");
    SpMV <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices,
			    d_y, numRows, numCols, numNonZeroElements);
  #endif
    cudaThreadSynchronize();

  /* ---------------------------------------------
     Step 4: Do a number of SpMV, get the average running time of GPU
     ---------------------------------------------*/
  #if TIMER
    gettimeofday(&st, NULL);
  #endif

  for (unsigned t=0; t<NUM_ITER; t++) {
  #if INSPECT
    SpMV_withInspect <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices);
  #elif INSPECT_INPUT
    SpMV_withInspectInput <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices, Sparse.d_ins_inputList);
  #else
    SpMV <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices,
			    d_y, numRows, numCols, numNonZeroElements);
  #endif  
    cudaThreadSynchronize();
  }

  #if TIMER
    gettimeofday(&et, NULL);
    gputime = ((et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0)/NUM_ITER;
  #endif

  #if CACHE 
    cudaUnbindTexture(tex_y_float); 
  #endif
  
    // copy result from device to host
    cudaMemcpy(h_x, d_x, memSize_row, cudaMemcpyDeviceToHost);

    float* reference = (float*) malloc(memSize_row);
  #if EXEC_CPU
    /* ---------------------------------------------
       Step 5: SpMV on CPU for once, get the average running time of CPU
     ---------------------------------------------*/
   #if TIMER
    gettimeofday(&st, NULL);
   #endif

    // compute reference solution using CPU
   #if BCSR
    float *val;
    unsigned *rowIndices, *indices;
    unsigned numblocks;
    genBCSRFormat(&m, &val, &rowIndices, &indices, &numblocks, BCSR_r, BCSR_c);
    computeSpMV_BCSR(reference, val, rowIndices, indices, h_y, numRows, numCols, BCSR_r, BCSR_c);
   #else
    // XXL: segmentation fault may occur if empty row happens
    computeSpMV(reference, h_val, h_rowIndices, h_indices, h_y, numRows);
   #endif

   #if TIMER
    gettimeofday(&et, NULL);
    cputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
   #endif

    float flops= ((numNonZeroElements*2)/(gputime*1000000));
    printf("\nRun Time Summary:\n\tGPU (ms) \tCPU (ms) \tGFLOPS\n");
    printf("\t%f\t%f\t%f\n", gputime, cputime, flops);

  #endif // EXEC_CPU

  #if VERIFY
    // check result
    float error_norm, ref_norm, diff;
    error_norm = 0;
    ref_norm = 0;
    for (int i = 0; i < numRows; ++i) {
        diff = reference[i] - h_x[i];
        error_norm += diff * diff;
        ref_norm += reference[i] * reference[i];
    }
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);
    if (fabs(ref_norm) < 1e-7)
      printf ("Test FAILED\n");
    else{
      printf( "Test %s\n", ((error_norm / ref_norm) < 1e-6f) ? "PASSED" : "FAILED");
      printf( "    Absolute Error (norm): %6.4e\n",error_norm);
      printf( "    Relative Error (norm): %6.4e\n",error_norm/ref_norm);
      }
  #endif // VERIFY

  #if DEBUG_R
    for (int i = 0; i < numRows; ++i){
        printf("y[%d]: %f ref[%d]: %f x[%d]: %f \n", i, h_y[i], i, reference[i], i, h_x[i], i);
    }
  #endif

    // if output vector file specified, write to it
    // else print it out to stdout, if input vector file is specified
    if (argc ==4) {
        strcpy(ovfileName,argv[3]);
        writeOutputVector(h_x, ovfileName, numRows);
    }
    else {
	if (argc>=3) {
    	    for (int i = 0; i < numRows; ++i)
            	printf("x[%d]: %f\n", i, h_x[i]);
	}
    }


  #if 0
    free(h_val);
    free(h_indices);
    free(h_rowIndices);
    free(h_x);
    free(h_y);
    free(reference);
    cudaFree(d_x);
    cudaFree(d_y);
  #endif
}

