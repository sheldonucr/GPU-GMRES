/*!	\file
	\brief the routine to solve a single linear equation Ax = b
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
#include "defs.h"
#include <assert.h>
#include <fstream>
#include <exception>

#include "cublas.h"
//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>

#include "gmres.h"


#include "rightLookingILU.h"
#include "leftILU.h"

#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

extern void ainvTest();

using namespace std;

const int restart=32, max_iter=60000;
const float tolerance = 1e-6;


//! main function of the program
/*!
  \param argc number of parameters
  \param argv parameters
  \brief argv[0]: binary name; argv[1]: A matrix name; [argv[2]: rhs name; argv[3]: output name]
 */
int main( int argc, char** argv) 
{

	PreconditionerType prcdtp = NONE;
	cout<<"Please input precondition type."<<endl;
	cout<<"\t0 for None preconditioner."<<endl;
	cout<<"\t1 for Diagonal preconditioner."<<endl;
	cout<<"\t2 for ILU0 preconditioner."<<endl;
	cout<<"\t3 for Ainv preconditioner."<<endl;

	int ttt;
	cin>>ttt;
	switch(ttt){
		case 0:
			prcdtp = NONE;
			break;
		case 1:
			prcdtp = DIAG;
			break;
		case 2:
			prcdtp = ILU0;
			break;
		case 3:
			prcdtp = AINV;
			break;
		default:
			cerr<<"Unknown type!"<<endl;
			exit(-1);
	}

	/*--------------------------------
	 * Step 0: check the input parameters and input files
	 *--------------------------------*/
	// read Sparse Matrix from file or generate
	if (argc < 2 || argc > 4) {
		printf("Correct Usage:\n"
				"  gmres_SpMV <matrix file> <rhs vector file> [<output vector file>]]\n");
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
	HandleError;

	cublasStatus status;       
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		prval(status);
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	/*--------------------------------
	 * Step 1: read in the sparse matrix from file
	 *--------------------------------*/
	float gputime=0.0, cputime=0.0;
	char spmfileName[256], ivfileName[256], ovfileName[256];

	// input matrix file and rhs file, just a check, will read it later
	strcpy(spmfileName,argv[1]);
	FILE* f;
	if ((f = fopen(spmfileName, "r")) == NULL) {
		printf("Non-existent input matrix file\n"); exit(-1);
	}
	else { fclose(f); }

	// read the input matrix
	SpMatrix m;
	readSparseMatrix(&m, spmfileName, 0);
	// m is in CPU's memory now


	/*--------------------------------
	 * Step 2: read in the input vector from file, or generate a random one
	 *--------------------------------*/
	int numRows = m.numRows;
	int numCols = m.numCols;
	int numNonZeroElements = m.numNZEntries;
	int memSize_row = sizeof(float) * numRows;
	int memSize_col = sizeof(float) * numCols;

	assert(numRows == numCols);

	// allocate host memory
	// 1) h_y is the input vector
	// 2) h_x is the output vector
	float* h_y = (float*) malloc(memSize_col);
	float* h_x = (float*) malloc(memSize_row); 

	// if input vector file specified, read from it
	// otherwise, initalize with random values
	if (argc >=3) { 
		strcpy(ivfileName,argv[2]);
		if ((f = fopen(ivfileName, "r")) == NULL) {
			printf("Non-existent input vector file\n"); exit(-1);
		}
		else { fclose(f); }
		// read input vector
		readInputVector(h_y, ivfileName, numCols);
	}
	else{// generate a radom vector
		for (int i = 0; i < numCols; i++){
			//h_y[i] = rand() / (float)RAND_MAX;
			h_y[i] = 0.5f;
		}
	}


	/*--------------------------------
	 * Step 3: format the sparse matrix, here it is in Padded CSR format.
	 *    All information in m will be saved in h_val, h_rowIndices, and h_indices.
	 *--------------------------------*/
#if PADDED_CSR
	float *h_val;
	int *h_indices, *h_rowIndices;
	genPaddedCSRFormat(&m, &h_val, &h_rowIndices, &h_indices);
	printf("padded csr: h_rowIndices[numRows]=%d\n",h_rowIndices[numRows]); // XXLiu
#else
	float* h_val = (float*) malloc(sizeof(float)*numNonZeroElements);
	int* h_indices = (int*) malloc(sizeof(int)*numNonZeroElements);
	int* h_rowIndices = (int*) malloc(sizeof(int)*(numRows+1));
	genCSRFormat(&m, h_val, h_rowIndices, h_indices);
#endif
	// After padding, the rowIndices look like 0, 16, 32, 48, . . .

	// allocate device memory
	SpMatrixGPU Sparse;
	allocateSparseMatrixGPU(&Sparse, &m, h_val, h_rowIndices, h_indices, numRows, numCols);

	float *d_x, *d_y;
	cudaMalloc((void**) &d_x, memSize_row);
	cudaMalloc((void**) &d_y, memSize_col);
	cudaMemcpy(d_y, h_y, memSize_col, cudaMemcpyHostToDevice); 

	// collect all the pointers to both Host and Device
	MySpMatrix mySpM;
	mySpM.numRows = m.numRows;
	mySpM.d_val = Sparse.d_val;
	mySpM.d_rowIndices = Sparse.d_rowIndices;
	mySpM.d_indices = Sparse.d_indices;
	mySpM.val = h_val;
	mySpM.rowIndices = h_rowIndices;
	mySpM.indices = h_indices;

#if CACHE
	// XXLiu: bind_y() defined in cache.h. Why do not use it?
	cudaBindTexture(NULL, tex_y_float, d_y); 
#endif

	int gridParam;
	gridParam = (int) floor((float)numRows/(BLOCKSIZE/HALFWARP));
	if ((gridParam * (BLOCKSIZE/HALFWARP)) < numRows) 
		gridParam++;
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
	struct timeval st, et;
	gettimeofday(&st, NULL);
#endif

	for (int t=0; t<NUM_ITER; t++) {
#if INSPECT
		SpMV_withInspect <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices);
#elif INSPECT_INPUT
		SpMV_withInspectInput <<<grid, block>>> (d_x, Sparse.d_val, Sparse.d_rowIndices, Sparse.d_indices, d_y, numRows, numCols, numNonZeroElements, Sparse.d_ins_rowIndices, Sparse.d_ins_indices, Sparse.d_ins_inputList);
#else// will run
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
#if BCSR// not run
	float *val;
	int *rowIndices, *indices;
	int numblocks;
	genBCSRFormat(&m, &val, &rowIndices, &indices, &numblocks, BCSR_r, BCSR_c);
	computeSpMV_BCSR(reference, val, rowIndices, indices, h_y, numRows, numCols, BCSR_r, BCSR_c);
#else// will run
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
	float error_norm=0, ref_norm=0, diff;
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
	for (int i = 0; i < numRows; ++i)
		printf("y[%d]: %f ref[%d]: %10.8e x[%d]: %10.8e \n", i, h_y[i], i, reference[i], i, h_x[i], i);
#endif

	// if output vector file specified, write to it
	// else print it out to stdout, if input vector file is specified
	if (argc == 4) {
		strcpy(ovfileName,argv[3]);
		writeOutputVector(h_x, ovfileName, numRows);
	}
	else {
		/*
		   if (argc>=3)
		   for (int i = 0; i < numRows; ++i)
		   printf("x[%d]: %f\n", i, h_x[i]);
		 */
	}

	//--------------------------------------
	// precondition here
	//--------------------------------------
	float prcdTime = 0.0;
	timeval sst, eet;
	gettimeofday(&sst, NULL); 
	Preconditioner *preconditioner;
	switch(prcdtp){
		case NONE:
			preconditioner = (Preconditioner *)new MyNONE();
			break;
		case DIAG:
			preconditioner = (Preconditioner *)new MyDIAG();
			break;
		case ILU0:
			preconditioner = (Preconditioner *)new MyILU0();
			break;
		case AINV:
			preconditioner = (Preconditioner *)new MyAINV();
			break;
		default:
			cerr<<"Unknow type"<<endl;
			exit(-1);
	}
	preconditioner->Initilize(mySpM);

	gettimeofday(&eet, NULL); 
	prcdTime = difftime(sst, eet);
	printf("****** Time for preconditioning is %fms\n", prcdTime);

	//--------------------------------------
	// test of GMRES here in CPU: START
	for(int i = 0; i < numRows; ++i)// guess soln
		h_x[i] = 1; 

	//const int restart=50, max_iter=6000;
	//const float tolerance = 1e-9;
	int max_it = max_iter;
	float tol = tolerance;
	gettimeofday(&st, NULL);


	// solve with preconditioned GMRES on Host
	int result = GMRES(h_val, h_rowIndices, h_indices, h_x, h_y, numRows, restart, &max_it, &tol, *preconditioner);

	gettimeofday(&et, NULL);
	cputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
	printf("CPU GMRES flag = %d\n", result);
	printf("  iterations performed: %d\n", max_it);
	printf("  tolerance achieved: %8.6e\n", tol);
	char xCPUfileName[] = "xCPU.txt";
	strcpy(ovfileName,xCPUfileName);
	writeOutputVector(h_x, ovfileName, numRows);
	// test of GMRES here in CPU: END
	//--------------------------------------

	// ------------------------------------------
	// test of GMRES here in GPU devices: START
	cublasSetVector(numRows, sizeof(float), h_y, 1, d_y, 1);

	for (int i=0; i<numRows; i++)  h_x[i] = 1;
	cudaMemcpy(d_x, h_x, memSize_col, cudaMemcpyHostToDevice);

	max_it = max_iter;
	tol = tolerance;
	gettimeofday(&st, NULL);

	// solve with preconditioned GMRES on Device
	result = GMRES_GPU(&Sparse, &m, &grid, &block, d_x, d_y, numRows, restart, &max_it, &tol, *preconditioner);

	gettimeofday(&et, NULL);
	gputime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
	cudaMemcpy(h_x, d_x, memSize_col, cudaMemcpyDeviceToHost);

	char xGPUfileName[] = "xGPU.txt";
	strcpy(ovfileName,xGPUfileName);
	writeOutputVector(h_x, ovfileName, numRows);

	printf("GPU GMRES flag = %d\n", result);
	printf("  iterations performed: %d\n", max_it);
	printf("  tolerance achieved: %8.6e\n", tol);

	printf("CPU time: %f (ms)\n", cputime);
	printf("GPU time: %f (ms)\n", gputime);
	// test of GMRES here in GPU devices: END
	// ------------------------------------------

	cublasShutdown();

#if 1
	free(h_val);
	free(h_indices);
	free(h_rowIndices);
	free(h_x);
	free(h_y);
	free(reference);
	cudaFree(d_x);
	cudaFree(d_y);
#endif

	return 0;
}


