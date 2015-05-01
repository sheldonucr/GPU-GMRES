/*!	\file
	\brief the rutine to do the transient simulations
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
#include <cutil.h>
#include <cutil_inline.h>

#include "gmres.h"


#include "rightLookingILU.h"
#include "leftILU.h"
//#include "myAINV.h"

#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

extern void ainvTest();

using namespace std;


//! main function of the program
/*!
	\param argc number of parameters
	\param argv parameters
	\brief argv[0]: binary name; argv[1]: GCB directory which should contains the file of: 
	A.mtx 
	B.mtx 
	C.mtx 
	u_vec.mtx 
	t_step.mtx. 
*/
int main( int argc, char** argv) 
{

	PreconditionerType prcdtp = NONE;
	cout<<"Please select preconditioner type: "<<endl;
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

	// Device Initialization
	// Support for one GPU as of now
	int deviceCount; 
	cudaGetDeviceCount(&deviceCount);
	printf("cuda device count: %d\n",deviceCount);
	if (deviceCount == 0) { 
		printf("No device supporting CUDA\n"); 
		exit(-1);
	}
	cout<<"Select device 0"<<endl;
	cudaSetDevice(0);
	HandleError;

	trace;

	cublasStatus status;       
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		prval(status);
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	trace;

	/*---------------------------------------------------*
	 * Step 0: check the input parameters and input files*
	 *---------------------------------------------------*/
	// read Sparse Matrix from file or generate
	if (argc != 2) {
		printf("Correct Usage:\n"
				"  gmres_SpMV dirName\n");
		exit(-1);
	}
	trace;

	/*----------------------------------------------*
	 * Step 1: read in the sparse matrices from file*
	 *----------------------------------------------*/
	float gputime=0.0, cputime=0.0;
	char spmfileName[256], ivfileName[256], ovfileName[256];
	string dirName;
	dirName = argv[1];

	trace;

	SpMatrix A;
	readSparseMatrix(&A, (dirName + "/A.mtx").c_str(), 0);
	SpMatrix B;
	readSparseMatrix(&B, (dirName + "/B.mtx").c_str(), 0);
	SpMatrix C;
	readSparseMatrix(&C, (dirName + "/C.mtx").c_str(), 0);
	trace;

	printf("Finised reading matrices for A, B and C!\n");

	float t_step;
	ifstream fin;
	fin.open((dirName + "/t_step.mtx").c_str());
	assert(!fin.fail());
	fin>>t_step;
	fin.close();

	// C <- C/h
	for(int i=0; i<C.numNZEntries; ++i){
		C.nzentries[i].val /= t_step;
	}

	MySpMatrix mySpM_B;
	MySpMatrix mySpM_C;
	mySpM_B.Initilize(B);
	mySpM_C.Initilize(C);

	trace;
	

	/*--------------------------------
	 * Step 2: read in the input vector from file, or generate a random one
	 *--------------------------------*/
	int numRows = A.numRows;
	int numCols = A.numCols;
	int numNonZeroElements = A.numNZEntries;
	int memSize_row = sizeof(float) * numRows;
	int memSize_col = sizeof(float) * numCols;

	assert(numRows == numCols);

	// allocate host memory
	// 1) h_y is the input vector
	// 2) h_x is the output vector
	float* h_y = (float*) malloc(memSize_col);
	float* h_x = (float*) malloc(memSize_row); 

	FILE *f;
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
	 *    All information in A will be saved in h_val, h_rowIndices, and h_indices.
	 *--------------------------------*/
#if PADDED_CSR
	float *h_val;
	int *h_indices, *h_rowIndices;
	genPaddedCSRFormat(&A, &h_val, &h_rowIndices, &h_indices);
	printf("padded csr: h_rowIndices[numRows]=%d\n",h_rowIndices[numRows]); // XXLiu
#else
	float* h_val = (float*) malloc(sizeof(float)*numNonZeroElements);
	int* h_indices = (int*) malloc(sizeof(int)*numNonZeroElements);
	int* h_rowIndices = (int*) malloc(sizeof(int)*(numRows+1));
	genCSRFormat(&A, h_val, h_rowIndices, h_indices);
#endif
	// After padding, the rowIndices look like 0, 16, 32, 48, . . .

	// allocate device memory
	SpMatrixGPU Sparse;
	allocateSparseMatrixGPU(&Sparse, &A, h_val, h_rowIndices, h_indices, numRows, numCols);

	float *d_x, *d_y;
	cudaMalloc((void**) &d_x, memSize_row);
	cudaMalloc((void**) &d_y, memSize_col);
	cudaMemcpy(d_y, h_y, memSize_col, cudaMemcpyHostToDevice); 

	// collect all the pointers to both Host and Device
	MySpMatrix mySpM;
	mySpM.numRows = A.numRows;
	mySpM.numCols= A.numCols;
	mySpM.numNZEntries = h_rowIndices[A.numRows];
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
	genBCSRFormat(&A, &val, &rowIndices, &indices, &numblocks, BCSR_r, BCSR_c);
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

	trace;


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

	trace;



	/*-----------------------------------------*
	 * Begin the transistant simulation in CPU *
	 *-----------------------------------------*/
	const int restart=32, max_iter=60000;
	int max_it = max_iter;
	const float tolerance = 1e-6;
	float tol = tolerance;
	float *h_temp = new float[A.numRows];
	int result;

	ifstream fin_u_vec;
	fin_u_vec.open((dirName+"/u_vec.mtx").c_str());
	assert(!fin_u_vec.fail());

	float f_u_vec_numLines, f_u_vec_numElements;
	int u_vec_numLines, u_vec_numElements;
	fin_u_vec>>f_u_vec_numLines>>f_u_vec_numElements;
	u_vec_numLines= (int)f_u_vec_numLines;
	u_vec_numElements = (int)f_u_vec_numElements;

	prval(f_u_vec_numLines);
	prval(f_u_vec_numElements);
	prval(u_vec_numLines);
	prval(u_vec_numElements);


	float *h_u_vec = new float[u_vec_numElements];
	float *d_u_vec;
	cudaMalloc((void**)&d_u_vec, u_vec_numElements * sizeof(float));

	prval(mySpM.numRows);
	prval(mySpM.numCols);
	assert(mySpM.numRows == mySpM.numCols);
	assert(mySpM.numRows == mySpM_B.numRows);
	prval(u_vec_numElements);
	prval(mySpM_B.numCols);
	assert(u_vec_numElements == mySpM_B.numCols);
	assert(mySpM_C.numRows == mySpM_C.numCols);
	assert(mySpM.numRows == mySpM_C.numRows);

	int numIter = u_vec_numLines;
	prval(numIter);

	FILE *fptr_cpu;
	char xCPUfileName[] = "xCPU.txt";
	fptr_cpu = fopen(xCPUfileName, "w");
	assert(fptr_cpu != NULL);

	for(int i = 0; i < numRows; ++i)// guess initial soln
		h_x[i] = 0; 


	//numIter = 100;

	//goto loop;

	for(int i=0; i<numIter; ++i){
		printf("Debug: %dth simulation for CPU!\n", i);
		// load u_vec from file 
		loadVectorFromFile(fin_u_vec, u_vec_numElements, h_u_vec);

		// h_temp <- B * u_vec
		TIME(computeSpMV(h_temp, mySpM_B.val, mySpM_B.rowIndices, mySpM_B.indices, h_u_vec, numRows), cputime);

		// h_y <- (C / h) * h_x
		TIME(computeSpMV(h_y, mySpM_C.val, mySpM_C.rowIndices, mySpM_C.indices, h_x, numRows), cputime);

		// h_y <- B * u_vec[i] + (C / h) * h_x
		TIME(addTwoVec2(h_temp, h_y, numRows), cputime);

		TIME( result = GMRES_tran(h_val, h_rowIndices, h_indices, h_x, h_y, numRows, restart, max_it, tol, *preconditioner), cputime);

	}
	// store h_x to file
	writeOutputVector(h_x, fptr_cpu, numRows);
	fclose(fptr_cpu);
	fin_u_vec.close();
	delete [] h_temp;


	loop:

	/*-----------------------------------------*
	 * Begin the transistant simulation in GPU *
	 *-----------------------------------------*/
	float *d_temp;
	cudaMalloc((void**)&d_temp, A.numRows * sizeof(float));
	float one = 1.0f;

	GMRES_GPU_Data gmres_gpu_data;
	gmres_gpu_data.Initilize(restart, numRows);

	ifstream fin_u_vec_gpu;
	fin_u_vec_gpu.open((dirName+"/u_vec.mtx").c_str());
	assert(!fin_u_vec_gpu.fail());
	fin_u_vec_gpu>>f_u_vec_numLines>>f_u_vec_numElements;

	char xGPUfileName[] = "xGPU.txt";
	FILE *fptr_gpu;
	fptr_gpu = fopen(xGPUfileName, "w");
	assert(fptr_gpu != NULL);

	for(int i = 0; i < numRows; ++i){// guess initial soln
		h_x[i] = 0; 
	}
	cudaMemcpy(d_x, h_x, numRows * sizeof(float), cudaMemcpyHostToDevice);

	prval(numIter);
	for(int i=0; i<numIter; ++i){
		printf("Debug: %dth simulation for GPU!\n", i);
		// load u_vec from file 
		loadVectorFromFile(fin_u_vec_gpu, u_vec_numElements, h_u_vec);
		cudaMemcpy(d_u_vec, h_u_vec, u_vec_numElements * sizeof(float), cudaMemcpyHostToDevice);

		// d_temp <- B * u_vec[i]
		TIME((SpMV<<<grid, block>>>(d_temp, mySpM_B.d_val, mySpM_B.d_rowIndices, mySpM_B.d_indices, d_u_vec, mySpM_B.numRows, mySpM_B.numCols, mySpM_B.numNZEntries)), gputime);

		// d_y <- (C / h) * d_x
		TIME((SpMV<<<grid, block>>>(d_y, mySpM_C.d_val, mySpM_C.d_rowIndices, mySpM_C.d_indices, d_x, mySpM_C.numRows, mySpM_C.numCols, mySpM_C.numNZEntries)), gputime);

		// d_y <- B * u_vec[i] + (C / h) * h_x
		TIME(cublasSaxpy(numRows, one, d_temp, 1, d_y, 1), gputime);

		TIME( result = GMRES_GPU_tran(&Sparse, &A, &grid, &block, d_x, d_y, numRows, restart, max_it, tol, *preconditioner, gmres_gpu_data), gputime);

	}
	// store d_x to file
	cudaMemcpy(h_x, d_x, numRows * sizeof(float), cudaMemcpyDeviceToHost);
	writeOutputVector(h_x, fptr_gpu, numRows);
	fclose(fptr_gpu);
	fin_u_vec_gpu.close();
	cudaFree(d_temp);

	printf("Time for CPU is %fms\n", cputime);
	printf("Time for GPU is %fms\n", gputime);


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


