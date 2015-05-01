/*! \file
	\brief the functions used to factorize a matrix with left-looking ILU method
*/

#include "leftILU.h"
#include "config.h"
//#include "cuPrintf.cu"
//#include "cuPrintf.cuh"

//! if GPU flag is defined, incomplete factorization based on DAC'12 paper will be performed
#define GPU

//! intermidia vector will be store in shared memory in sparse format
//#define V1

//! intermidia vector will be store in global memory in sparse format
#define V2

//! intermidia vector will be store in global memory in full vector format
//#define V3

extern __shared__ char sd_array[];

static timeval st, et;

//! the function used to factoriza a matrix with left-looking method
void leftILU(const int numRows, 
		float* h_val, int* h_rowIndices, int* h_indices, 
		float*& l_val, int*& l_rowIndices, int*& l_indices, 
		float*& u_val, int*& u_rowIndices, int*& u_indices){

	vector<int> node;
	vector<int> level;

	gettimeofday(&st, NULL);
	generateLevel(numRows, h_val, h_rowIndices, h_indices, node, level);
	gettimeofday(&et, NULL);
	//printf("The time used for generate the level is %f (ms)\n", difftime(st, et));

	// sort the level information
	vector< vector<int> > egraph;
	try{

		for(vector<int>::iterator iter = level.begin(); iter != level.end(); ++iter){
			if(*iter+1 > egraph.size()){
				egraph.resize(*iter+1);
			}
			egraph[*iter].push_back(iter - level.begin());
		}

	}
	catch(bad_alloc& ba){
		cerr<<"Bad allocation exception!"<<endl;
	}
	catch(...){
		cerr<<"Catch Unexpected exceptions!"<<endl;
	}


	/*
	// TODO
	// WRONG CODE FOR DEBUG
	// --------------------
	egraph.clear();
	egraph.resize(numRows);
	for(int i=0; i<numRows; ++i){
	egraph[i].push_back(i);
	}
	// --------------------
	 */


	prval(egraph.size());
	// for debug
	/*
	   for(int i=0; i<egraph.size(); ++i){
	   cout<<"Level "<<i<<" contains "<<egraph[i].size()<<" nodes!"<<endl<<"\t";
	   for(int j=0; j<egraph[i].size(); ++j){
	   cout<<egraph[i][j]<<' ';
	   }
	   cout<<endl;
	   }
	 */
	printf("\n*** LEVEL DISTRIBUTION ***\n");
	for(int i=0; i<egraph.size(); ++i){
		cout<<egraph[i].size()<<'\t';
	}
	cout<<endl<<endl;

	int nnz = h_rowIndices[numRows];// may contain padded values
	int nnz_L = 0;
	int nnz_U = 0;


	for(int curRow=0; curRow<numRows; ++curRow){
		int lb = h_rowIndices[curRow];
		int ub = h_rowIndices[curRow+1];

		for(int curIndex = lb; curIndex != ub; ++curIndex){
			int curCol = h_indices[curIndex];
			float curVal = h_val[curIndex];
			if(curCol < curRow && !Equal(curVal, 0)){
				nnz_L++;
			}
			else if(curCol >= curRow && !Equal(curVal, 0)){
				nnz_U++;
			}
		}
		nnz_L++;// the diagonal of L matrix is unit value
	}
	prval(numRows);
	prval(nnz);
	prval(nnz_L);
	prval(nnz_U);


	// change CSR format to CSC format
	float *csc_val;
	int *csc_colIndices, *csc_indices;
	csr2csc(numRows, h_val, h_rowIndices, h_indices, csc_val, csc_colIndices, csc_indices);
	assert(csc_colIndices[numRows] == nnz);

	// generate L and U non-zero pattern(csc sparse format)
	float *L_val = NULL, *U_val = NULL;
	int *L_colIndices = NULL, *L_indices = NULL;
	int *U_colIndices = NULL, *U_indices = NULL;
	splitLU_csc(numRows, csc_val, csc_colIndices, csc_indices, L_val, L_colIndices, L_indices, U_val, U_colIndices, U_indices);

	prval(nnz_L);
	prval(L_colIndices[numRows]);
	prval(nnz_U);
	prval(U_colIndices[numRows]);

	assert(nnz_L == L_colIndices[numRows]);
	assert(nnz_U == U_colIndices[numRows]);


	// unitize L matrix
	for(int curCol=0; curCol<numRows; ++curCol){
		int lb = L_colIndices[curCol];
		int ub = L_colIndices[curCol+1];

		for(int curIndex=lb; curIndex<ub; ++curIndex){
			if(curIndex == lb){// the first element of each column in L matrix should be 1.0f
				assert( Equal(L_val[curIndex], 1.0f) );
			}
			else{
				L_val[curIndex] = 0.0f;
			}
		}
	}


	gettimeofday(&st, NULL);

	// allocate space for L and U matrix in device memory
	float *d_L_val = NULL, *d_U_val = NULL;
	int *d_L_colIndices = NULL, *d_L_indices = NULL;
	int *d_U_colIndices = NULL, *d_U_indices = NULL;
	cudaMalloc((void**)&d_L_val, nnz_L * sizeof(float));
	cudaMalloc((void**)&d_L_colIndices, (numRows+1) * sizeof(int));
	cudaMalloc((void**)&d_L_indices, nnz_L * sizeof(int));

	cudaMalloc((void**)&d_U_val, nnz_U * sizeof(float));
	cudaMalloc((void**)&d_U_colIndices, (numRows+1) * sizeof(int));
	cudaMalloc((void**)&d_U_indices, nnz_U * sizeof(int));

	// copy L and U contents to device memory
	cudaMemcpy(d_L_val, L_val, nnz_L * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L_colIndices, L_colIndices, (numRows+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L_indices, L_indices, nnz_L * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_U_val, U_val, nnz_U * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_colIndices, U_colIndices, (numRows+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_indices, U_indices, nnz_U * sizeof(int), cudaMemcpyHostToDevice);

	// allocate the egraph in device memory
	int *h_levelPtr, *h_nodes;
	int *d_levelPtr, *d_nodes;
	int totalLevel = egraph.size();
	h_levelPtr = (int*)malloc((totalLevel+1) * sizeof(int));
	h_nodes = (int*)malloc(numRows * sizeof(int));
	cudaMalloc((void**)&d_levelPtr, (totalLevel+1) * sizeof(int));
	cudaMalloc((void**)&d_nodes, numRows * sizeof(int));

	h_levelPtr[0] = 0;
	for(int i=0; i<totalLevel; ++i){
		h_levelPtr[i+1] = h_levelPtr[i] + egraph[i].size();
		memcpy(&h_nodes[ h_levelPtr[i] ], &egraph[i][0], egraph[i].size() * sizeof(int));
	}
	assert(h_levelPtr[totalLevel] == numRows);


	cudaMemcpy(d_levelPtr, h_levelPtr, (totalLevel+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, h_nodes, numRows * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate the csc matrix A to device memroy
	float *d_csc_val;
	int *d_csc_colIndices, *d_csc_indices;
	cudaMalloc((void**)&d_csc_val, csc_colIndices[numRows] * sizeof(float));
	cudaMalloc((void**)&d_csc_colIndices, (numRows + 1) * sizeof(int));
	cudaMalloc((void**)&d_csc_indices, csc_colIndices[numRows] * sizeof(int));

	cudaMemcpy(d_csc_val, csc_val, csc_colIndices[numRows] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csc_colIndices, csc_colIndices, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csc_indices, csc_indices, csc_colIndices[numRows] * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate the full vector for the intermedia calculation, for the third version only
#ifdef V3
	float *d_full_vector;
	cudaMalloc((void**)&d_full_vector, MAX_NUM_BLOCK * numRows * sizeof(float));
#endif

	gettimeofday(&et, NULL);
	//printf("The time used for copy data before execution is %f (ms)\n", difftime(st, et));

#ifdef GPU
	//---------- the real Left-looking algorithm
	// L has been unitized

	gettimeofday(&st, NULL);
	int curLelvel = 0;
	for(curLelvel=0; curLelvel<totalLevel; ++curLelvel){
		//printf("Current level for solving is %d, with %d nodes.\n", curLelvel, egraph[curLelvel].size());

		int dimGrid = egraph[curLelvel].size();
		int dimBlock = WARPSIZE;

		if(dimGrid < 32){
			//printf("*** Solved with GPU for %d levels\n", curLelvel);
			//printf("*** The remaining levels has too few nodes in each level, solve the remaining levels with CPU\n");
			break;
		}

		if(dimGrid > MAX_NUM_BLOCK){
			int ptr_level = 0;
			while(ptr_level < egraph[curLelvel].size()){
				int remaining_nodes_of_this_level = egraph[curLelvel].size() - ptr_level;
				dimGrid = remaining_nodes_of_this_level < MAX_NUM_BLOCK ? remaining_nodes_of_this_level : MAX_NUM_BLOCK;

#ifdef V2
				sparseTriSolve_V2<<<dimGrid, dimBlock>>>(numRows, d_csc_val, d_csc_colIndices, d_csc_indices, &d_nodes[ h_levelPtr[curLelvel] + ptr_level]);
#endif

#ifdef V3
				sparseTriSolve_V3<<<dimGrid, dimBlock>>>(numRows, d_csc_val, d_csc_colIndices, d_csc_indices, &d_nodes[ h_levelPtr[curLelvel] + ptr_level], d_full_vector);
#endif

				cudaDeviceSynchronize();

				ptr_level += dimGrid;
			}
			assert(ptr_level == egraph[curLelvel].size());
		}
		else{
#ifdef V2
			sparseTriSolve_V2<<<dimGrid, dimBlock>>>(numRows, d_csc_val, d_csc_colIndices, d_csc_indices, &d_nodes[ h_levelPtr[curLelvel] ]);
#endif

#ifdef V3
			sparseTriSolve_V3<<<dimGrid, dimBlock>>>(numRows, d_csc_val, d_csc_colIndices, d_csc_indices, &d_nodes[ h_levelPtr[curLelvel] ], d_full_vector);
#endif

			cudaDeviceSynchronize();
		}

		cudaDeviceSynchronize();
		HandleError;

	}
	cudaDeviceSynchronize();

	gettimeofday(&et, NULL);
	//printf("The time used for CUDA execution is %f (ms)\n", difftime(st, et));

	gettimeofday(&st, NULL);
	// copy the un-finished data from host to device
	cudaMemcpy(csc_val, d_csc_val, nnz * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(csc_colIndices, d_csc_colIndices, (numRows+1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(csc_indices, d_csc_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
	gettimeofday(&et, NULL);
	//printf("The time used for copy data back to host is %f (ms)\n", difftime(st, et));

	gettimeofday(&st, NULL);
	for(; curLelvel<totalLevel; ++curLelvel){
		//printf("Current level for solving is %d, with %d nodes.\n", curLelvel, egraph[curLelvel].size());
		int numNodes = egraph[curLelvel].size();
		cpuSequentialTriSolve(numRows, csc_val, csc_colIndices, csc_indices, &h_nodes[ h_levelPtr[curLelvel] ], numNodes);
	}
	gettimeofday(&et, NULL);
	//printf("The time used for CPU execution is %f (ms)\n", difftime(st, et));


	assert(nnz == h_rowIndices[numRows]);
	csr2csc(numRows, csc_val, csc_colIndices, csc_indices, h_val, h_rowIndices, h_indices);
	splitLU_csr(numRows, h_val, h_rowIndices, h_indices, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices);


#else

	// try solve it with CPU
	gettimeofday(&st, NULL);
	leftLookingILU0Cpu(numRows, csc_val, csc_colIndices, csc_indices);
	gettimeofday(&et, NULL);
	//printf("The time used for CPU execution is %f (ms)\n", difftime(st, et));

	csr2csc(numRows, csc_val, csc_colIndices, csc_indices, h_val, h_rowIndices, h_indices);
	splitLU_csr(numRows, h_val, h_rowIndices, h_indices, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices);
#endif



	//----- Memory free operations
	free(csc_val); free(csc_colIndices); free(csc_indices);

	free(L_val); free(L_colIndices); free(L_indices);

	free(U_val); free(U_colIndices); free(U_indices);

	free(h_levelPtr); free(h_nodes);

	cudaFree(d_L_val); cudaFree(d_L_colIndices); cudaFree(d_L_indices);

	cudaFree(d_U_val); cudaFree(d_U_colIndices); cudaFree(d_U_indices);

	cudaFree(d_levelPtr); cudaFree(d_nodes);

	cudaFree(d_csc_val); cudaFree(d_csc_colIndices); cudaFree(d_csc_indices);

#ifdef V3
	cudaFree(d_full_vector);
#endif

}


void generateLevel(const int numRows, 
		const float* h_val, const int* h_rowIndices, const int* h_indices, 
		vector<int>& node, vector<int>& level){

	node.resize(numRows);
	level.resize(numRows);

	// initialize the level
	for(vector<int>::iterator iter = level.begin(); iter != level.end(); ++iter){
		*iter = 0;
	}
	// initialize the node
	for(vector<int>::iterator iter = node.begin(); iter != node.end(); ++iter){
		*iter = iter - node.begin();
	}

	for(int curRow=0; curRow<numRows; ++curRow){
		int lb = h_rowIndices[curRow];
		int ub = h_rowIndices[curRow+1];

		for(int curIndex = lb; curIndex != ub; ++curIndex){
			if(!Equal(h_val[curIndex], 0) && h_indices[curIndex] > curRow){
				int curCol = h_indices[curIndex];
				if(level[curCol] < level[curRow] + 1){
					level[curCol] = level[curRow] + 1;
				}
			}
		}
	}
}


void csr2csc(const int numRows, 
		const float* csr_val, const int* csr_rowIndices, const int* csr_indices, 
		float*& csc_val, int*& csc_colIndices, int*& csc_indices){

	// use two vectors to help format converting
	vector< vector<int> > indices(numRows);
	vector< vector<float> > val(numRows);


	for(int curRow=0; curRow<numRows; ++curRow){
		int lb = csr_rowIndices[curRow];
		int ub = csr_rowIndices[curRow+1];

		for(int curIndex=lb; curIndex<ub; ++curIndex){
			int curCol = csr_indices[curIndex];
			float curVal = csr_val[curIndex];

			val[curCol].push_back(curVal);
			indices[curCol].push_back(curRow);
		}
	}

	int nnz = csr_rowIndices[numRows];
	csc_val = (float*)malloc(nnz*sizeof(float));
	csc_colIndices = (int*)malloc((numRows+1)*sizeof(int));
	csc_indices = (int*)malloc(nnz*sizeof(int));

	// generate the column ptr for csc format
	csc_colIndices[0] = 0;
	for(int i=0; i<numRows; ++i){
		csc_colIndices[i+1] = csc_colIndices[i] + indices[i].size();
	}
	assert(csc_colIndices[numRows] == nnz);

	// generate the row indices and values array for csc format
	try{
		for(int curCol=0; curCol<numRows; ++curCol){
			memcpy(&csc_val[ csc_colIndices[curCol] ], &val[curCol][0], val[curCol].size() * sizeof(float));
			memcpy(&csc_indices[ csc_colIndices[curCol] ], &indices[curCol][0], indices[curCol].size() * sizeof(int));
			assert(val[curCol].size() == indices[curCol].size());
		}
	}
	catch(...){
		cerr<<"Exception happened!"<<endl;
	}

}


void splitLU_csc(const int numRows, 
		const float* csc_val, const int* csc_colIndices, const int* csc_indices, 
		float*& L_val, int*& L_colIndices, int*& L_indices, 
		float*& U_val, int*& U_colIndices, int*& U_indices){

	vector< vector<float> > vec_L_val(numRows), vec_U_val(numRows);
	vector< vector<int> > vec_L_indices(numRows), vec_U_indices(numRows);

	for(int curCol = 0; curCol < numRows; ++curCol){
		int lb = csc_colIndices[curCol];
		int ub = csc_colIndices[curCol+1];

		// push back the diagonal value to L matrix
		vec_L_val[curCol].push_back(1.0f);
		vec_L_indices[curCol].push_back(curCol);

		for(int curIndex = lb; curIndex < ub; ++curIndex){
			int curRow = csc_indices[curIndex];
			float curVal = csc_val[curIndex];

			if(!Equal(curVal, 0) && curRow <= curCol){// U entry
				vec_U_val[curCol].push_back(curVal);
				vec_U_indices[curCol].push_back(curRow);
			}
			else if(!Equal(curVal, 0) && curRow > curCol){// L entry
				vec_L_val[curCol].push_back(curVal);
				vec_L_indices[curCol].push_back(curRow);
			}
		}
	}

	// allocate the space for the L and U column ptr
	L_colIndices = (int*)malloc((numRows+1)*sizeof(int));
	U_colIndices = (int*)malloc((numRows+1)*sizeof(int));

	// generate the column ptr
	L_colIndices[0] = 0;
	U_colIndices[0] = 0; 
	for(int curCol = 0; curCol < numRows; ++curCol){ 
		L_colIndices[curCol+1] = L_colIndices[curCol] + vec_L_val[curCol].size();
		U_colIndices[curCol+1] = U_colIndices[curCol] + vec_U_val[curCol].size();
	}

	// allocate the space for the L and U value and index array
	L_val= (float*)malloc(L_colIndices[numRows] * sizeof(float));
	U_val= (float*)malloc(U_colIndices[numRows] * sizeof(float));
	L_indices = (int*)malloc(L_colIndices[numRows] * sizeof(int));
	U_indices = (int*)malloc(U_colIndices[numRows] * sizeof(int));

	// generate the content in the L and U value and index array
	for(int curCol = 0; curCol < numRows; ++curCol){
		memcpy(&L_val[ L_colIndices[curCol] ], &vec_L_val[curCol][0], vec_L_val[curCol].size() * sizeof(float));
		memcpy(&U_val[ U_colIndices[curCol] ], &vec_U_val[curCol][0], vec_U_val[curCol].size() * sizeof(float));

		memcpy(&L_indices[ L_colIndices[curCol] ], &vec_L_indices[curCol][0], vec_L_indices[curCol].size() * sizeof(int));
		memcpy(&U_indices[ U_colIndices[curCol] ], &vec_U_indices[curCol][0], vec_U_indices[curCol].size() * sizeof(int));
	}

}


void splitLU_csr(const int numRows, 
		const float* csr_val, const int* csr_rowIndices, const int* csr_indices, 
		float*& L_val, int*& L_rowIndices, int*& L_indices, 
		float*& U_val, int*& U_rowIndices, int*& U_indices){

	vector< vector<float> > vec_L_val(numRows), vec_U_val(numRows);
	vector< vector<int> > vec_L_indices(numRows), vec_U_indices(numRows);

	for(int curRow = 0; curRow < numRows; ++curRow){
		int lb = csr_rowIndices[curRow];
		int ub = csr_rowIndices[curRow+1];


		for(int curIndex = lb; curIndex < ub; ++curIndex){
			int curCol = csr_indices[curIndex];
			float curVal = csr_val[curIndex];

			if(!Equal(curVal, 0) && curCol < curRow){// L entry
				vec_L_val[curRow].push_back(curVal);
				vec_L_indices[curRow].push_back(curCol);
			}
			else if(!Equal(curVal, 0) && curCol >= curRow){// U entry
				vec_U_val[curRow].push_back(curVal);
				vec_U_indices[curRow].push_back(curCol);
			}
		}

		// push back the diagonal value to L matrix to the last of a row for csr format
		vec_L_val[curRow].push_back(1.0f);
		vec_L_indices[curRow].push_back(curRow);

	}

	// allocate the space for the L and U column ptr
	L_rowIndices = (int*)malloc((numRows+1)*sizeof(int));
	U_rowIndices = (int*)malloc((numRows+1)*sizeof(int));

	// generate the column ptr
	L_rowIndices[0] = 0;
	U_rowIndices[0] = 0; 
	for(int curRow = 0; curRow < numRows; ++curRow){ 
		L_rowIndices[curRow+1] = L_rowIndices[curRow] + vec_L_val[curRow].size();
		U_rowIndices[curRow+1] = U_rowIndices[curRow] + vec_U_val[curRow].size();
	}

	// allocate the space for the L and U value and index array
	L_val= (float*)malloc(L_rowIndices[numRows] * sizeof(float));
	U_val= (float*)malloc(U_rowIndices[numRows] * sizeof(float));
	L_indices = (int*)malloc(L_rowIndices[numRows] * sizeof(int));
	U_indices = (int*)malloc(U_rowIndices[numRows] * sizeof(int));

	// generate the content in the L and U value and index array
	for(int curRow = 0; curRow < numRows; ++curRow){
		memcpy(&L_val[ L_rowIndices[curRow] ], &vec_L_val[curRow][0], vec_L_val[curRow].size() * sizeof(float));
		memcpy(&U_val[ U_rowIndices[curRow] ], &vec_U_val[curRow][0], vec_U_val[curRow].size() * sizeof(float));

		memcpy(&L_indices[ L_rowIndices[curRow] ], &vec_L_indices[curRow][0], vec_L_indices[curRow].size() * sizeof(int));
		memcpy(&U_indices[ U_rowIndices[curRow] ], &vec_U_indices[curRow][0], vec_U_indices[curRow].size() * sizeof(int));
	}

}


//! use shared memory to store the intermedia vector
__global__ 
void sparseTriSolve_V1(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		float* d_L_val, int* d_L_colIndices, int* d_L_indices, 
		float* d_U_val, int* d_U_colIndices, int* d_U_indices, 
		int* cols){

	int tgtCol = cols[blockIdx.x];// target row of the current block, i.e., 'k'

	int lb = d_csc_colIndices[tgtCol];
	int ub = d_csc_colIndices[tgtCol + 1];

	int colLength = ub - lb;// the length of the target column

	float* rhsVal = (float*)sd_array;// right hand side of L*x = b, size: colLength
	int* rhsRow = (int*)&rhsVal[colLength];// the index of the rhs, size: colLength

	// copy the right hand side of the equation to shared memory
	for(int i = 0 + threadIdx.x; i < colLength; i += blockDim.x){
		int curIndex = i+lb;
		rhsVal[i] = d_csc_val[curIndex];
		rhsRow[i] = d_csc_indices[curIndex];
	}
	__syncthreads();


	// the non-zero parttern of U is the same as that of A, which is rhs
	for(int i=0; i<colLength-1; ++i){// for j = 1 : k-1, where U(j, k)!=0
		int pivotRow = rhsRow[i];
		if(pivotRow >= tgtCol){// the diagonal element should not be calculated
			break;
		}

		int col_L = pivotRow;
		int lb_L = d_csc_colIndices[col_L];
		int ub_L = d_csc_colIndices[col_L+1];

		int left_pos = lb_L;

		for(int j=i+1+threadIdx.x; j<colLength; j += blockDim.x){
			int curRow = rhsRow[j];

			//left_pos = biSearchLowerBound(curRow, left_pos, ub_L, d_csc_indices);// search in L matrix for 'curRow'
			//left_pos = liSearchLowerBound(curRow, left_pos, ub_L, d_csc_indices);// search in the jth row of L matrix for 'curRow'
			// TODO
			left_pos = liSearchLowerBound(curRow, lb_L, ub_L, d_csc_indices);// search in the jth row of L matrix for 'curRow'


			if(left_pos < ub_L && d_csc_indices[left_pos] == curRow){
				rhsVal[j] = rhsVal[j] - d_csc_val[left_pos]*rhsVal[i];// where (L[j][:] != 0) && (rhsVal[j] != 0)
			}
		}
		__syncthreads();
	}

	// load the result back to d_csc matrix
	lb = d_csc_colIndices[tgtCol];
	ub = d_csc_colIndices[tgtCol + 1];

	float diag_val = 0.0f;
	for(int i=0 + threadIdx.x; i < colLength; i += blockDim.x){
		int curIndex = i+lb;

		if(rhsRow[i] < tgtCol){// U entry
			d_csc_val[curIndex] = rhsVal[i];
		}
		else if(rhsRow[i] == tgtCol){// diagonal element, U entry too
			d_csc_val[curIndex] = rhsVal[i];
			diag_val = rhsVal[i];
		}
		else{// L entry
			if(!Equal(diag_val, 0.0f)){
				d_csc_val[curIndex] = rhsVal[i] / diag_val;
			}
			else{
				d_csc_val[curIndex] = 0.0f;
			}
		}
	}


	// load the result back to L and U matrix
	// U
	int csc_stride = 0;
	int lb_U = d_U_colIndices[tgtCol];
	int ub_U = d_U_colIndices[tgtCol+1];
	int length_U = ub_U - lb_U;
	for(int i = 0 + threadIdx.x; i < length_U; i += blockDim.x){
		d_U_val[i + lb_U] = d_csc_val[csc_stride + i];
	}

	// L
	csc_stride = length_U;
	int lb_L = d_L_colIndices[tgtCol];
	int ub_L = d_L_colIndices[tgtCol+1];
	int length_L = ub_L - lb_L;
	for(int i = 0 + threadIdx.x + 1; i < length_L; i += blockDim.x){
		d_L_val[i + lb_L] = d_csc_val[csc_stride + i];// the original format has not store the diagonal element
	}

}


//! use global to store the vector in sparse format
__global__ 
void sparseTriSolve_V2(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		int* cols){

	int tgtCol = cols[blockIdx.x];// target row of the current block, i.e., 'k'

	int lb = d_csc_colIndices[tgtCol];
	int ub = d_csc_colIndices[tgtCol + 1];

	float U_diag = 0.0f;
	for(int curIndex = lb; curIndex < ub; ++curIndex){
		int curRow = d_csc_indices[curIndex];
		if(curRow > tgtCol){// index exceed to L matrix
			break;
		}
		if(curRow == tgtCol){
			U_diag = d_csc_val[curIndex];
			break;
		}

		int leftCol = curRow;
		int left_lb = d_csc_colIndices[leftCol];
		int left_ub = d_csc_colIndices[leftCol+1];
		int pleftColIndex = left_lb;
		for(int pcurColIndex = curIndex + threadIdx.x; pcurColIndex < ub; pcurColIndex += blockDim.x){
			int curRow = d_csc_indices[pcurColIndex];
			pleftColIndex = liSearchLowerBound(curRow, pleftColIndex, left_ub, d_csc_indices);
			if(pleftColIndex < left_ub && d_csc_indices[pleftColIndex] == curRow){
				// where (L[j][:] != 0) && (rhsVal[j] != 0)
				d_csc_val[pcurColIndex] -= d_csc_val[pleftColIndex]*d_csc_val[curIndex];
			}
			if(pleftColIndex == left_ub){
				break;
			}
		}
	}

	// unitize the L matrix with U_diag
	for(int curIndex = lb+threadIdx.x; curIndex<ub; curIndex += blockDim.x){
		int curRow = d_csc_indices[curIndex];
		if(curRow <= tgtCol){// still in U parts
			continue;
		}

		if(!Equal(U_diag, 0)){
			d_csc_val[curIndex] /= U_diag;
		}
		else{
			d_csc_val[curIndex] = 0.0f;
		}
	}

}


//! use global memory to store the column vectors in full vector format
__global__ 
void sparseTriSolve_V3(const int numRows, 
		float* d_csc_val, int* d_csc_colIndices, int* d_csc_indices, 
		int* cols, float* d_full_vector){


	int tgtCol = cols[blockIdx.x];// target row of the current block, i.e., 'k'

	int lb = d_csc_colIndices[tgtCol];
	int ub = d_csc_colIndices[tgtCol + 1];

	// initialize full vector
	float *ptrCurCol = &d_full_vector[blockIdx.x * numRows];
	for(int curIndex=lb+threadIdx.x; curIndex<ub; curIndex += blockDim.x){
		int curRow = d_csc_indices[curIndex];
		float curVal = d_csc_val[curIndex];

		ptrCurCol[curRow] = curVal;
	}
	__syncthreads();


	float U_diag = 0.0f;
	for(int curIndex = lb; curIndex < ub; ++curIndex){
		int curRow = d_csc_indices[curIndex];
		if(curRow > tgtCol){// index exceed to L matrix
			break;
		}
		if(curRow == tgtCol){
			U_diag = d_csc_val[curIndex];
			break;
		}

		int leftCol = curRow;
		int left_lb = d_csc_colIndices[leftCol];
		int left_ub = d_csc_colIndices[leftCol+1];
		for(int pleftColIndex = left_lb+threadIdx.x; pleftColIndex < left_ub; pleftColIndex += blockDim.x){
			int leftRow = d_csc_indices[pleftColIndex];

			ptrCurCol[leftRow] -= d_csc_val[pleftColIndex] * ptrCurCol[curRow];
		}
	}
	__syncthreads();

	// write the result back to d_csc_val while unitize the L entries
	for(int curIndex=lb+threadIdx.x; curIndex<ub; curIndex += blockDim.x){
		int curRow = d_csc_indices[curIndex];

		if(curRow <= tgtCol){// U entry
			d_csc_indices[curIndex] = ptrCurCol[curRow];
		}
		if(curRow > tgtCol){// L entry
			if(Equal(U_diag, 0.0f)){
				d_csc_indices[curIndex] = 0.0f;
			}
			else{
				d_csc_indices[curIndex] = ptrCurCol[curRow] / U_diag;
			}
		}
	}
}


void cpuSequentialTriSolve(const int numRows, 
		float* csc_val, int* csc_colIndices, int* csc_indices, 
		const int* cols, const int numNodes){


	for(int curNode=0; curNode<numNodes; ++curNode){

		int tgtCol = cols[curNode];// target row of the current block, i.e., 'k'

		int lb = csc_colIndices[tgtCol];
		int ub = csc_colIndices[tgtCol + 1];

		float U_diag = 0.0f;
		for(int curIndex = lb; curIndex < ub - 1; ++curIndex){
			int curRow = csc_indices[curIndex];
			if(curRow > tgtCol){// index exceed to L matrix
				break;
			}
			if(curRow == tgtCol){
				U_diag = csc_val[curIndex];
				break;
			}

			int leftCol = curRow;
			int left_lb = csc_colIndices[leftCol];
			int left_ub = csc_colIndices[leftCol+1];
			int pleftColIndex = left_lb;
			for(int pcurColIndex = curIndex + 1; pcurColIndex < ub; ++pcurColIndex){
				int curRow = csc_indices[pcurColIndex];

				pleftColIndex = liSearchLowerBound(curRow, left_lb, left_ub, csc_indices);
				if(pleftColIndex < left_ub && csc_indices[pleftColIndex] == curRow){
					csc_val[pcurColIndex] -= csc_val[pleftColIndex] * csc_val[curIndex];
				}
			}
		}

		// unitize the L matrix with U_diag
		for(int curIndex = lb; curIndex<ub; ++curIndex){
			int curRow = csc_indices[curIndex];
			if(curRow <= tgtCol){// still in U parts
				continue;
			}

			if(!Equal(U_diag, 0)){
				csc_val[curIndex] /= U_diag;
			}
			else{
				csc_val[curIndex] = 0.0f;
			}
		}

	}// end of for loop



}


/*!\brief Search in array for the lower bound for the element of target. 
  \return The index of the lower bound. Where retVal = min( {pos|array[pos] >= target} )
 */
template<typename T> 
__device__ __host__ 
int liSearchLowerBound(const T target, const int lpos, const int upos, const T* array){
	for(int i=lpos; i<upos; ++i){
		if(array[i] >= target){
			return i;
		}
	}
	return upos;
}


/*!\brief Search in array for the lower bound for the element of target. 
  \return The index of the lower bound. Where retVal = min( {pos|array[pos] >= target} )
 */
template<typename T> 
__device__ 
int biSearchLowerBound(const T target, const int lpos, const int upos, const T* array){
	return -1;
}


void leftLookingILU0Cpu(const int numRows, 
		float* csc_val, int* csc_colIndices, int* csc_indices){


	for(int curCol=0; curCol<numRows; ++curCol){
		int lb=csc_colIndices[curCol];
		int ub=csc_colIndices[curCol+1];
		for(int curIndex=lb; curIndex<ub; ++curIndex){// look for the non-zero pattern in U matrix
			int curRow = csc_indices[curIndex];
			if(curRow >= curCol){// index exceed to the L matrix
				break;
			}

			// Look left, use the column of 'curRow' to substitute curCol
			int leftCol = curRow;
			int left_lb = csc_colIndices[leftCol];
			int left_ub = csc_colIndices[leftCol+1];
			int pcurColIndex = curIndex;// the solution ptr
			for(int leftIndex=left_lb; leftIndex<left_ub; ++leftIndex){
				int leftRow = csc_indices[leftIndex];
				if(leftRow <= curRow){// skip the entry of U matrix
					continue;
				}

				while(pcurColIndex < ub && csc_indices[pcurColIndex] < leftRow){// the final entry is zero
					++pcurColIndex;
				}

				if(pcurColIndex == ub){
					break;
				}
				else if(csc_indices[pcurColIndex] == pcurColIndex){
					csc_val[pcurColIndex] = csc_val[pcurColIndex] - csc_val[leftIndex]*csc_val[curIndex];// x(j+1:n) = x(j+1:n) - L(j+1:n, j)*x(j)
				}
				else{
					continue;
				}

			}
		}

		// unitize the result to U and L matrix
		float U_diag = 0.0f;
		for(int curIndex=lb; curIndex<ub; ++curIndex){
			int curRow = csc_indices[curIndex];
			if(curRow <= curCol){// U part, contains the diagonal element
				if(curRow == curCol){
					U_diag = csc_val[curIndex];
					if(Equal(U_diag, 0)){ 
						break; 
					}
				}
				continue;
			}
			else{// L part, not contains the diagonal element
				if(!Equal(U_diag, 0.0f)){
					csc_val[curIndex] = csc_val[curIndex] / U_diag;
				}
				else{
					csc_val[curIndex] = 0.0f;
				}
			}
		}
	}

}



