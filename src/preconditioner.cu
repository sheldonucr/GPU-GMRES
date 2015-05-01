/*!	\file
	\brief implement the functions for the decendants of the class of \p Preconditioner
*/
//#include <cutil.h>
#include <helper_cuda.h>
#include "preconditioner.h"

#include "gpuData.h"

/*!
	\brief the function used to extract the sparse matrix from a string stream
*/
bool getMatrixFromStringStream(stringstream &ss, int *&rowIndices, int *&indices, float *&val){
	string temp;

	while(ss){
		ss>>temp;
		//cout<<"Dummy contents from string stream is "<<temp<<endl;
		if(temp == string("entries")){
			break;
		}
	}
	if(!ss)
		return false;


	vector<int> vecRow, vecColumn;
	vector<float> vecVal;
	int count = 0;
	while(ss){
		ss >>temp;
		if(count % 3 == 0){
			vecRow.push_back(atoi(temp.c_str()));
		}
		else if(count % 3 == 1){
			vecColumn.push_back(atoi(temp.c_str()));
		}
		else{
			vecVal.push_back(atof(temp.c_str()));
		}
		count++;
	}


	vecRow.pop_back();
	assert(vecRow.size() == vecColumn.size() && vecColumn.size() == vecVal.size());


	int numRows, nnz;
	numRows = vecRow.back() + 1;
	nnz = vecVal.size();

	rowIndices = (int *)malloc((numRows + 1) * sizeof(int));
	indices = (int *)malloc(nnz * sizeof(int));
	val = (float *)malloc(nnz * sizeof(float));

	// generate rowIndices
	rowIndices[0] = 0;
	int curRow = 0;
	for(int i=0; i<nnz; ++i){
		if(vecRow[i] != curRow){
			rowIndices[++curRow] = i;
		}
	}
	rowIndices[++curRow] = nnz;
	assert(curRow == numRows);

	// generate indices and val
	memcpy(indices, &vecColumn[0], nnz * sizeof(int));
	memcpy(val, &vecVal[0], nnz * sizeof(float));

	return true;
}

/*!
	\brief the function used to extract the vector from a string stream
*/
bool getArray1dFromStringStream(stringstream &ss, float *&val, int &dim){
	string temp;

	while(ss){
		ss>>temp;
		//cout<<"Dummy content from string stream is "<<temp<<endl;
		if(temp[temp.length()-1] == '>'){
			break;
		}
	}
	if(!ss)
		return false;


	vector<float> vecVal;
	while(ss){
		ss >>temp;
		vecVal.push_back(atof(temp.c_str()));
	}


	vecVal.pop_back();

	dim = vecVal.size();

	val = (float *)malloc(dim * sizeof(float));

	memcpy(val, &vecVal[0], dim * sizeof(float));

	return true;
}

void MyAINV::HostPrecond(const ValueType *i_data, ValueType *o_data){

	ValueType *temp = new ValueType[numRows];

	// o_data = w_t * i_data
	computeSpMV(o_data, this->z_val, this->z_rowIndices, this->z_indices, i_data, numRows);

	// temp = D * w_t * i_data
	for(IndexType i=0; i<numRows; ++i){
		//assert(!Equal(this->diag_val[i], 0.0f));
		temp[i] = o_data[i] * this->diag_val[i];
	}

	// o_data= z * (D^-1) * w_t * i_data
	computeSpMV(o_data, this->w_t_val, this->w_t_rowIndices, this->w_t_indices, temp, numRows);

	delete [] temp;
}

void MyAINV::DevPrecond(const ValueType *i_data, ValueType *o_data){
	checkCudaErrors(cudaMemcpy(pin_array, i_data, numRows*sizeof(ValueType),
                                  cudaMemcpyDeviceToDevice));

	(*ainv_M)(*in_array, *out_array);

	checkCudaErrors(cudaMemcpy(o_data, pout_array, numRows*sizeof(ValueType),
                                  cudaMemcpyDeviceToDevice));
}


void MyAINV::Initilize(const MySpMatrix &mySpM){
	this->numRows = mySpM.numRows;
	trace;

	cusp::csr_matrix<IndexType, ValueType, MemorySpace> cusp_csr_A;
	cusp_csr_A.resize(numRows, numRows, mySpM.rowIndices[numRows]);
	trace;

	assert(cusp_csr_A.row_offsets.size() == (mySpM.numRows + 1));
	assert(cusp_csr_A.values.size() == mySpM.rowIndices[mySpM.numRows]);

	int *ptr_row_offsets = thrust::raw_pointer_cast(&cusp_csr_A.row_offsets[0]);
	int *ptr_column_indices = thrust::raw_pointer_cast(&cusp_csr_A.column_indices[0]);
	float *ptr_values = thrust::raw_pointer_cast(&cusp_csr_A.values[0]);
	trace;

	checkCudaErrors(cudaMemcpy(ptr_row_offsets, mySpM.d_rowIndices,
                                  (mySpM.numRows+1)*sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(ptr_column_indices, mySpM.d_indices,
                                  mySpM.rowIndices[mySpM.numRows]*sizeof(int),
                                  cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(ptr_values, mySpM.d_val,
                                  mySpM.rowIndices[mySpM.numRows]*sizeof(float),
                                  cudaMemcpyDeviceToDevice));

	// setup preconditioner
	this->ainv_M = new cusp::precond::nonsym_bridson_ainv<ValueType, MemorySpace>(cusp_csr_A, .1);
	trace;

	stringstream ss;
	stringstream ss1;
	stringstream ss2;
	cusp::print(ainv_M->w_t, ss);
	cusp::print(ainv_M->z, ss1);
	cusp::print(ainv_M->diagonals, ss2);
	trace;

	/*
	cusp::print(cusp_csr_A);
	cusp::print(ainv_M.w_t);
	cusp::print(ainv_M.z);
	cusp::print(ainv_M.diagonals);
	 */

	// for the Host part preconditioner
	getMatrixFromStringStream(ss, w_t_rowIndices, w_t_indices, w_t_val);
	getMatrixFromStringStream(ss1, z_rowIndices, z_indices, z_val);
	getArray1dFromStringStream(ss2, diag_val, numRows);
	trace;

	in_array = new cusp::array1d<float, cusp::device_memory>();
	out_array = new cusp::array1d<float, cusp::device_memory>();

	prval(numRows);
	in_array->resize(numRows);
	trace;
	out_array->resize(numRows);
	trace;
	pin_array = thrust::raw_pointer_cast(&(*in_array)[0]);
	pout_array = thrust::raw_pointer_cast(&(*out_array)[0]);
	trace;
}


// solve L * U * o_data = i_data
void MyILU0::HostPrecond(const ValueType *i_data, ValueType *o_data){
	float* v = new float[numRows];
	memcpy(v, i_data, numRows*sizeof(float));

	float *x = o_data;
	//float *y = i_data;

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

	delete [] v;
}

void MyILU0::DevPrecond(const ValueType *i_data, ValueType *o_data){


	cudaDeviceSynchronize();
	float alpha = 1.0f;

	float *v;
	checkCudaErrors(cudaMalloc((void**)&v, numRows * sizeof(float)));

	// L v = y
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info, i_data, v);
	cudaDeviceSynchronize();
	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	// U x = v
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info, v, o_data);
	cudaDeviceSynchronize();

	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	checkCudaErrors(cudaFree(v));

}


void MyILU0::Initilize(const MySpMatrix &mySpM){

	status = new cusparseStatus_t();
	handle = new cusparseHandle_t();
	L_info = new cusparseSolveAnalysisInfo_t();
	U_info = new cusparseSolveAnalysisInfo_t();
	L_des = new cusparseMatDescr_t();
	U_des = new cusparseMatDescr_t();
	A_des = new cusparseMatDescr_t();


	this->numRows = mySpM.numRows;

	//warning: currently no permutation is adopted, i.e., the element on the diag must not be zero!
	leftILU(this->numRows, mySpM.val, mySpM.rowIndices, mySpM.indices,
                l_val, l_rowIndices, l_indices,
                u_val, u_rowIndices, u_indices);

	int l_nnz = l_rowIndices[numRows];
	int u_nnz = u_rowIndices[numRows];


	checkCudaErrors(cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));

	checkCudaErrors(cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz,
                                  cudaMemcpyHostToDevice));

	cusparseCreate(handle);
	int cuspase_version;
	cusparseGetVersion(*handle, &cuspase_version);
	// printf("The version of cusparse is %d\n", cuspase_version);


	assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
	//cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);

	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);
	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);

	printf("void MyILU0::Initilize() finished.\n");
}

void readCSR(int *m, int *n, int *nnz,
             int **rowPtr, int **colIdx, float **val,
             const char *filename)
{
  FILE *f;
  f = fopen(filename,"rb");
  if(!f) {
    fprintf(stdout,"Cannot open file: %s\n",filename);
    exit(-1);
  }

  fread(m, sizeof(int), 1, f);
  fread(n, sizeof(int), 1, f);
  fread(nnz, sizeof(int), 1, f);
  
  *rowPtr=(int*)malloc((*m+1)*sizeof(int));
  *colIdx=(int*)malloc((*nnz)*sizeof(int));
  *val=(float*)malloc((*nnz)*sizeof(float));
  
  fwrite(*rowPtr, sizeof(int), *m+1, f);
  fwrite(*colIdx, sizeof(int), *nnz, f);
  fwrite(*val, sizeof(float), *nnz, f);
  
  fclose(f);
}

// void MyILU0::Initilize(const MySpMatrix &mySpM){
// 
// 	status = new cusparseStatus_t();
// 	handle = new cusparseHandle_t();
// 	L_info = new cusparseSolveAnalysisInfo_t();
// 	U_info = new cusparseSolveAnalysisInfo_t();
// 	L_des = new cusparseMatDescr_t();
// 	U_des = new cusparseMatDescr_t();
// 	A_des = new cusparseMatDescr_t();
// 
// 
// 	this->numRows = mySpM.numRows;
// 
// 	//warning: currently no permutation is adopted, i.e., the element on the diag must not be zero!
// 	// leftILU(this->numRows, mySpM.val, mySpM.rowIndices, mySpM.indices,
//         //         l_val, l_rowIndices, l_indices,
//         //         u_val, u_rowIndices, u_indices);
// 
//         int nRowsFile, nColsFile, nnzFile;
//         readCSR(&nRowsFile, &nColsFile, &nnzFile,
//                 &l_rowIndices, &l_indices, &l_val, "csrFileL1G.dat");
//         assert(nRowsFile == mySpM.numRows);
//         assert(nColsFile == mySpM.numCols);
//         readCSR(&nRowsFile, &nColsFile, &nnzFile,
//                 &u_rowIndices, &u_indices, &u_val, "csrFileU1G.dat");
//         assert(nRowsFile == mySpM.numRows);
//         assert(nColsFile == mySpM.numCols);
// 
// 
// 	int l_nnz = l_rowIndices[numRows];
// 	int u_nnz = u_rowIndices[numRows];
// 
// 
// 	checkCudaErrors(cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz));
// 	checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
// 	checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));
// 	checkCudaErrors(cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz));
// 	checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
// 	checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));
// 
// 	checkCudaErrors(cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1),
//                                   cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz,
//                                   cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1),
//                                   cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz,
//                                   cudaMemcpyHostToDevice));
// 
// 	cusparseCreate(handle);
// 	int cuspase_version;
// 	cusparseGetVersion(*handle, &cuspase_version);
// 	printf("The version of cusparse is %d\n", cuspase_version);
// 
// 
// 	assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);
// 
// 	cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
// 	cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
// 	cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_UNIT);
// 	cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
// 	cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
// 	cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
// 	cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	//cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
// 	//cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
// 	cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	*status = cusparseScsrsv_analysis
//           (*handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//            numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
// 	assert(*status == CUSPARSE_STATUS_SUCCESS);
// 	*status = cusparseScsrsv_analysis
//           (*handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//            numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
// 	assert(*status == CUSPARSE_STATUS_SUCCESS);
// 
// 	printf("void MyILU0::Initilize() finished.\n");
// }

void MyDIAG::HostPrecond(const ValueType *i_data, ValueType *o_data){
	for(int i=0; i<numRows; ++i){
		o_data[i] = i_data[i] * this->val[i];
	}
}

__global__ 
void diagGpu(const int numRows, const float *val, const float *i_data, float *o_data){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=tid; i<numRows; i += stride){
		o_data[tid] = i_data[i] * val[i];
	}
}

void MyDIAG::DevPrecond(const ValueType *i_data, ValueType *o_data){
	int dim_block = 512;
	int dim_grid = (numRows + (dim_block - 1)) / dim_block;
	if(dim_grid > 65535)
		dim_grid = 65535;
	
	diagGpu<<<dim_grid, dim_block>>>(numRows, d_val, i_data, o_data);
}


void MyDIAG::Initilize(const MySpMatrix &mySpM){
	this->numRows = mySpM.numRows;

	this->val = new ValueType[numRows];
	trace;

	for(int i = 0; i <numRows; ++i){
		int lb = mySpM.rowIndices[i];
		int ub = mySpM.rowIndices[i+1];
		int j;
		float diagVal = 0.0f;
		int dist = 2 * mySpM.numRows;

		for(j=lb; j<ub; ++j){
			int curCol = mySpM.indices[j];

			if(abs(curCol - i) < dist){
				dist = abs(curCol - i);
				diagVal = mySpM.val[j];
			}
		}

		assert(!Equal(diagVal, 0.0f));

		diagVal = 1.0f / diagVal;

		this->val[i] = diagVal;
	}
	assert(numRows == mySpM.numRows);
	trace;

	checkCudaErrors(cudaMalloc((void**)&this->d_val, mySpM.numRows * sizeof(ValueType)));
	checkCudaErrors(cudaMemcpy(this->d_val, this->val, mySpM.numRows * sizeof(ValueType),
                                  cudaMemcpyHostToDevice));
}



void addUnitCSR(int numRows, int numCols,
                int **rowPtrIn, int **colIdxIn, float **valIn)
{
  int *rowPtr=*rowPtrIn, *colIdx=*colIdxIn;
  float *val=*valIn;

  int nnz=rowPtr[numRows];
  int *rowPtrNew=(int*)malloc((numRows+1)*sizeof(int));
  int *colIdxNew=(int*)malloc((nnz+min(numRows,numCols))*sizeof(int));
  float *valNew=(float*)malloc((nnz+min(numRows,numCols))*sizeof(float));

  int k=0;
  rowPtrNew[0] = 0;
  for(int i=0; i<numRows; i++) {
    int diagSet=0;
    int lb=rowPtr[i], ub=rowPtr[i+1];
    if(lb == ub) {
      colIdxNew[k] = i;  valNew[k] = 1.0;
      k++;
    }
    else {
      for(int j=lb; j<ub; j++) {
        if(colIdx[j] < i) {
          colIdxNew[k] = colIdx[j];  valNew[k] = val[j];
          k++;
        }
        else if(colIdx[j] == i) {
          colIdxNew[k] = colIdx[j];  valNew[k] = (val[j]==0.0) ? 1.0 : val[j];
          k++;
          diagSet = 1;
        }
        else {
          if(diagSet == 0) {
            colIdxNew[k] = i;  valNew[k] = 1.0;
            k++;
            diagSet = 1;
          }
          colIdxNew[k] = colIdx[j];  valNew[k] = val[j];
          k++;
        }
      }
      if(diagSet == 0) {
        colIdxNew[k] = i;  valNew[k] = 1.0;
        k++;
        diagSet = 1;
      }
    }
    rowPtrNew[i+1] = k;
  }
  free(rowPtr); free(colIdx); free(val);
  *rowPtrIn = rowPtrNew;
  *colIdxIn = colIdxNew;
  *valIn = valNew;
}


// solve L * U * o_data = i_data
void MyILUK::HostPrecond(const ValueType *i_data, ValueType *o_data){
	float* v = new float[numRows];
	memcpy(v, i_data, numRows*sizeof(float));

	float *x = o_data;
	//float *y = i_data;

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

	delete [] v;
}

void MyILUK::DevPrecond(const ValueType *i_data, ValueType *o_data){


	cudaDeviceSynchronize();
	float alpha = 1.0f;

	float *v;
	checkCudaErrors(cudaMalloc((void**)&v, numRows * sizeof(float)));

	// L v = y
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info, i_data, v);
	cudaDeviceSynchronize();
	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	// U x = v
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info, v, o_data);
	cudaDeviceSynchronize();

	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	checkCudaErrors(cudaFree(v));

}
/////////////////////////////////////////////////////////

int CSRcs( int n, float *a, int *ja, int *ia, csptr mat )
{
/*----------------------------------------------------------------------
| Convert CSR matrix to SpaFmt struct
|----------------------------------------------------------------------
| on entry:
|==========
| a, ja, ia  = Matrix stored in CSR format (with FORTRAN indexing).
|
| On return:
|===========
|
| ( mat )  =  Matrix stored as SpaFmt struct. (C indexing)
|
|       integer value returned:
|             0   --> successful return.
|             1   --> memory allocation error.
|--------------------------------------------------------------------*/
  int i, j, j1, len;//, col, nnz;
  float *bra;
  int *bja;
  /*    setup data structure for mat (csptr) struct */
  setupCS( mat, n, 1 );

  for (j=0; j<n; j++) { // j is row index, 0-based
    len = ia[j+1] - ia[j];
    mat->nzcount[j] = len;
    if (len > 0) {
      bja = (int *) Malloc( len*sizeof(int), "CSRcs" );
      bra = (float *) Malloc( len*sizeof(float), "CSRcs" );
      i = 0;
      for (j1=ia[j]; j1<ia[j+1]; j1++) { // XXLiu: modifications made here for the shift of one.
        bja[i] = ja[j1]; // XXLiu: modifications made here for the shift of one.
        bra[i] = a[j1];
        i++;
      }
      mat->ja[j] = bja;
      mat->ma[j] = bra;
    }
  }
  return 0;
}
/*---------------------------------------------------------------------
|     end of CSRcs
|--------------------------------------------------------------------*/

int csCSR( csptr mat, float **valIn, int **rowPtrIn, int **colIdxIn )
{
  int n = mat->n;
  int i, j, k, len, nnz=0;
  for(i=0; i<n; i++)
    nnz += mat->nzcount[i];

  *valIn = (float*)malloc(nnz*sizeof(float));
  *colIdxIn = (int*)malloc(nnz*sizeof(int));
  *rowPtrIn = (int*)malloc((n+1)*sizeof(int));
  
  float *val = *valIn;
  int *colIdx=*colIdxIn, *rowPtr=*rowPtrIn;

  rowPtr[0] = 0;
  for(i=1; i<=n; i++)
    rowPtr[i] = rowPtr[i-1] + mat->nzcount[i-1];

  k = 0;
  for(i=0; i<n; i++) {
    len = mat->nzcount[i];
    if( len ) {
      for(j=0; j<len; j++) {
        val[k] = mat->ma[i][j];
        colIdx[k] = mat->ja[i][j];
        j++;
      }
    }
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
// void MyILUPP::Initilize(const MySpMatrix &mySpM)
// {
//   printf("ERROR: This function has not been implemented, \n");
//   printf("       and therefore is not supposed to be run.\n");
//   exit(-1);
// }

void MyILUPP::Initilize(const MySpMatrix &mySpM){

	status = new cusparseStatus_t();
	handle = new cusparseHandle_t();
	L_info = new cusparseSolveAnalysisInfo_t();
	U_info = new cusparseSolveAnalysisInfo_t();
	L_des = new cusparseMatDescr_t();
	U_des = new cusparseMatDescr_t();
	A_des = new cusparseMatDescr_t();

	this->numRows = mySpM.numRows;
        // XXLiu added the following section adapted from ITSOL_2.
        csptr csmat = NULL;  /* matrix in csr formt             */
        csmat = (csptr)Malloc( sizeof(SparMat), "MyILUPP::Initilize" );
        CSRcs( mySpM.numRows, mySpM.val, mySpM.indices, mySpM.rowIndices, csmat );
        iluptr lu = NULL;    /* ilu preconditioner structure    */
        lu = (iluptr)Malloc( sizeof(ILUSpar), "MyILUPP::Initilize" );
        /*-------------------- call ILUK preconditioner set-up  */
        int lfil = 10; // level of fill
        int ierr = ilukC(lfil, csmat, lu, stdout );
        if( ierr == -2 ) {
          fprintf( stdout, "zero diagonal element found...\n" );
          cleanILU( lu );
          exit(-1);
        } else if( ierr != 0 ) {
          fprintf( stdout, "*** iluk error, ierr != 0 ***\n" );
          exit(-1);
        }
        csCSR(lu->L, &Lval_ITSOL, &LrowIndices_ITSOL, &Lindices_ITSOL);
        csCSR(lu->U, &Uval_ITSOL, &UrowIndices_ITSOL, &Uindices_ITSOL);
        


	//warning: currently no permutation is adopted, i.e., the element on the diag must not be zero!
	leftILU(this->numRows, mySpM.val, mySpM.rowIndices, mySpM.indices,
                l_val, l_rowIndices, l_indices,
                u_val, u_rowIndices, u_indices);

	int l_nnz = l_rowIndices[numRows];
	int u_nnz = u_rowIndices[numRows];


	checkCudaErrors(cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));

	checkCudaErrors(cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz,
                                  cudaMemcpyHostToDevice));

	cusparseCreate(handle);
	int cuspase_version;
	cusparseGetVersion(*handle, &cuspase_version);
	// printf("The version of cusparse is %d\n", cuspase_version);


	assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
	//cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);

	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);
	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);

	printf("void MyILUPP::Initilize() finished.\n");
}

MyILUPP::~MyILUPP()
{
  delete [] tmpvector;
}

void MyILUPP::Initilize(const MySpMatrixDouble &PrLeft_mySpM,
                        const MySpMatrixDouble &PrRight_mySpM,
                        const MySpMatrix &PrMiddle_mySpM,
                        const MySpMatrix &PrPermRow,
                        const MySpMatrix &PrPermCol,
                        const MySpMatrixDouble &PrLscale,
                        const MySpMatrixDouble &PrRscale)
{

  status = new cusparseStatus_t();
  handle = new cusparseHandle_t();
  L_info = new cusparseSolveAnalysisInfo_t();
  U_info = new cusparseSolveAnalysisInfo_t();
  L_des = new cusparseMatDescr_t();
  U_des = new cusparseMatDescr_t();
  A_des = new cusparseMatDescr_t();
 
  this->numRows = PrLeft_mySpM.numRows;
  l_val_double = PrLeft_mySpM.val;
  l_rowIndices = PrLeft_mySpM.rowIndices;
  l_indices = PrLeft_mySpM.indices;
  u_val_double = PrRight_mySpM.val;
  u_rowIndices = PrRight_mySpM.rowIndices;
  u_indices = PrRight_mySpM.indices;
  
  p_val = PrPermRow.val;
  p_rowIndices = PrPermRow.rowIndices;
  permRow_indices = PrPermRow.indices;
  permCol_indices = PrPermCol.indices;

  middle_val = PrMiddle_mySpM.val;
  middle_rowIndices = PrMiddle_mySpM.rowIndices;
  middle_indices = PrMiddle_mySpM.indices;

  lscale_val = PrLscale.val;
  rscale_val = PrRscale.val;

  tmpvector = new float[numRows];
  int l_nnz = l_rowIndices[numRows];
  int u_nnz = u_rowIndices[numRows];

  checkCudaErrors(cudaMalloc((void**)&d_l_val_double, sizeof(double)*l_nnz));
  checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
  checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));

  checkCudaErrors(cudaMalloc((void**)&d_u_val_double, sizeof(double)*u_nnz));
  checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
  checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));

  checkCudaErrors(cudaMemcpy(d_l_val_double, l_val_double, sizeof(double)*l_nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_u_val_double, u_val_double, sizeof(double)*u_nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz, cudaMemcpyHostToDevice));
 
  cusparseCreate(handle);
  int cuspase_version;
  cusparseGetVersion(*handle, &cuspase_version);
  // printf("The version of cusparse is %d\n", cuspase_version);

// 	assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
// 	assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);
// 
// 	cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
// 	cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
// 	cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_UNIT);
// 	cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
// 	cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
// 	cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
// 	cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	//cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
// 	//cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
// 	cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);
// 
// 	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
// 	assert(*status == CUSPARSE_STATUS_SUCCESS);
// 	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
// 	assert(*status == CUSPARSE_STATUS_SUCCESS);
// 
// 	printf("void MyILUPP::Initilize() finished.\n");
}
// solve L * U * o_data = i_data
void MyILUPP::HostPrecond(const ValueType *i_data, ValueType *o_data){
  //float* v = new float[numRows];
  float *v = tmpvector;
	memcpy(v, i_data, numRows*sizeof(float));

	float *x = o_data;
	//float *y = i_data;

	// sovle Ux = v, backward substitution
	//memcpy(x, v, numRows*sizeof(float));
	for(int i=numRows-1; i>=0; i--){
		int lb = u_rowIndices[i];// lb is the low bound for the L, not U
		int ub = u_rowIndices[i+1];
                for(int j=ub-1; j>lb; j--){
                  v[i] -= u_val_double[j] * v[u_indices[j]];
			// if(u_indices[j] <= i){// search to the L matrix
			// 	break;
			// }
			// else{
                        //   v[i] -= u_val[j] * v[u_indices[j]];
			// }
		}
		//assert(u_indices[j] == i);// if the element on the diagnoal of U matrix, just ignore it without update the orginal value of that row
		// if(u_indices[j] == i && !Equal(u_val[j], 0)){
                //   v[i] /= u_val[j];
		// }
	}
        for(int i=0; i<numRows; i++)  x[i] = v[ permCol_indices[i] ];
        for(int i=0; i<numRows; i++)  v[i] = x[ permRow_indices[i] ];

	// solve Lv = y, forward substitution
	for(int i=0; i<numRows; ++i){
		int lb = l_rowIndices[i];
		int ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
		
		for(int j=lb; j<ub-1; j++){
                  v[i] -= l_val_double[j] * v[l_indices[j]];
                        // if(l_indices[j] >= i){
			// 	break;
			// }
			// else{
			// 	v[i] -= l_val[j] * v[l_indices[j]];
			// }
		}
		// assert(l_indices[j] == i);// int the L matrix, the location of the diagonal element
		// assert(!Equal(l_val[j], 0));
		// v[i] /= l_val[j];
	}

	memcpy(x, v, numRows*sizeof(float));
        
	// memcpy(v, x, numRows*sizeof(float));
	// for(int i=0; i<numRows; ++i)  x[i] = v[ perCol_indices[i] ];

	//delete [] v;
}

void MyILUPP::DevPrecond(const ValueType *i_data, ValueType *o_data){


	cudaDeviceSynchronize();
	float alpha = 1.0f;

	float *v;
	checkCudaErrors(cudaMalloc((void**)&v, numRows * sizeof(float)));

	// L v = y
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info, i_data, v);
	cudaDeviceSynchronize();
	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	// U x = v
	*status = cusparseScsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &alpha, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info, v, o_data);
	cudaDeviceSynchronize();

	if(*status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", *status);
		assert(false);
	}

	checkCudaErrors(cudaFree(v));

}


// solve L * U * o_data = i_data
void MyILUPP::HostPrecond_rhs(const ValueType *i_data, ValueType *o_data){
  this->HostPrecond_left(i_data, o_data);
  // float *x = o_data;
  // float* v = new float[numRows];
  // memcpy(v, i_data, numRows*sizeof(float));

  // // solve Lv = y, forward substitution
  // for(int i=0; i<numRows; ++i){
  //   int lb = l_rowIndices[i];
  //   int ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
  //   
  //   for(int j=lb; j<ub-1; j++)
  //     v[i] -= l_val[j] * v[l_indices[j]];
  // }
  // for(int i=0; i<numRows; ++i)  x[i] = v[ permRow_indices[i] ];

  // delete [] v;
}

void MyILUPP::HostPrecond_starting_value(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  //float* v = new float[numRows];
  float *v = tmpvector;
  
  for(int i=0; i<numRows; ++i)  v[i] = i_data[i]*rscale_val[i];
  for(int i=0; i<numRows; ++i)  x[ permCol_indices[i] ] = v[i];
  for(int i=0; i<numRows; i++) {
    v[i] = 0;
    int lb = u_rowIndices[i];
    int ub = u_rowIndices[i+1];
    for(int j=lb; j<ub; j++)
      v[i] += u_val_double[j] * x[ u_indices[j] ];
  }
  for(int i=0; i<numRows; ++i)  x[i] = v[i]/middle_val[i];

  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPP::HostPrecond_left(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  //float *v = new float[numRows];
  float *v = tmpvector;

  int i, j, lb, ub;
  for(i=0; i<numRows; ++i)  v[i] = i_data[i]/lscale_val[i];
  //memcpy(v, x, numRows*sizeof(float));
  for(i=0; i<numRows; ++i)  x[i] = v[ permRow_indices[i] ];

  // solve Lv = y, forward substitution
  for(i=0; i<numRows; ++i){
    lb = l_rowIndices[i];
    ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
    for(j=lb; j<ub-1; j++)
      x[i] -= l_val_double[j] * x[l_indices[j]];
    x[i] /= l_val_double[ub-1];
  }

  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPP::HostPrecond_right(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  // float* v = new float[numRows];
  float *v = tmpvector;
  
  int i, j, lb, ub;
  for(i=0; i<numRows; ++i)  v[i] = i_data[i]*middle_val[i];

  for(i=numRows-1; i>=0; i--){
    lb = u_rowIndices[i];
    ub = u_rowIndices[i+1];
    for(j=lb+1; j<ub; j++)
      v[i] -= u_val_double[j] * v[ u_indices[j] ];
    v[i] /= u_val_double[lb];
  }
  for(i=0; i<numRows; ++i)  x[i] = v[ permCol_indices[i] ] / rscale_val[i];
  
  // for(int i=0; i<numRows; ++i)  x[i] /= rscale_val[i];

  // delete [] v;
}


void MyILUPP::DevPrecond_rhs(float *i_data, float *o_data)
{

}

void MyILUPP::DevPrecond_right(float *i_data, float *o_data)
{

}

void MyILUPP::DevPrecond_left(float *i_data, float *o_data)
{

}

void MyILUPP::DevPrecond_starting_value(float *i_data, float *o_data)
{

}
        
///////////////////////////////////////////////////////////////////////////

MyILUPPfloat::~MyILUPPfloat()
{
  checkCudaErrors(cudaFree(d_l_val_double));
  checkCudaErrors(cudaFree(d_l_rowIndices));
  checkCudaErrors(cudaFree(d_l_indices));

  checkCudaErrors(cudaFree(d_u_val_double));
  checkCudaErrors(cudaFree(d_u_rowIndices));
  checkCudaErrors(cudaFree(d_u_indices));

  checkCudaErrors(cudaFree(d_p_val));
  checkCudaErrors(cudaFree(d_p_rowIndices));
  checkCudaErrors(cudaFree(d_permRow_indices));
  checkCudaErrors(cudaFree(d_permCol_indices));

  checkCudaErrors(cudaFree(d_middle_val));
  checkCudaErrors(cudaFree(d_lscale_val));
  checkCudaErrors(cudaFree(d_rscale_val));

  delete [] tmpvector;
  checkCudaErrors(cudaFree(d_tmpvector_single));
  checkCudaErrors(cudaFree(d_tmpvector_double));
  checkCudaErrors(cudaFree(d_tmp_solution_double));


  free(H);   free(s);   free(cs);   free(sn);
  checkCudaErrors(cudaFree(d_r));
  checkCudaErrors(cudaFree(d_rr));
  checkCudaErrors(cudaFree(d_bb));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_w));
  checkCudaErrors(cudaFree(d_ww));
}

void MyILUPPfloat::Initilize(const MySpMatrix &mySpM)
{
  printf("ERROR: This function has not been implemented, and therefore is not supposed to be run.\n");
  exit(-1);
}

// Called by ((MyILUPPfloat *) Precond)->Initilize(...)
// in void gmresInterfacePGfloat::setPrecondPG(...)
void MyILUPPfloat::Initilize(const MySpMatrixDouble &PrLeft_mySpM,
                             const MySpMatrixDouble &PrRight_mySpM,
                             const MySpMatrix &PrMiddle_mySpM,
                             const MySpMatrix &PrPermRow,
                             const MySpMatrix &PrPermCol,
                             const MySpMatrix &PrLscale,
                             const MySpMatrix &PrRscale,
                             int m) // m is GMRES restart number
{
  status = new cusparseStatus_t();
  handle = new cusparseHandle_t();
  L_info = new cusparseSolveAnalysisInfo_t();
  U_info = new cusparseSolveAnalysisInfo_t();
  L_des = new cusparseMatDescr_t();
  U_des = new cusparseMatDescr_t();
  A_des = new cusparseMatDescr_t();

  this->numRows = PrLeft_mySpM.numRows;
  //---
  l_val = PrLeft_mySpM.val;
  l_rowIndices = PrLeft_mySpM.rowIndices;
  l_indices = PrLeft_mySpM.indices;

  u_val = PrRight_mySpM.val;
  u_rowIndices = PrRight_mySpM.rowIndices;
  u_indices = PrRight_mySpM.indices;
  
  l_nnz = l_rowIndices[numRows];
  u_nnz = u_rowIndices[numRows];

  checkCudaErrors(cudaMalloc((void**)&d_l_val_double, sizeof(double)*l_nnz));
  checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
  checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));

  checkCudaErrors(cudaMalloc((void**)&d_u_val_double, sizeof(double)*u_nnz));
  checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
  checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));
   
  checkCudaErrors(cudaMemcpy(d_l_val_double, l_val, sizeof(double)*l_nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_u_val_double, u_val, sizeof(double)*u_nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz, cudaMemcpyHostToDevice));
  //---
  p_val = PrPermRow.val;
  p_rowIndices = PrPermRow.rowIndices;
  permRow_indices = PrPermRow.indices;
  permCol_indices = PrPermCol.indices;

  checkCudaErrors(cudaMalloc((void**)&d_p_val, sizeof(double)*numRows));
  double *p_val_double=new double [numRows];
  for(int i=0; i<numRows; i++) p_val_double[i] = 1.0;
  checkCudaErrors(cudaMemcpy(d_p_val, p_val_double, sizeof(double)*numRows, cudaMemcpyHostToDevice));
  delete [] p_val_double;

  checkCudaErrors(cudaMalloc((void**)&d_p_rowIndices, sizeof(int)*(numRows+1)));
  checkCudaErrors(cudaMalloc((void**)&d_permRow_indices, sizeof(int)*numRows));
  checkCudaErrors(cudaMalloc((void**)&d_permCol_indices, sizeof(int)*numRows));
  checkCudaErrors(cudaMemcpy(d_p_rowIndices, p_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_permRow_indices, permRow_indices, sizeof(int)*numRows, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_permCol_indices, permCol_indices, sizeof(int)*numRows, cudaMemcpyHostToDevice));
  //---
  middle_val = PrMiddle_mySpM.val;
  middle_rowIndices = PrMiddle_mySpM.rowIndices;
  middle_indices = PrMiddle_mySpM.indices;
  lscale_val = PrLscale.val;
  rscale_val = PrRscale.val;

  checkCudaErrors(cudaMalloc((void**)&d_middle_val, sizeof(float)*numRows));
  checkCudaErrors(cudaMalloc((void**)&d_lscale_val, sizeof(float)*numRows));
  checkCudaErrors(cudaMalloc((void**)&d_rscale_val, sizeof(float)*numRows));
  checkCudaErrors(cudaMemcpy(d_middle_val, middle_val, sizeof(float)*numRows, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_lscale_val, lscale_val, sizeof(float)*numRows, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_rscale_val, rscale_val, sizeof(float)*numRows, cudaMemcpyHostToDevice));
  //---
  tmpvector = new float[numRows];
  checkCudaErrors(cudaMalloc((void**)&d_tmpvector_single, sizeof(float)*numRows));
  checkCudaErrors(cudaMalloc((void**)&d_tmpvector_double, sizeof(double)*numRows));
  checkCudaErrors(cudaMalloc((void**)&d_tmp_solution_double, sizeof(double)*numRows));
  
  cusparseCreate(handle);
  int cuspase_version;
  cusparseGetVersion(*handle, &cuspase_version);
  printf("The version of cusparse is %d\n", cuspase_version);

  assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
  assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);

  cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);

  cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);
 
  cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
  //cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
  //cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);

  *status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
  assert(*status == CUSPARSE_STATUS_SUCCESS);
  *status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
  assert(*status == CUSPARSE_STATUS_SUCCESS);

  //---- For GMRES
  checkCudaErrors(cudaMalloc((void**) &d_r, numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_rr, numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_bb, numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_y, numRows*sizeof(float)));

  s = (float*) malloc((m+1)*sizeof(float));
  cs = (float*) malloc((m+1)*sizeof(float));
  sn = (float*) malloc((m+1)*sizeof(float));
  H = (float*) malloc(m*(m+1)*sizeof(float));
  
  checkCudaErrors(cudaMalloc((void**) &d_v, (m+1)*numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_w, numRows*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_ww, numRows*sizeof(float)));

  printf("void MyILUPPfloat::Initilize() finished.\n");
}

void MyILUPPfloat::HostPrecond(const ValueType *i_data, ValueType *o_data){
  printf("ERROR: This function has not been implemented,\n");
  printf("       and therefore is not supposed to be run.\n");
  exit(-1);
}


void MyILUPPfloat::DevPrecond(const ValueType *i_data, ValueType *o_data){
  printf("ERROR: This function has not been implemented,\n");
  printf("       and therefore is not supposed to be run.\n");
  exit(-1);
}


void MyILUPPfloat::HostPrecond_rhs(const ValueType *i_data, ValueType *o_data){
  this->HostPrecond_left(i_data, o_data);
}

void MyILUPPfloat::HostPrecond_starting_value(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  //float* v = new float[numRows];
  float *v = tmpvector;
  
  for(int i=0; i<numRows; ++i)  v[i] = i_data[i]*rscale_val[i];
  for(int i=0; i<numRows; ++i)  x[ permCol_indices[i] ] = v[i];
  for(int i=0; i<numRows; i++) {
    v[i] = 0;
    int lb = u_rowIndices[i];
    int ub = u_rowIndices[i+1];
    for(int j=lb; j<ub; j++)
      v[i] += u_val[j] * x[ u_indices[j] ];
  }
  for(int i=0; i<numRows; ++i)  x[i] = v[i]/middle_val[i];

  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPPfloat::HostPrecond_left(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  //float *v = new float[numRows];
  float *v = tmpvector;

  int i, j, lb, ub;
  for(i=0; i<numRows; ++i)  v[i] = i_data[i]/lscale_val[i];
  //memcpy(v, x, numRows*sizeof(float));
  for(i=0; i<numRows; ++i)  x[i] = v[ permRow_indices[i] ];

  // solve Lv = y, forward substitution
  for(i=0; i<numRows; ++i){
    lb = l_rowIndices[i];
    ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
    for(j=lb; j<ub-1; j++)
      x[i] -= l_val[j] * x[l_indices[j]];
    x[i] /= l_val[ub-1];
  }

  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPPfloat::HostPrecond_right(const ValueType *i_data, ValueType *o_data){
  float *x = o_data;
  //float* v = new float[numRows];
  float *v = tmpvector;
  
  int i, j, lb, ub;
  for(i=0; i<numRows; ++i)  v[i] = i_data[i]*middle_val[i];

  for(i=numRows-1; i>=0; i--){
    lb = u_rowIndices[i];
    ub = u_rowIndices[i+1];
    for(j=lb+1; j<ub; j++)
      v[i] -= u_val[j] * v[ u_indices[j] ];
    v[i] /= u_val[lb];
  }
  for(i=0; i<numRows; ++i)  x[i] = v[ permCol_indices[i] ] / rscale_val[i];
  
  //for(int i=0; i<numRows; ++i)  x[i] /= rscale_val[i];
  //delete [] v;
}

//---
void MyILUPPfloat::DevPrecond_rhs(float *i_data, float *o_data){
  this->DevPrecond_left(i_data, o_data);
}


__global__ void kernel_divide_S2D(double *outArr, float *inArr, float *scaleArr, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float in, scale;
  double out;
  if(idx < N) {
    in = inArr[idx];
    scale = scaleArr[idx];
    
    out = in/scale; // difference is here

    outArr[idx] = out;
  }
}

__global__ void kernel_divide_S
(float *outArr, float *inArr, float *scaleArr, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float in, scale;
  double out;
  if(idx < N) {
    in = inArr[idx];
    scale = scaleArr[idx];

    out = in/scale; // difference is here

    outArr[idx] = out;
  }
}
__global__ void kernel_perm_precondLeft_S2D(double *w, float *v, int *colIdxArr, int n)
{
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int colIdx;
    float vtmp;
    if (tid < n) {
        colIdx = colIdxArr[tid]; // Coalesced read
        vtmp = v[colIdx]; // Not coalesced read
        w[tid] = vtmp; // Coalesced write
    }
}

__global__ void kernel_divide_perm_precondLeft_S2D
(double *outArr, float *inArr, float *scaleArr, int *colIdxArr, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int colIdx;
  float in, scale;
  double out;
  if(tid < n) {
    colIdx = colIdxArr[tid]; // Coalesced read

    in = inArr[colIdx]; // Not coalesced read
    scale = scaleArr[colIdx]; // Not coalesced read

    out = in/scale; // difference is here

    outArr[tid] = out;
  }
}

__global__ void kernel_divide_D2S(float *outArr, double *inArr, float *scaleArr, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float out, scale;
  double in;
  if(idx < N) {
    in = inArr[idx];
    scale = scaleArr[idx];
    
    out = in/scale; // difference is here

    outArr[idx] = out;
  }
}

__global__ void kernel_perm_divide_precondRight_D2S
(float *w, double *v, float *scaleArr, int *colIdxArr, int n)
{
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int colIdx;
    float vtmp, scale;
    if (tid < n) {
        colIdx = colIdxArr[tid]; // Coalesced read
        vtmp = v[colIdx]; // Not coalesced read
        scale = scaleArr[tid]; // Coalesced read

        vtmp /= scale;

        w[tid] = vtmp; // Coalesced write
    }
}


__global__ void kernel_multiply_S2D(double *outArr, float *inArr, float *scaleArr, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float in, scale;
  double out;
  if(idx < N) {
    in = inArr[idx];
    scale = scaleArr[idx];
    
    out = in*scale; // difference is here

    outArr[idx] = out;
  }
}

__global__ void kernel_multiply_perm_startingValue_S2D
(double *outArr, float *inArr, float *scaleArr, int *colIdxArr, int n)
{
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int colIdx;
  float vtmp, scale;
  if(tid < n) {
    vtmp = inArr[tid]; // Coalesced read
    colIdx = colIdxArr[tid]; // Coalesced read    
    scale = scaleArr[tid]; // Coalesced read    

    vtmp *= scale;

    outArr[colIdx] = vtmp; // Not coalesced write
  }
}

__global__ void kernel_D2S(float *outArr, double *inArr, int N)
{
  double tmp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < N) {
    tmp = inArr[idx];
    outArr[idx] = tmp;
  }
}

// a reverse operation of right preconditioning
void MyILUPPfloat::DevPrecond_starting_value(float *i_data, float *o_data){
  double one=1.0, zero=0.0;
  double *v = d_tmpvector_double;
  
  // host: // for(int i=0; i<numRows; ++i)  v[i] = i_data[i]*rscale_val[i];
  // host: // for(int i=0; i<numRows; ++i)  x[ permCol_indices[i] ] = v[i]; // ???
  // kernel_multiply_S2D<<<(numRows+255)/256, 256>>>(v, i_data, d_rscale_val, numRows);
  // cusparseDcsrmv(*handle, CUSPARSE_OPERATION_TRANSPOSE, numRows, numRows, numRows,
  //                &one, *A_des, d_p_val, d_p_rowIndices, d_permCol_indices,
  //                v, &zero, d_tmp_solution_double); // Note the transpose here.
  kernel_multiply_perm_startingValue_S2D<<<(numRows+255)/256, 256>>>
    (d_tmp_solution_double, i_data, d_rscale_val, d_permCol_indices, numRows);

  // host: // for(int i=0; i<numRows; i++) {
  // host: //   v[i] = 0;
  // host: //   int lb = u_rowIndices[i];
  // host: //   int ub = u_rowIndices[i+1];
  // host: //   for(int j=lb; j<ub; j++)
  // host: //     v[i] += u_val[j] * x[ u_indices[j] ];
  // host: // }
  cusparseDcsrmv(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, u_nnz,
                 &one, *A_des, d_u_val_double, d_u_rowIndices, d_u_indices,
                 d_tmp_solution_double, &zero, v);

  //for(int i=0; i<numRows; ++i)  x[i] = v[i]/middle_val[i];
  kernel_divide_D2S<<<(numRows+255)/256, 256>>>(o_data, v, d_middle_val, numRows);

  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPPfloat::DevPrecond_left(float *i_data, float *o_data){
  double one=1.0;//, zero=0.0;
  double *v = d_tmpvector_double;

  // host: // int i, j, lb, ub;
  // host: // for(i=0; i<numRows; ++i)  v[i] = i_data[i]/lscale_val[i];
  // host: // for(i=0; i<numRows; ++i)  x[i] = v[ permRow_indices[i] ];
  // kernel_divide_S2D<<<(numRows+255)/256, 256>>>(v, i_data, d_lscale_val, numRows);
  // cusparseDcsrmv(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, numRows,
  //                &one, *A_des, d_p_val, d_p_rowIndices, d_permRow_indices,
  //                v, &zero, d_tmp_solution_double);
 
  kernel_divide_S<<<(numRows+255)/256, 256>>>
    (d_tmpvector_single, i_data, d_lscale_val, numRows);
  kernel_perm_precondLeft_S2D<<<(numRows+255)/256, 256>>>
    (d_tmp_solution_double, d_tmpvector_single, d_permRow_indices, numRows);

  // kernel_divide_perm_precondLeft_S2D<<<(numRows+255)/256, 256>>>
  //   (d_tmp_solution_double, i_data, d_lscale_val, d_permRow_indices, numRows);

  cusparseDcsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &one,
                       *L_des, d_l_val_double, d_l_rowIndices, d_l_indices, *L_info,
                       d_tmp_solution_double, v);
  // solve Lv = y, forward substitution
  // for(i=0; i<numRows; ++i){
  //   lb = l_rowIndices[i];
  //   ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
  //   for(j=lb; j<ub-1; j++)
  //     x[i] -= l_val[j] * x[l_indices[j]];
  //   x[i] /= l_val[ub-1];
  // }

  kernel_D2S<<<(numRows+255)/256, 256>>>(o_data, v, numRows);
  //delete [] v;
}

// solve L * U * o_data = i_data
void MyILUPPfloat::DevPrecond_right(float *i_data, float *o_data){
  //float* v = new float[numRows];
  double *v = d_tmpvector_double;
  
  //int i, j, lb, ub;
  //for(i=0; i<numRows; ++i)  v[i] = i_data[i]*middle_val[i];
  double one=1.0;//, zero=0.0;
  kernel_multiply_S2D<<<(numRows+255)/256, 256>>>(v, i_data, d_middle_val, numRows);
  cusparseDcsrsv_solve(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, &one,
                       *U_des, d_u_val_double, d_u_rowIndices, d_u_indices, *U_info,
                       v, d_tmp_solution_double);
  //kernel_D2S<<<(numRows+255)/256, 256>>>(d_tmpvector_single, d_tmp_solution_double, numRows);
  // host: // for(i=numRows-1; i>=0; i--){
  // host: //   lb = u_rowIndices[i];
  // host: //   ub = u_rowIndices[i+1];
  // host: //   for(j=lb+1; j<ub; j++)
  // host: //     v[i] -= u_val[j] * v[ u_indices[j] ];
  // host: //   v[i] /= u_val[lb];
  // host: // }

  // host: // for(i=0; i<numRows; ++i)  x[i] = v[ permCol_indices[i] ] / rscale_val[i];
  // cusparseDcsrmv(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numRows, numRows,
  //                &one, *A_des, d_p_val, d_p_rowIndices, d_permCol_indices,
  //                d_tmp_solution_double, &zero, v);
  // kernel_divide_D2S<<<(numRows+255)/256, 256>>>(o_data, v, d_rscale_val, numRows);

  kernel_perm_divide_precondRight_D2S<<<(numRows+255)/256, 256>>>
    (o_data, d_tmp_solution_double, d_rscale_val, d_permCol_indices, numRows);
}
//-------------------------------------------------------------
void MyILUK::Initilize(const MySpMatrix &mySpM){

	status = new cusparseStatus_t();
	handle = new cusparseHandle_t();
	L_info = new cusparseSolveAnalysisInfo_t();
	U_info = new cusparseSolveAnalysisInfo_t();
	L_des = new cusparseMatDescr_t();
	U_des = new cusparseMatDescr_t();
	A_des = new cusparseMatDescr_t();

	this->numRows = mySpM.numRows;
        // XXLiu added the following section adapted from ITSOL_2.
        csptr csmat = NULL;  /* matrix in csr formt             */
        csmat = (csptr)Malloc( sizeof(SparMat), "MyILUK::Initilize" );
        CSRcs( mySpM.numRows, mySpM.val, mySpM.indices, mySpM.rowIndices, csmat );
        iluptr lu = NULL;    /* ilu preconditioner structure    */
        lu = (iluptr)Malloc( sizeof(ILUSpar), "MyILUK::Initilize" );
        /*-------------------- call ILUK preconditioner set-up  */
        int lfil = 10; // level of fill
        int ierr = ilukC(lfil, csmat, lu, stdout );
        if( ierr == -2 ) {
          fprintf( stdout, "zero diagonal element found...\n" );
          cleanILU( lu );
          exit(-1);
        } else if( ierr != 0 ) {
          fprintf( stdout, "*** iluk error, ierr != 0 ***\n" );
          exit(-1);
        }
        csCSR(lu->L, &Lval_ITSOL, &LrowIndices_ITSOL, &Lindices_ITSOL);
        csCSR(lu->U, &Uval_ITSOL, &UrowIndices_ITSOL, &Uindices_ITSOL);
        


	//warning: currently no permutation is adopted, i.e., the element on the diag must not be zero!
	leftILU(this->numRows, mySpM.val, mySpM.rowIndices, mySpM.indices,
                l_val, l_rowIndices, l_indices,
                u_val, u_rowIndices, u_indices);

	int l_nnz = l_rowIndices[numRows];
	int u_nnz = u_rowIndices[numRows];


	checkCudaErrors(cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz));
	checkCudaErrors(cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1)));
	checkCudaErrors(cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz));

	checkCudaErrors(cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz,
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1),
                                  cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz,
                                  cudaMemcpyHostToDevice));

	cusparseCreate(handle);
	int cuspase_version;
	cusparseGetVersion(*handle, &cuspase_version);
	// printf("The version of cusparse is %d\n", cuspase_version);


	assert(cusparseCreateSolveAnalysisInfo(L_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateSolveAnalysisInfo(U_info) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(L_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(U_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(A_des) == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(*L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*L_des, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(*L_des, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseSetMatIndexBase(*L_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(*U_des, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(*U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*U_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(*A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatFillMode(*A_des, CUSPARSE_FILL_MODE_UPPER);
	//cusparseSetMatDiagType(*A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(*A_des, CUSPARSE_INDEX_BASE_ZERO);

	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, l_nnz, *L_des, d_l_val, d_l_rowIndices, d_l_indices, *L_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);
	*status = cusparseScsrsv_analysis(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, u_nnz, *U_des, d_u_val, d_u_rowIndices, d_u_indices, *U_info);
	assert(*status == CUSPARSE_STATUS_SUCCESS);

	printf("void MyILUK::Initilize() finished.\n");
}

// XXLiu: My modified version
// solve L * U * o_data = i_data
// void MyILU0::HostPrecond(const ValueType *i_data, ValueType *o_data){
// 	float* v = new float[numRows];
// 	memcpy(v, i_data, numRows*sizeof(float));
// 
// 	float *x = o_data;
// 	//float *y = i_data;
// 
// 	// solve Lv = y, forward substitution
// 	for(int i=0; i<numRows; ++i){
// 		int lb = l_rowIndices[i];
// 		int ub = l_rowIndices[i+1];// ub is the up bound to U, not the L matrix
// 		int j=lb;
// 		for(; j<ub; ++j){
// 			if(l_indices[j] >= i){
// 				break;
// 			}
// 			else{
// 				v[i] -= l_val[j] * v[l_indices[j]];
// 			}
// 		}
// 		assert(l_indices[ub-1] == i);// int the L matrix, the location of the diagonal element
// 		assert(!Equal(l_val[ub-1], 0));
// 		v[i] /= l_val[ub-1];
// 	}
// 
// 	// sovle Ux = v, backward substitution
// 	memcpy(x, v, numRows*sizeof(float));
// 	for(int i=numRows-1; i>=0; --i){
// 		int lb = u_rowIndices[i];// lb is the low bound for the L, not U
// 		int ub = u_rowIndices[i+1];
// 		int j=ub - 1;
// 		for(; j>lb; --j){
// 			if(u_indices[j] <= i){// search to the L matrix
// 				break;
// 			}
// 			else{
// 				x[i] -= u_val[j] * x[u_indices[j]];
// 			}
// 		}
// 		assert(u_indices[lb] == i);// if the element on the diagnoal of U matrix, just ignore it without update the orginal value of that row
// 		assert(!Equal(u_val[lb], 0));
// 		//if(u_indices[j] == i && !Equal(u_val[j], 0)){
//                 x[i] /= u_val[lb];
//                         //}
// 	}
// 
// 	delete [] v;
// }































