/*!	\file
	\brief implement the functions for the decendants of the class of \p Preconditioner
*/

#include "preconditioner.h"

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
	cudaMemcpy(pin_array, i_data, numRows * sizeof(ValueType), cudaMemcpyDeviceToDevice);

	(*ainv_M)(*in_array, *out_array);

	cudaMemcpy(o_data, pout_array, numRows * sizeof(ValueType), cudaMemcpyDeviceToDevice);
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

	cudaMemcpy(ptr_row_offsets, mySpM.d_rowIndices, (mySpM.numRows+1)*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(ptr_column_indices, mySpM.d_indices, mySpM.rowIndices[mySpM.numRows]*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(ptr_values, mySpM.d_val, mySpM.rowIndices[mySpM.numRows]*sizeof(float), cudaMemcpyDeviceToDevice);

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
	cudaMalloc((void**)&v, numRows * sizeof(float));

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

	cudaFree(v);

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
	leftILU(this->numRows, mySpM.val, mySpM.rowIndices, mySpM.indices, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices);

	int l_nnz = l_rowIndices[numRows];
	int u_nnz = u_rowIndices[numRows];


	cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz);
	cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(numRows+1));
	cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz);
	cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz);
	cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(numRows+1));
	cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz);

	cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(numRows+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz, cudaMemcpyHostToDevice);

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

}

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

	cudaMalloc((void**)&this->d_val, mySpM.numRows * sizeof(ValueType));
	cudaMemcpy(this->d_val, this->val, mySpM.numRows * sizeof(ValueType), cudaMemcpyHostToDevice);
}



































