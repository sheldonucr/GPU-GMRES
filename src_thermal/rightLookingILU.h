/*!	\file
 	\brief define and implements the functions used for right-looking ILU factorization
 */

#include <iostream>
#include <assert.h>
#include <vector>

//! select different type of right-looking method for matrix factorization
#define KIJ
//#define IKJ

using namespace std;

//! complete LU decomposition without pivoting
/*!
	\param m the sparse matrix to be decomposted
	\param h_val the value array of the matrix of m
	\param l_val the value array of the output L matrix
	\param u_val the value array of the output U matrix
*/
void ILU_decomposition_nopermutation(const SpMatrix& m, const float* const h_val, const  int* const h_rowIndices, const  int* const h_indices, float*& l_val, int*& l_rowIndices, int*& l_indices, float*& u_val, int*& u_rowIndices, int*& u_indices){

	int val_size = h_rowIndices[m.numRows] - h_rowIndices[0];
	int rowIndice_size = m.numRows + 1;
	int indice_size = h_rowIndices[m.numRows] - h_rowIndices[0];

	float* m_val = (float*)malloc(val_size * sizeof(float));
	int* m_rowIndices = (int*)malloc((rowIndice_size) * sizeof( int));
	int* m_indices = (int*)malloc(indice_size * sizeof( int));

	memcpy(m_val, h_val, (val_size * sizeof(float)));
	memcpy(m_rowIndices, h_rowIndices, ((rowIndice_size) * sizeof( int)));
	memcpy(m_indices, h_indices, (indice_size * sizeof( int)));

#ifdef KIJ
	// KIJ version
	for(int k=0; k<m.numRows-1; ++k){

		int klb = m_rowIndices[k];
		int kub = m_rowIndices[k+1];
		int kk = klb;
		float pivotVal = 1.0;
		for(; kk<kub; ++kk){
			if(m_indices[kk] == k && !Equal(m_val[kk], 0)){// the diagnoal element
				pivotVal = m_val[kk];
				break;
			}
		}
		if(kk == kub || Equal(pivotVal, 0)){// if the element on the diagnoal is not exisi, just ignore this row
			continue;
		}

		for(int i=k+1; i<m.numRows; ++i){

			int ilb = m_rowIndices[i];
			int iub = m_rowIndices[i+1];

			int ii = ilb;
			for(; ii<iub; ++ii){
				if(m_indices[ii] == k && !Equal(m_val[ii], 0)){// the element in ith row and kth column
					m_val[ii] /= pivotVal;
					break;
				}
				else if(m_indices[ii] > k){
					ii = iub;
					break;
				}
				else{
				}
			}

			if(ii != iub){// a[i][k] != 0
				int ipos = ii+1;
				int kpos = kk+1;
				while(ipos != iub && kpos != kub){
					if(m_indices[ipos] == m_indices[kpos]/* && !Equal(m_val[ipos], 0) && !Equal(m_val[kpos], 0)*/){
						m_val[ipos] = m_val[ipos] - m_val[kpos] * m_val[ii];
						ipos++;
						kpos++;
					}
					else if(m_indices[ipos] > m_indices[kpos]){
						ipos++;
					}
					else if(m_indices[ipos] < m_indices[kpos]){
						kpos++;
					}
					else{
						assert(false);
					}
				}
			}
		}
	}

#elif defined IKJ
	// IKJ version without pivoting
	for(int i=1; i<m.numRows; ++i){
		// pivoting missing here
		int ilb = m_rowIndices[i];
		int iub = m_rowIndices[i+1];

		int ii = ilb;

		for(int k=0; k<i; ++k){

			int klb = m_rowIndices[k];
			int kub = m_rowIndices[k+1];


			// pivotVal <---> a[k][k]
			float pivotVal = 1.0f;
			int kk = klb;
			for(; kk<kub; ++kk){
				if(m_indices[kk] == k && !Equal(m_val[kk], 0)){// the diagonal element
					pivotVal = m_val[kk];
					break;
				}
			}

			if(kk == kub){
				continue;
			}

			// a[i][k] /= a[k][k], a[i][k] <---> m_val[ii]
			bool aik_nz = false;
			for(; ii<iub; ++ii){
				if(m_indices[ii] == k && !Equal(m_val[ii], 0)){
					m_val[ii] /= pivotVal;

					//printf("a[i][k]: (%u, %u, %f)\n\n", i, m_indices[ii], m_val[ii]);

					aik_nz = true;
					break;
				}
				else if(m_indices[ii] > k && !Equal(m_val[ii], 0)){
					break;
				}
				else{
				}
			}

			// a[i][k] not ZERO
			if(aik_nz == true){
				int ipos = ii+1;
				int kpos = kk+1;
				while(ipos != iub && kpos != kub){
					// a[i][j] not zero and a[k][j]
					if(m_indices[ipos] == m_indices[kpos]/* && !Equal(m_val[ipos], 0) && !Equal(m_val[kpos], 0)*/){
						m_val[ipos] = m_val[ipos] - m_val[ii]*m_val[kpos];

						//printf("a[i][j]: (%u, %u, %f)\n", i, m_indices[ipos], m_val[ipos]);

						ipos++;
						kpos++;
					}
					else if(m_indices[ipos] > m_indices[kpos]){
						kpos++;
					}
					else if(m_indices[ipos] < m_indices[kpos]){
						ipos++;
					}
					else{
						assert(false);
					}

				}
			}// end of j loop(modified implemented)
		}// end of k loop
	}// end of i loop

#endif

	// copy the L and U component int m to l and u matrix
	// the L and U matrix is NOT padded

	l_rowIndices = (int*)malloc((m.numRows+1) * sizeof(int));
	u_rowIndices = (int*)malloc((m.numRows+1) * sizeof(int));
	l_rowIndices[0] = 0;
	u_rowIndices[0] = 0;
	vector<float> vecLval, vecUval;
	vector<int> vecLindices, vecUindices;

	for(int i=0; i<m.numRows; ++i){
		int ld = m_rowIndices[i];
		int ud = m_rowIndices[i+1];

		int j=ld;
		for(;m_indices[j]<i && j<ud; ++j){// construct L matrix
			vecLval.push_back(m_val[j]);
			vecLindices.push_back(m_indices[j]);
		}
		vecLval.push_back(1);// the diagonal element of L matrix
		vecLindices.push_back(i);


		/*
		   if(m_indices[j] != i){// the diagnoal element of U not exisits
		//vecUval.push_back(vecUval.back());// random select an element as the diagonal element
		vecUval.push_back(1);// random select an element as the diagonal element
		vecUindices.push_back(i);
		}
		 */
		for(; j<ud; ++j){
			if(Equal(m_val[j], 0) && m_indices[j] == 0){// reach padded elements
				break;
			}
			vecUval.push_back(m_val[j]);
			vecUindices.push_back(m_indices[j]);
		}


		/*
		   for(int j=ld; j<ud; ++j){
		   if(Equal(m_val[j], 0) && m_indices[j] == 0){
		   continue;
		   }
		   else if(m_indices[j] < i){// L component
		   vecLval.push_back(m_val[j]);
		   vecLindices.push_back(m_indices[j]);

		   }
		   else if(m_indices[j] >= i){
		   vecUval.push_back(m_val[j]);
		   vecUindices.push_back(m_indices[j]);
		   }
		   else{
		   assert(false);
		   }
		   }
		   vecLval.push_back(1);// the diagnoal element of L set to 1
		   vecLindices.push_back(i);
		 */


		l_rowIndices[i+1] = vecLval.size();
		u_rowIndices[i+1] = vecUval.size();
	}
	assert(vecLval.size() == vecLindices.size());
	assert(vecUval.size() == vecUindices.size());

	l_val = (float*)malloc(vecLval.size() * sizeof(float));
	u_val = (float*)malloc(vecUval.size() * sizeof(float));
	l_indices = (int*)malloc(vecLindices.size() * sizeof(int));
	u_indices = (int*)malloc(vecUindices.size() * sizeof(int));
	memcpy(l_val, &vecLval[0], (vecLval.size() * sizeof(float)));
	memcpy(u_val, &vecUval[0], (vecUval.size() * sizeof(float)));
	memcpy(l_indices, &vecLindices[0], (vecLindices.size() * sizeof(int)));
	memcpy(u_indices, &vecUindices[0], (vecUindices.size() * sizeof(int)));

	/*
	   for(int i=0; i<vecLindices.size(); ++i)
	   printf("i: %d, column index: %d\n", i, vecLindices[i]);
	 */

	free(m_val);
	free(m_rowIndices);
	free(m_indices);
}


//! old ILU decomposition version, depreciated
void ILU_decomposition_old(const SpMatrix& m, const float* const h_val, const  int* const h_rowIndices, const  int* const h_indices, float*& m_val,  int*& m_rowIndices,  int*& m_indices, int*& permutationMatrix){
	int val_size = h_rowIndices[m.numRows] - h_rowIndices[0];
	int rowIndice_size = m.numRows + 1;
	int indice_size = h_rowIndices[m.numRows] - h_rowIndices[0];

	m_val = (float*)malloc(val_size * sizeof(float));
	m_rowIndices = ( int*)malloc((rowIndice_size) * sizeof( int));
	m_indices = ( int*)malloc(indice_size * sizeof( int));
	permutationMatrix = (int*)malloc(m.numRows * sizeof(int));

	memcpy(m_val, h_val, (val_size * sizeof(float)));
	memcpy(m_rowIndices, h_rowIndices, ((rowIndice_size) * sizeof( int)));
	memcpy(m_indices, h_indices, (indice_size * sizeof( int)));
	for(int i=0; i<m.numRows; ++i){
		permutationMatrix[i] = i;// initialize the permutation matrix as a identity matrix
	}

	// for debug, verify the order of the element
	for(int i=0; i<m.numRows; ++i){
		int lb = m_rowIndices[i];
		int ub = m_rowIndices[i+1];
		for(int j=lb; j<ub-1; ++j){
			if(m_val[j] != 0 && m_val[j+1] != 0){
				assert(m_indices[j] < m_indices[j+1]);
			}
		}
	}


#ifdef IKJ
	// IKJ version without pivoting
	for(int i=1; i<m.numRows; ++i){
		// pivoting missing here
		// TODO
		int ilb = m_rowIndices[i];
		int iub = m_rowIndices[i+1];

		int ii = ilb;

		for(int k=0; k<i; ++k){

			int klb = m_rowIndices[k];
			int kub = m_rowIndices[k+1];


			// pivotVal <---> a[k][k]
			float pivotVal = 1.0f;
			int kk = klb;
			for(; kk<kub; ++kk){
				if(m_indices[kk] == k && !Equal(m_val[kk], 0)){// the diagonal element
					pivotVal = m_val[kk];

					//printf("pivot: (%u, %u, %3.2f)\n", k, m_indices[kk], m_val[kk]);

					break;
				}
			}

			if(kk == kub){
				printf("Error, the element in the diagnoal is ZERO, the LU decomposition is terminated!\n");

				free(m_val);
				free(m_rowIndices);
				free(m_indices);
				free(permutationMatrix);

				assert(false);
				//exit(-1);
			}

			// a[i][k] /= a[k][k], a[i][k] <---> m_val[ii]
			bool aik_nz = false;
			for(; ii<iub; ++ii){
				if(m_indices[ii] == k && !Equal(m_val[ii], 0)){
					m_val[ii] /= pivotVal;

					//printf("a[i][k]: (%u, %u, %f)\n\n", i, m_indices[ii], m_val[ii]);

					aik_nz = true;
					break;
				}
				else if(m_indices[ii] > k && !Equal(m_val[ii], 0)){
					break;
				}
				else{
				}
			}

			// a[i][k] not ZERO
			if(aik_nz == true){
				int ipos = ii+1;
				int kpos = kk+1;
				while(ipos != iub && kpos != kub){
					// a[i][j] not zero and a[k][j]
					if(m_indices[ipos] == m_indices[kpos]/* && !Equal(m_val[ipos], 0) && !Equal(m_val[kpos], 0)*/){
						m_val[ipos] = m_val[ipos] - m_val[ii]*m_val[kpos];

						//printf("a[i][j]: (%u, %u, %f)\n", i, m_indices[ipos], m_val[ipos]);

						ipos++;
						kpos++;
					}
					else if(m_indices[ipos] > m_indices[kpos]){
						kpos++;
					}
					else if(m_indices[ipos] < m_indices[kpos]){
						ipos++;
					}
					else{
						assert(false);
					}

				}
			}// end of j loop(modified implemented)
		}// end of k loop
	}// end of i loop

#elif defined KIJ

	// KIJ version
	for(int k=0; k<m.numRows-1; ++k){

		float pivotVal = 1.0f;
		int klb = m_rowIndices[k];
		int kub = m_rowIndices[k+1];
		int kk = klb;
		for(; kk<kub; ++kk){
			if(m_indices[kk] == k && !Equal(m_val[kk], 0)){// the diagnoal element
				pivotVal = m_val[kk];
				break;
			}
		}
		assert(kk != kub);

		for(int i=k+1; i<m.numRows; ++i){

			int ilb = m_rowIndices[i];
			int iub = m_rowIndices[i+1];
			int ii = ilb;

			for(; ii<iub; ++ii){
				if(m_indices[ii] == k && !Equal(m_val[ii], 0)){// the element in ith row and kth column
					m_val[ii] /= pivotVal;
					break;
				}
				else if(m_indices[ii] > k){
					ii = iub;
					break;
				}
				else{
				}
			}

			if(ii != iub){// a[i][k] != 0
				int ipos = ii+1;
				int kpos = kk+1;
				while(ipos != iub && kpos != kub){
					if(m_indices[ipos] == m_indices[kpos]/* && !Equal(m_val[ipos], 0) && !Equal(m_val[kpos], 0)*/){
						m_val[ipos] = m_val[ipos] - m_val[kpos] * m_val[ii];
						ipos++;
						kpos++;
					}
					else if(m_indices[ipos] > m_indices[kpos]){
						ipos++;
					}
					else if(m_indices[ipos] < m_indices[kpos]){
						kpos++;
					}
					else{
						assert(false);
					}
				}
			}
		}
	}
#endif

}







