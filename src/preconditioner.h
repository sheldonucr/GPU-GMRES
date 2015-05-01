/*! \file
	\brief definition of the class for the preconditioners. Including the base class and the inherited classes
*/

#ifndef PRECONDITIONER_H_
#define PRECONDITIONER_H_

#include <assert.h>
#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cublas.h>
#include <cusparse_v2.h>

#include <sys/time.h>

#include "SpMV.h"
#include "config.h"
#include "SpMV_kernel.h"
#include "defs.h"

#include "leftILU.h"
#include "gpuData.h"

using namespace std;

//! The virtual class for different kinds of preconditioner
/*!
  \brief All the interface need to be implemented by the inheriate class
*/
class Preconditioner{
	public:
	//! index type
	typedef int IndexType;
	//! value type
	typedef float ValueType;
	//! the storage for the memory
	typedef cusp::device_memory MemorySpace;

	//! dimension of the matrix. Only handle the matrix with same number of rows and columns
	int numRows;


        //---- For GMRES ---
        float *d_r, *d_rr, *d_bb, *d_y;
        float *s, *cs, *sn, *H;
        float *d_v, *d_w, *d_ww;

	//! The function used to add preconditoning operation on a host array
	/*!
		\param i_data the data to be preconditioned
		\param o_data the data after preconditioned
	*/
	virtual void HostPrecond(const ValueType *i_data, ValueType *o_data) = 0;

	//! The funciton used to add preconditioning operation on a device array
	/*!
		\param i_data the data to be preconditioned
		\param o_data the data after preconditioned
	*/
	virtual void DevPrecond(const ValueType *i_data, ValueType *o_data) = 0;

	//! The function used to initialize the preconditioner
	/*!
		\param mySpM The information of the \p A matrix in CSR format. The object contains both the Host data and Device data
	*/
	virtual void Initilize(const MySpMatrix &mySpM) = 0;
	//virtual void InitilizeILUPP(const ucr_cs_dl &Mucr) = 0;
        
        virtual void HostPrecond_rhs(const ValueType *i_data, ValueType *o_data)=0;
        virtual void HostPrecond_right(const ValueType *i_data, ValueType *o_data)=0;
        virtual void HostPrecond_left(const ValueType *i_data, ValueType *o_data)=0;
        virtual void HostPrecond_starting_value(const ValueType *i_data, ValueType *o_data)=0;
        
        virtual void DevPrecond_rhs(float *i_data, float *o_data)=0;
        virtual void DevPrecond_right(float *i_data, float *o_data)=0;
        virtual void DevPrecond_left(float *i_data, float *o_data)=0;
        virtual void DevPrecond_starting_value(float *i_data, float *o_data)=0;
        
        virtual ~Preconditioner() {};
};

//! class for AINV preconditioner
class MyAINV: public Preconditioner{
	private:
		//! The value array for \p w_t matrix for AINV
		ValueType *w_t_val;
		//! The value array for \p z matrix for AINV
		ValueType *z_val;
		//! The value array for \p diag elements for AINV
		ValueType *diag_val;

		IndexType *w_t_rowIndices, *w_t_indices, *z_rowIndices, *z_indices;

		//! Intermedia data for AINV preconditioning
		cusp::array1d<float, cusp::device_memory> *in_array;
		//! Intermedia data for AINV preconditioning
		cusp::array1d<float, cusp::device_memory> *out_array;
		//! raw pointer pointed to in_array
		float *pin_array;
		//! raw pointer pointed to out_array
		float *pout_array;
		//! pointer pointed to a nonsym_brison_ainv preconditioner
		cusp::precond::nonsym_bridson_ainv<ValueType, MemorySpace> *ainv_M;

	public:
		//! The AINV precondition for host array
		void HostPrecond(const ValueType *i_data, ValueType *o_data);
		//! The AINV precondition for device array
		void DevPrecond(const ValueType *i_data, ValueType *o_data);
		//! The initialize operations for AINV preconditioner
		void Initilize(const MySpMatrix &mySpM);
};

//! The class of ILU0 preconditioer
class MyILU0 : public Preconditioner{
	public:
		ValueType *l_val;
		IndexType *l_rowIndices, *l_indices;
		ValueType *u_val;
		IndexType *u_rowIndices, *u_indices;

		ValueType *d_l_val;
		IndexType *d_l_rowIndices, *d_l_indices;
		ValueType *d_u_val;
		IndexType *d_u_rowIndices, *d_u_indices;

		cusparseStatus_t *status;
		cusparseHandle_t *handle;
		cusparseSolveAnalysisInfo_t *L_info, *U_info;
		cusparseMatDescr_t *L_des, *U_des, *A_des;


	public:
		//! The ILU0 value on Host side
		void HostPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILU0 value on Device side
		void DevPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILU0 initilize operations
		void Initilize(const MySpMatrix &mySpM);
};

//! The class of ILUK preconditioer
class MyILUK : public Preconditioner{
	private:
		ValueType *l_val;
		IndexType *l_rowIndices, *l_indices;
		ValueType *u_val;
		IndexType *u_rowIndices, *u_indices;

		ValueType *d_l_val;
		IndexType *d_l_rowIndices, *d_l_indices;
		ValueType *d_u_val;
		IndexType *d_u_rowIndices, *d_u_indices;

		cusparseStatus_t *status;
		cusparseHandle_t *handle;
		cusparseSolveAnalysisInfo_t *L_info, *U_info;
		cusparseMatDescr_t *L_des, *U_des, *A_des;


	public:
                float *Lval_ITSOL;
                int *LrowIndices_ITSOL, *Lindices_ITSOL;
                float *Uval_ITSOL;
                int *UrowIndices_ITSOL, *Uindices_ITSOL;
                float *Dval_ITSOL;

		//! The ILUK value on Host side
		void HostPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILUK value on Device side
		void DevPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILUK initilize operations
		void Initilize(const MySpMatrix &mySpM);
};

//! The class of ILUK preconditioer
class MyILUPP : public Preconditioner{
  //private:
 public:
  //
  ValueType *l_val; // float
  IndexType *l_rowIndices, *l_indices;
  ValueType *u_val; // float
  IndexType *u_rowIndices, *u_indices;
  
  double *l_val_double, *u_val_double;
  
  ValueType *d_l_val; // float
  IndexType *d_l_rowIndices, *d_l_indices;
  double *d_l_val_double;
  
  ValueType *d_u_val; // float
  IndexType *d_u_rowIndices, *d_u_indices;
  double *d_u_val_double;
  //
  ValueType *middle_val;
  IndexType *middle_rowIndices, *middle_indices;
  //
  ValueType *p_val;
  IndexType *p_rowIndices, *permRow_indices, *permCol_indices;

  
  double *lscale_val, *rscale_val;
  
  cusparseStatus_t *status;
  cusparseHandle_t *handle;
  cusparseSolveAnalysisInfo_t *L_info, *U_info;
  cusparseMatDescr_t *L_des, *U_des, *A_des;
  
  float *tmpvector;
	public:
                float *Lval_ITSOL;
                int *LrowIndices_ITSOL, *Lindices_ITSOL;
                float *Uval_ITSOL;
                int *UrowIndices_ITSOL, *Uindices_ITSOL;
                float *Dval_ITSOL;

		//! The ILUK value on Host side
		void HostPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILUK value on Device side
		void DevPrecond(const ValueType *i_data, ValueType *o_data);
		//! The ILUK initilize operations
		void Initilize(const MySpMatrix &mySpM);
		void Initilize(const MySpMatrixDouble &PrLeft, const MySpMatrixDouble &PrRight,
                               const MySpMatrix &PrMiddle_mySpM,
                               const MySpMatrix &PrPermRow, const MySpMatrix &PrPermCol,
                               const MySpMatrixDouble &PrLscale, const MySpMatrixDouble &PrRscale);

                void HostPrecond_rhs(const ValueType *i_data, ValueType *o_data);
                void HostPrecond_right(const ValueType *i_data, ValueType *o_data);
                void HostPrecond_left(const ValueType *i_data, ValueType *o_data);
                void HostPrecond_starting_value(const ValueType *i_data, ValueType *o_data);

                void DevPrecond_rhs(float *i_data, float *o_data);
                void DevPrecond_right(float *i_data, float *o_data);
                void DevPrecond_left(float *i_data, float *o_data);
                void DevPrecond_starting_value(float *i_data, float *o_data);
        
                ~MyILUPP();
};

class MyILUPPfloat : public Preconditioner{
  //private:
 public:
  //ValueType *l_val;
  double *l_val;
  IndexType *l_rowIndices, *l_indices;
  //ValueType *u_val;
  double *u_val;
  IndexType *u_rowIndices, *u_indices;
  
  ValueType *middle_val;
  IndexType *middle_rowIndices, *middle_indices;
  float *d_middle_val;

  ValueType *p_val;
  IndexType *p_rowIndices, *permRow_indices, *permCol_indices;
  int *d_p_rowIndices, *d_permRow_indices, *d_permCol_indices;
  double *d_p_val;
  //float *d_p_val_single;
  
  float *lscale_val, *rscale_val;
  float *d_lscale_val, *d_rscale_val;

  ValueType *d_l_val;
  double *d_l_val_double;
  IndexType *d_l_rowIndices, *d_l_indices;

  ValueType *d_u_val;
  double *d_u_val_double;
  IndexType *d_u_rowIndices, *d_u_indices;

  cusparseStatus_t *status;
  cusparseHandle_t *handle;
  cusparseSolveAnalysisInfo_t *L_info, *U_info;
  cusparseMatDescr_t *L_des, *U_des, *A_des;

  float *tmpvector;
  double *d_tmpvector_double, *d_tmp_solution_double;
  float *d_tmpvector_single;

  int l_nnz, u_nnz;

 public:
  // float *Lval_ITSOL;
  // int *LrowIndices_ITSOL, *Lindices_ITSOL;
  // float *Uval_ITSOL;
  // int *UrowIndices_ITSOL, *Uindices_ITSOL;
  // float *Dval_ITSOL;
  
  //! The ILUK value on Host side
  void HostPrecond(const ValueType *i_data, ValueType *o_data);
  //! The ILUK value on Device side
  void DevPrecond(const ValueType *i_data, ValueType *o_data);
  //! The ILUK initilize operations
  void Initilize(const MySpMatrix &mySpM);
  void Initilize(const MySpMatrixDouble &PrLeft, const MySpMatrixDouble &PrRight,
                 const MySpMatrix &PrMiddle_mySpM,
                 const MySpMatrix &PrPermRow, const MySpMatrix &PrPermCol,
                 const MySpMatrix &PrLscale, const MySpMatrix &PrRscale,
                 int restart);
  
  void HostPrecond_rhs(const ValueType *i_data, ValueType *o_data);
  void HostPrecond_right(const ValueType *i_data, ValueType *o_data);
  void HostPrecond_left(const ValueType *i_data, ValueType *o_data);
  void HostPrecond_starting_value(const ValueType *i_data, ValueType *o_data);

  void DevPrecond_rhs(float *i_data, float *o_data);
  void DevPrecond_right(float *i_data, float *o_data);
  void DevPrecond_left(float *i_data, float *o_data);
  void DevPrecond_starting_value(float *i_data, float *o_data);
  
  ~MyILUPPfloat();
};

//! class for diagonal preconditioer
class MyDIAG : public Preconditioner{
	private:
		//! The diagonal value on Host side
		ValueType *val;
		//! The diagonal value on Device side
		ValueType *d_val;

	public:
		//! diagonal preconditioning on host
		void HostPrecond(const ValueType *i_data, ValueType *o_data);
		//! diagonal preconditioning on device
		void DevPrecond(const ValueType *i_data, ValueType *o_data);
		//! initialize diagonal preconditioner
		void Initilize(const MySpMatrix &mySpM);
};

//! class for the method with no preconditioer
/*!
  \brief the class is used to fit the situation when no preconditioer is need
 */
class MyNONE : public Preconditioner{
	public:
		//! Just copy \p i_data to \p o_data
		void HostPrecond(const ValueType *i_data, ValueType *o_data){
			memcpy(o_data, i_data, this->numRows * sizeof(ValueType));
		}
		//! Just copy \p i_data to \p o_data
		void DevPrecond(const ValueType *i_data, ValueType *o_data){
			cudaMemcpy(o_data, i_data, this->numRows * sizeof(float), cudaMemcpyDeviceToDevice);
		}
		//! initialize the dimension for the preconditioner
		void Initilize(const MySpMatrix &mySpM){
			this->numRows = mySpM.numRows;
		}
};

#endif
