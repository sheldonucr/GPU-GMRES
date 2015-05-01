/***************************************************************************
 *   Copyright (C) 2006 by Jan Mayer                                       *
 *   jan.mayer@mathematik.uni-karlsruhe.de                                 *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


// this program uses the templates of ILU++ for full functionality.
// Some of these features are illustrated in this program.
// The object file does not need to be linked to (and should not be linked to)
// the iluplusplus library


// this includes the templates
#include "iluplusplus.h"


using namespace std;

typedef iluplusplus::Real Real;
typedef iluplusplus::Complex Complex;
typedef iluplusplus::Integer Integer;

typedef iluplusplus::matrix_sparse<Real> Matrix;
typedef iluplusplus::matrix_dense<Real> Matrix_Dense;
typedef iluplusplus::vector_dense<Real> Vector;

typedef iluplusplus::matrix_sparse<Complex> Complex_Matrix;
typedef iluplusplus::matrix_dense<Complex> Complex_Matrix_Dense;
typedef iluplusplus::vector_dense<Complex> Complex_Vector;

typedef iluplusplus::index_list Index_List;                 // used for accessing vectors of Real-type. Abused as permutation



int main(int argc, char *argv[]){

  try {

    Matrix Acol, Arow;

    iluplusplus::preprocessing_sequence L;
    iluplusplus::iluplusplus_precond_parameter param;


    // cout<<endl<<flush;
    // cout<<endl<<flush;
    // cout<<"**********************************************************************************"<<endl;
    // cout<<"PART I: Preprocessing."<<endl;
    // cout<<"**********************************************************************************"<<endl;
    // cout<<endl<<flush;
    // cout<<endl<<flush;
    // 
    // 
    // Acol.read_hb("./hb_matrices/test1.rua");
    // //Acol.read_hb("./hb_matrices/west2021.rua");
    // Arow.change_orientation_of_data(Acol);
    // 
    // Vector Dr0,Dc0;
    // Index_List pr0,pc0,ipr0,ipc0;
    // 
    // //L.set_PQ();
    // //L.set_METIS_NODE_ND_ORDERING();                                 //requires METIS
    // //L.set_PQ_METIS_NODE_ND_ORDERING();
    // //L.set_MAX_WEIGHTED_MATCHING_ORDERING();
    // //L.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER();
    // //L.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM();
    // //L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR();
    L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
    
    cout<<"Preprocessing selected:"<<endl;
    L.print();
    cout<<endl;
    
    // if(Arow.rows()<50){
    //     cout<<"The coefficient matrix:"<<endl;
    //     cout<<Arow.expand();        // expand() makes Arow into a Matrix_Dense, which is then written as usual
    // }
    // else cout<<"Matrix is too large to print."<<endl;
    
    param.init(L,11,"test");    // setup some other default values. Has no effect on preprocessing (except choice of PQ-Algorithm)
    
    // Arow.preprocess(param,pr0,pc0,ipr0,ipc0,Dr0,Dc0);
    // 
    // if(Arow.rows()<50){
    //     cout<<"The coefficient matrix after preprocessing:"<<endl;
    //     cout<<Arow.expand();        // expand() makes Arow into a Matrix_Dense, which is then written as usual
    // } else
    //     cout<<"The coefficient matrix is too large to print."<<endl;
    // cout<<"Row Permutation used:"<<endl<<pr0;
    // cout<<"Column Permutation used:"<<endl<<pc0;
    // cout<<"Row Scaling used:"<<endl<<Dr0;
    // cout<<"Column Scaling used:"<<endl<<Dc0;
    // 
    // 
    // cout<<endl<<flush;
    // cout<<endl<<flush;
    // cout<<"**********************************************************************************"<<endl;
    // cout<<"PART II: Preprocessing, Preconditioning, Solving."<<endl;
    // cout<<"**********************************************************************************"<<endl;
    // cout<<endl<<flush;
    // cout<<endl<<flush;


    cout<<"*****************************************************************************"<<endl;
    cout<<"Examples for real matrices."<<endl;
    cout<<"*****************************************************************************"<<endl;

    //Acol.read_hb("./hb_matrices/test1.rua");
    Acol.read_hb("./hb_matrices/west2021.rua");
    Arow.change_orientation_of_data(Acol);

    iluplusplus::multilevelILUCDPPreconditioner<Real,Matrix,Vector> Pr;

    param.set_threshold(2.0);
    param.set_MEM_FACTOR(1.0);
    param.set_MAX_LEVELS(1);

    Pr.make_preprocessed_multilevelILUCDP(Acol,param);

    cout<<"Information on the preconditioner:"<<endl;
    Pr.print_info();

    if(Arow.rows()<50){
        cout<<endl;
        cout<<"Matrices, permutations and scalings used:"<<endl<<endl;
        cout<<endl;
        for(int k=0;k<Pr.levels();k++){
            cout<<"   *********************************** "<<endl;
            cout<<"       Information on Level "<<k<<endl;
            cout<<"   *********************************** "<<endl;
            cout<<"            L:"<<endl<<Pr.extract_left_matrix(k).expand();
            cout<<"            U:"<<endl<<Pr.extract_right_matrix(k).expand();
            cout<<"            D^{-1}:"<<endl<<Pr.extract_middle_matrix(k);
            cout<<"            Row Permutation:"<<endl<<Pr.extract_permutation_rows(k);
            cout<<"            Column Permutation:"<<endl<<Pr.extract_permutation_columns(k);
            cout<<"            Inverse of Row Permutation:"<<endl<<Pr.extract_inverse_permutation_rows(k);
            cout<<"            Inverse of Column Permutation:"<<endl<<Pr.extract_inverse_permutation_columns(k);
            cout<<"            Scaling of Rows:"<<endl<<Pr.extract_left_scaling(k);
            cout<<"            Scaling of Columns:"<<endl<<Pr.extract_right_scaling(k)<<endl;
        }
    } else
        cout<<"Matrix is too large to print preconditioner."<<endl;

    cout<<"fill-in: "<<((Real) Pr.total_nnz())/(Real)Acol.non_zeroes()<<endl;

    cout<<endl<<flush;

    Vector b,x_exact,xbicgstab,xcgs,xgmres;

    x_exact.resize(Acol.rows(),1.0);
    xgmres.resize(Acol.rows(),0.0);

    //Acol.matrix_vector_multiplication(iluplusplus::ID,x_exact,b);
    b.resize(Acol.rows(),1.0);

    Integer min_iter =0;
    Integer reach_max_iter = 100;
    Real reach_rel_tol = 4.0;
    Real reach_abs_tol = 4.0;

    Real rel_tol =  reach_rel_tol;
    Real abs_tol = reach_abs_tol;
    Integer max_iter = reach_max_iter;

// std::cout<<"Arow"<<std::endl<<Arow.expand()<<std::endl;
// std::cout<<"Acol"<<std::endl<<Acol.expand()<<std::endl;
// std::cout<<"b"<<std::endl<<b<<std::endl;

    cout<<"*****************************************************************************"<<endl;
    cout<<"BiCGstab"<<endl;
    cout<<"*****************************************************************************"<<endl;
    iluplusplus::bicgstab<Real,Matrix,Vector>(Pr,iluplusplus::SPLIT,Acol,b,xbicgstab,min_iter,max_iter,rel_tol,abs_tol,true);
    cout<<"Iterations "<<max_iter<<endl;
    cout<<"error: "<<(xbicgstab-x_exact).norm_max()<<endl<<flush;
    cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
    cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;

// std::cout<<"x"<<std::endl<<xbicgstab<<std::endl;

    rel_tol =  reach_rel_tol;
    abs_tol = reach_abs_tol;
    max_iter = reach_max_iter;

    cout<<"*****************************************************************************"<<endl;
    cout<<"CGS"<<endl;
    cout<<"*****************************************************************************"<<endl;
    iluplusplus::cgs<Real,Matrix,Vector>(Pr,iluplusplus::SPLIT,Acol,b,xcgs,min_iter,max_iter,rel_tol,abs_tol,true);
    cout<<"Iterations "<<max_iter<<endl;
    cout<<"error: "<<(xcgs-x_exact).norm_max()<<endl<<flush;
    cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
    cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;


    rel_tol =  reach_rel_tol;
    abs_tol = reach_abs_tol;
    max_iter = reach_max_iter;
    Integer restart = 100;

    cout<<"*****************************************************************************"<<endl;
    cout<<"GMRES"<<endl;
    cout<<"*****************************************************************************"<<endl;

    iluplusplus::gmres<Real,Matrix,Vector>(Pr,iluplusplus::SPLIT,Acol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,false);
    cout<<"Iterations "<<max_iter<<endl;
    cout<<"error: "<<(xgmres-x_exact).norm_max()<<endl<<flush;
    cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
    cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;
    cout<<endl;

#ifdef ILUPLUSPLUS_USES_PARDISO
    cout<<"*****************************************************************************"<<endl;
    cout<<"PARDISO direct solver"<<endl;
    cout<<"*****************************************************************************"<<endl;
    Vector xpardiso;
    int peak_memory,perm_memory,nnzLU; 
    double solve_time;
    Arow.solve_pardiso(b,xpardiso,peak_memory,perm_memory,nnzLU,solve_time);
    cout<<"error: "<<(xpardiso-x_exact).norm_max()<<endl<<flush;
    cout<<"time: "<<solve_time<<endl;
    cout<<"peak_memory: "<<peak_memory<<endl;
    cout<<"perm_memory: "<<perm_memory<<endl;
    cout<<"nnz of factors LU: "<<nnzLU<<endl;
#endif

   return 0;

  }
  catch(iluplusplus::iluplusplus_error ippe){
      std::cerr<<"main_templates: Error allocating memory. "<<ippe.error_message()<<"Returning."<<std::endl;
      return 1;
  }
}
