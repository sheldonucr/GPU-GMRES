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


#include<vector>

// this program uses only the ILU++ interface to allow easy use in other programs.
// it requires the iluplusplus library including the most important features of ILU++
// Both the library and this program need to be compiled for real matrices.

// include the classes and functions defined in the library iluplusplus(version).a
#include "iluplusplus_interface.h"

typedef iluplusplus::Coeff_Field Coeff_Field;

int main(int argc, char *argv[]){
  try {
  // *************************************************************************** //
  // PART I: solving linear systems stored in Harwell-Boeing format.             //
  // *************************************************************************** //


    // define an iluplusplus_precond_parameter to store all parameters needed for preprocessing and preconditioning.
    iluplusplus::iluplusplus_precond_parameter param;

    // Standard configurations of both preprocessing and preconditioning. (Somewhat older, better to use newer routines)
    // normalize rows, columns, use PQ. Pivoting by rows and columns.
    // param.default_configuration(0);
    // normalize rows, columns, use PQ. No pivoting. (Because no pivoting in performed at all, this fails for more matrices).
    // param.default_configuration(1);

    // better preprocessing to produce an I-matrix. Best overall routines:
    // I-matrix preprocessing; pivot by rows and columns
    param.default_configuration(10);
    // I-matrix preprocessing; Additional preprocessing to make a diagonally dominant submatrix. No pivoting
    // param.default_configuration(11);

    param.set_MEM_FACTOR(5.0);         // by default ILU++ allocates memory very generously (default here is 20.0). Often, less sufficient.

              // alternative for setting up parameters: select preprocessing method and preconditioner separately:
              //  1) declare parameter for preprocessing type:
              //  iluplusplus::preprocessing_sequence L;
              //  2) set preprocesing type
              //  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
              //  3) setup preconditioner parameters: preprocessing type + preconditioner number (here 10)
              //  param.init(L,10,"my comment");



    // declare (sparse) matrices and (dense) vectors
    iluplusplus::matrix Acol;
    iluplusplus::vector X,X_exact,B;


    std::cout<<std::endl<<std::flush;
    std::cout<<std::endl<<std::flush;
    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<"PART I: solving a linear system for a matrix and rhs in Harwell-Boeing format."<<std::endl;
    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<std::endl<<std::flush;
    std::cout<<std::endl<<std::flush;
#ifndef ILUPLUSPLUS_USES_COMPLEX
    // read a matrix out of a Harwell-Boeing file
    Acol.read_hb("./hb_matrices/sherman1.rua");
   // read first RHS (not always available), but available for this matrix
    B.read_hb("./hb_matrices/sherman1.rua",1);
#endif

#ifdef ILUPLUSPLUS_USES_COMPLEX
    // read a matrix out of a Harwell-Boeing file
    Acol.read_hb("./hb_matrices/young1c.rb");
   // read first RHS (not always available), actually, not available for this matrix
    B.read_hb("./hb_matrices/young1c.rb",1);
#endif


    // if a RHS was successfully read, the vector b will have dimension larger than 0
    bool have_rhs = (B.dim()>0);

    // make artificial RHS, if we do not have one.
    if (have_rhs){
        std::cout<<"  RHS read successfully."<<std::endl;
    } else {
        std::cout<<"  RHS not read successfully. Making artificial RHS."<<std::endl;
        X_exact.resize(Acol.rows(),1.0);
        // X_exact.set_all(1.0);
        // works too but requires copying:
        // B = Acol*X_exact;
        // better, especially for large problems:
        Acol.multiply(X_exact,B);
    }

    // set dropping parameter. set_threshold(tau) sets dropping threshold to 10^{-tau}
    param.set_threshold(2.0);

    // set first tolerance for terminating iterative method (also used as return variable for reduction of residual obtained, i.e. the relative residual)
    iluplusplus::Real rel_tol   = 8.0;  // actual tolerance is 10^{-rel_tol}
    // set second tolerance for terminating iterative method (also used as return variable for residual obtained, i.e. the absolute residual)
    iluplusplus::Real abs_tol   = 8.0;  // actual tolerance is 10^{-abs_tol}
    // variable for returning the absolute error (if exact solution is known -- otherwise nan is returned)
    iluplusplus::Real abs_error;
    // maximal number of iterations. The variable returns the number of iterations required for convergence
    iluplusplus::Integer max_iter = 600;

    // seach directory for matrix files
    std::string matrix_dir  = "./hb_matrices/";
    // directory for output files
    std::string output_dir = "./output/";


    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***   PART Ia: solving in one step using iluplusplus::solve_with_multilevel_preconditioner."<<std::endl;
    std::cout<<std::endl<<std::flush;

    // set up preconditioner,solve linear system, and write information on solve in file -- all in one step.
    iluplusplus::solve_with_multilevel_preconditioner(Acol,B,X_exact,X,!have_rhs,rel_tol,abs_tol,max_iter,abs_error,output_dir,"sherman1",param);

    std::cout<<"        Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<") and reduction of the initial residual by 10^(-"<<rel_tol<<")"<<std::endl;
    //std::cout<<"The calculated solution is "<<std::endl<<x;




    // alternatively: do each step above by hand, i.e. setup preconditioner and solve:
    // we now need to determine the minimum number of iterations we want (in solve_with_multilevel_preconditioner, min_iter = 1 is used)

    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***  PART Ib: solving in several steps: setting up preconditioner and solving subsequently."<<std::endl;
    std::cout<<std::endl<<std::flush;


    iluplusplus::Integer min_iter = 1;
    // thes variables need to be reset, as they contain the results of the previous call of iluplusplus::solve_with_multilevel_preconditioner
    abs_tol = 8.0; rel_tol = 8.0; max_iter = 600;
    // declare preconditioner
    iluplusplus::multilevel_preconditioner Pr;
    // setup preconditioner
    Pr.setup(Acol,param);


    std::cout<<"                Preconditioner has a fill-in of "<<(iluplusplus::Real) Pr.total_nnz() / (iluplusplus::Real) Acol.non_zeroes()<<std::endl;
    std::cout<<"                Preconditioner requires approximately "<< Pr.memory()<<" bytes to store."<<std::endl;
    std::cout<<"                Preconditioner calculation required approximately "<< Pr.memory_used_calculations()<<" bytes."<<std::endl;
    std::cout<<"                Approximately "<< Pr.memory_allocated_calculations()<<" bytes were allocated."<<std::endl<<std::endl;


    // call solver
    // Note: BiCGstab returns rel_tol abs_tol achieved
    BiCGstab(Pr,Acol,B,X,min_iter,max_iter,rel_tol,abs_tol);

    std::cout<<"        Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<") and reduction of the initial residual by 10^(-"<<rel_tol<<")"<<std::endl;
    //std::cout<<"The calculated solution is "<<std::endl<<x;

  // *************************************************************************** //
  // PART II: solving linear systems for matrices provided with C-style arrays
  //          or C++ vectors:                                                    //
  // *************************************************************************** //

    std::cout<<std::endl<<std::flush;
    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<"PART II: solving a linear system for a matrix using C-style arrays or C++ vectors."<<std::endl;
    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<std::endl<<std::flush;

  // iluplusplus provides powerful matrix and vector classes. If you need to use
  // C-style arrays, interfaces are provided, i.e.
  // if you have your own matrices, defined using pointers, the following approach is suitable.
  // WARNING:  iluplusplus performs no checking to make sure that:
  //    1) the format is correct and consistent
  //    2) data types are compatible
  //    3) size checks
  // You are responsible to make sure that the input data conforms to iluplusplus data types
  // Segmentation faults are likely to be due to errors in data conversion
  // NOTE: all pointers need to be initialized - if no memory is allocated, then a pointer needs to point to 0. (Null pointer).


  // *************************************************************************** //
  // PART IIa: First possibility: you provide the matrix and rhs,                //
  //           in C-style arrays and                                             //
  //           iluplusplus does the rest and returns an array with the solution  //
  // *************************************************************************** //

    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***  PART IIa: solving a linear system for a matrix and rhs (C-style arrays) given by user."<<std::endl;
    std::cout<<"                  ILU++ calculates and returns a solution as a C-style array"<<std::endl;
    std::cout<<"                  and writes information on results (not solution) to a file."<<std::endl;
    std::cout<<std::endl<<std::flush;


    int *ia = 0;             // make sure that this type is of iluplusplus::Integer
    int *ja = 0;             // make sure that this type is of iluplusplus::Integer
    Coeff_Field *a = 0;      // make sure that this type corresponds to the Coeff_Field of iluplusplus matrices and vectors 

   // setup matrix in sparse compressed row format

     ia  = new int[9];
          ia[0] = 0; ia[1] = 4; ia[2] = 7; ia[3] = 9; ia[4] = 11; ia[5] = 12; ia[6] = 15; ia[7] = 17; ia[8] = 20;

     ja = new int[20];
          ja[0]  = 0; ja[1]  = 2; ja[2]  = 5; ja[3]  = 6;
          ja[4]  = 1; ja[5]  = 2; ja[6]  = 4;
          ja[7]  = 2; ja[8]  = 7;
          ja[9]  = 3; ja[10] = 6;
          ja[11] = 1;
          ja[12] = 2; ja[13] = 5; ja[14] = 7;
          ja[15] = 1; ja[16] = 6;
          ja[17] = 2; ja[18] = 6; ja[19] = 7;

     a = new iluplusplus::Coeff_Field[20];
          a[0]  = (Coeff_Field) 7.0;  a[1]  =  (Coeff_Field) 1.0; a[2]  =  (Coeff_Field) 2.0; a[3]  =  (Coeff_Field) 7.0;
          a[4]  = (Coeff_Field) -4.0; a[5]  =  (Coeff_Field) 8.0; a[6]  =  (Coeff_Field) 2.0;
          a[7]  = (Coeff_Field) 1.0;  a[8]  =  (Coeff_Field) 5.0;
          a[9]  = (Coeff_Field) 7.0;  a[10] =  (Coeff_Field) 9.0;
          a[11] = (Coeff_Field) -4.0;
          a[12] = (Coeff_Field)  7.0; a[13] =  (Coeff_Field) 3.0; a[14] =  (Coeff_Field) 8.0;
          a[15] = (Coeff_Field)  1.0; a[16] =  (Coeff_Field) 11.0;
          a[17] = (Coeff_Field) -3.0; a[18] =  (Coeff_Field) 2.0; a[19] =  (Coeff_Field) 5.0;

    // Setup right hand side

    iluplusplus::Coeff_Field *b = 0;
    b = new iluplusplus::Coeff_Field[8];
           b[0]  =  (Coeff_Field) 17.0; b[1]  =   (Coeff_Field) 6.0;  b[2]  =   (Coeff_Field) 6.0;  b[3]  =  (Coeff_Field) 16.0;
           b[4]  =  (Coeff_Field) -4.0; b[5]  =   (Coeff_Field) 18.0; b[6]  =   (Coeff_Field) 12.0; b[7]  =  (Coeff_Field) 4.0;
    // int n_b = 8; not needed, we assume implictly that the dimension the same as for the coefficient matrix!

    iluplusplus::Coeff_Field*     x_exact = 0;    // do not allocate memory, unless you know the solution and just want to check what iluplusplus calculates
                                                  // NOTE: but DO set the pointer to 0, if no memory is allocated. Otherwise, expect a segmentation fault!
    int                         n_x_exact = 0;    // indicates size of x_exact
    iluplusplus::Coeff_Field*           x = 0;    // allocate memory, if wanted. Otherwise it will be allocated automatically.
                                                  // NOTE: but DO set the pointer to 0, if no memory is allocated. Otherwise, expect a segmentation fault!
    int                               n_x = 0;    // indicates size of x (should correspond to the memory allocated)

    // here we set the dropping parameter to 10^{-1.0} = 0.1
    param.set_threshold(1.0);
    // reset other parameters (clear values from previous call):
    abs_tol = 8.0; rel_tol = 8.0; max_iter = 600;
    // call solver; parameters are:
    //          dimension n of matrix,
    //          nnz matrix,
    //          matrix orientation,
    //          matrix data (a),
    //          matrix index pointer (ia),
    //          matrix pointer (ja),
    //          rhs,
    //          n_x_exact (dimension n of x_exact),
    //          x_exact (input: exact solution, if known; otherwise vector of dimension 0 is exact solution is not known),
    //          n_x (dimension  n of x),
    //          x (input: meaningless, output: solution),
    //          exact solution known (boolean),
    //          rel_tol   (input: stopping criterion (rel. reduction of residual required) output: relative reduction obtained),
    //          abs_tol   (input: stopping criterion (norm of residual required. output: residual obtained),
    //          max_iter  (input: max. iterations allowed. output: number of iterations needed)
    //          abs_error (output: error of solution, if exact solution known), 
    //          working directory, 
    //          matrix name,
    //          parameters for preprocessing and preconditioning
    // Note:  solve_with_multilevel_preconditioner returns -log10(rel_tol), -log10(abs_tol) achieved. As no exact solution is known, abs_error = nan
    solve_with_multilevel_preconditioner(8,20,iluplusplus::ROW,a,ja,ia, b,n_x_exact,x_exact,n_x,x,false,rel_tol,abs_tol,max_iter,abs_error,output_dir,"test1",param);
    // look at result: should be (1,...1)^T:
    std::cout<<"Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<") and reduction of the initial residual by 10^(-"<<rel_tol<<")"<<std::endl;
    std::cout<<"Solution using solve_with_multilevel_preconditioner with C-style arrays -- should be (1,...1)^T:"<<std::endl<<std::flush;
    for(int i=0;i<n_x;i++)std::cout<<"x["<<i<<"] = "<<x[i]<<std::endl<<std::flush;    std::cout<<std::endl<<std::flush;

    // suppose you know the solution, then you can check the accuracy of iluplusplus as follows:

    // make C-style array for x_exact. Always also declare a variable indicating the dimension!
    x_exact = new iluplusplus::Coeff_Field[8];
           x_exact[0]  =  (Coeff_Field) 1.0; x_exact[1]  =  (Coeff_Field) 1.0; x_exact[2]  =  (Coeff_Field) 1.0; x_exact[3]  =  (Coeff_Field) 1.0;
           x_exact[4]  =  (Coeff_Field) 1.0; x_exact[5]  =  (Coeff_Field) 1.0; x_exact[6]  =  (Coeff_Field) 1.0; x_exact[7]  =  (Coeff_Field) 1.0;
    n_x_exact = 8;

    // reset parameters:
    abs_tol = 8.0; rel_tol = 8.0; max_iter = 600;
    // Note:  solve_with_multilevel_preconditioner returns -log10(rel_tol), -log10(abs_tol), -log10(abs_error) achieved
    solve_with_multilevel_preconditioner(8,20,iluplusplus::ROW,a,ja,ia,b,n_x_exact,x_exact,n_x,x,true,rel_tol,abs_tol,max_iter,abs_error,output_dir,"test2",param);
    // look at result: should be (1,...1)^T:
    std::cout<<"Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<"), a reduction of the initial residual by 10^(-"<<rel_tol<<") and an absolute error of 10^(-"<<abs_error<<")"<<std::endl;
    std::cout<<"Solution using solve_with_multilevel_preconditioner with C-style arrays -- should be (1,...1)^T:"<<std::endl<<std::flush;
    for(int i=0;i<n_x;i++)std::cout<<"x["<<i<<"] = "<<x[i]<<std::endl<<std::flush;    std::cout<<std::endl<<std::flush;

  // ***************************************************************************************** //
  // PART IIb: Second possibility:                                                             //
  //       you provide the matrix, the rhs (in C-style arrays),                                //
  //       the matrix-vector-multiplication and the iterative solver.                          //
  //       ILU++ calculates the preconditioner and provides a routine for applying             //
  //       the preconditioner to a C-style array.                                              //
  // ***************************************************************************************** //

    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***  PART IIb: User provides a matrix as C-style array."<<std::endl;
    std::cout<<"                  ILU++ calculates preconditioner and applies it to a C-style array."<<std::endl;
    std::cout<<"                  User provides rhs, matrix-vector-multiplication and his own solver."<<std::endl;
    std::cout<<std::endl<<std::flush;

    // here we set the dropping parameter to 10^{-1000.0} = 0, i.e. no dropping is performed, hence the preconditioner is exact,
    //  i.e. the inverse of the coefficient matrix.
    param.set_threshold(1000.0);

    // setup preconditioner
    // a, ja, ia correspond to the compressed sparse row format,
    // next we have the dimension, the number of non-zero elements, the orientation of the matrix,
    // and the parameter for the preconditioner.
    Pr.setup(a,ja,ia,8,20,iluplusplus::ROW,param);

    // if desired, print information on preconditioner.
    // Pr.print_info();

    std::cout<<"                Preconditioner has a fill-in of "<<(iluplusplus::Real) Pr.total_nnz() / (iluplusplus::Real) 20<<std::endl;
    std::cout<<"                Preconditioner requires approximately "<< Pr.memory()<<" bytes to store."<<std::endl;
    std::cout<<"                Preconditioner calculation required approximately "<< Pr.memory_used_calculations()<<" bytes."<<std::endl;
    std::cout<<"                Approximately "<< Pr.memory_allocated_calculations()<<" bytes were allocated."<<std::endl<<std::endl;

    // define a vector to which we apply the preconditioner.
    // Generally, it is best to use iluplusplus data types, but an interface is provided to apply a preconditioner to a C-style array.
    // Again, you are responsible to make sure the data makes sense.

    iluplusplus::Coeff_Field *v;

    v = new iluplusplus::Coeff_Field[8];
           v[0]  =  (Coeff_Field) 1.0; v[1]  =  (Coeff_Field) 2.0;  v[2]  =  (Coeff_Field) 3.0; v[3]  =  (Coeff_Field) 4.0;
           v[4]  =  (Coeff_Field) 5.0; v[5]  =  (Coeff_Field) 6.0;  v[6]  =  (Coeff_Field) 7.0; v[7]  =  (Coeff_Field) 8.0;
    int  n_v = 8;


    // NOTE: if you want to use your own iterative solver and have implemented your own matrix-vector-multiplication,
    // then you only need a routine to apply the preconditioner to an arbitrary vector (in a C-style array). This is provided as follows:

    // apply preconditioner to an array v of length 8
    Pr.apply_preconditioner(v,8);
    // now, v is an array containing A^{-1}v (because we had calculated an "exact" preconditioner, i.e. the inverse of A)
    std::cout<<"        preconditioned vector using C-style arrays"<<std::endl<<std::flush;
    for (int i=0; i<n_v; i++) std::cout<<v[i]<<std::endl<<std::flush;

    // check results. Construct an iluplusplus matrix using the data
    std::cout<<"        Checking results"<<std::endl<<std::flush;

    iluplusplus::matrix A;
    int nrow = 8;
    int ncol = 8;
    int nnz = 20;
    iluplusplus::orientation_type orientation = iluplusplus::ROW;
    // the call A.interchange(a,ja,ia,nrow,ncol,nnz,orientation);
    // interchanges arrays defined above to make an iluplusplus matrix. As the original matrix was a 0x0 matrix,
    // a,ja,ia,nrow,ncol,nnz,orientation will describe a 0x0 matrix after switching.
    // NOTE: interchange requires arrays that are set-up dynamically. Make sure to free these arrays in the end.
    // iluplusplus takes care of freeing any iluplusplus matrices or vectors.
    // NOTE: no data is actually copied in this process. In particular, no additional memory is needed than what
    // you have allocated to store the matrix
    A.interchange(a,ja,ia,nrow,ncol,nnz,orientation);
    std::cout<<std::endl<<std::flush;
    //std::cout<<"        Coefficient Matrix "<<std::endl<<A<<std::flush;

    // make an iluplusplus vector. It has dimension 0 and no memory is allocated.
    iluplusplus::vector w;
    // turn v into an iluplusplus vector w by swapping:
    w.interchange(v,n_v);
    // w now is the preconditioned vector; v is an empty vector
    //  std::cout<<std::endl<<std::flush;
    //  std::cout<<"        preconditioned vector: "<<std::endl<<w<<std::flush;

   // multiply. The result should be A*A^{-1}*w = w.
    A.multiply(w);
    std::cout<<"        preconditioned and multiplied vector -- should be (1,2,...,8)^T:"<<std::endl<<w<<std::flush;

    // interchange again
    A.interchange(a,ja,ia,nrow,ncol,nnz,orientation);
    w.interchange(v,n_v);
    // free memory
    delete [] a;
    delete [] ia;
    delete [] ja;
    delete [] b;
    // for some of these arrays, it might (theoretically) be possible that no memory was allocated.
    if (n_v > 0) delete [] v;
    if (n_x > 0) delete [] x;
    if (n_x_exact > 0) delete [] x_exact;



  // *************************************************************************** //
  // PART IIc: Third possibility: you provide the matrix and rhs,                //
  //           in C++ vectors and ILU++ calculates a solutions and               //
  //           returns a vector a C++ vector                                     //
  // *************************************************************************** //

    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***  PART IIc: solving a linear system for a matrix and rhs (C++ vectors) given by user."<<std::endl;
    std::cout<<"                  ILU++ calculates and returns a solution as a C++ vector"<<std::endl;
    std::cout<<"                  and writes information on results (not solution) to a file."<<std::endl;
    std::cout<<std::endl<<std::flush;


    std::vector<int> ia_vec(9);  // make sure that this is of the same type as iluplusplus::Integer
          ia_vec[0] = 0; ia_vec[1] = 4; ia_vec[2] = 7; ia_vec[3] = 9; ia_vec[4] = 11; ia_vec[5] = 12; ia_vec[6] = 15; ia_vec[7] = 17; ia_vec[8] = 20;

    std::vector<int> ja_vec(20); // make sure that this is of the same type as iluplusplus::Integer
          ja_vec[0]  = 0; ja_vec[1]  = 2; ja_vec[2]  = 5; ja_vec[3]  = 6;
          ja_vec[4]  = 1; ja_vec[5]  = 2; ja_vec[6]  = 4;
          ja_vec[7]  = 2; ja_vec[8]  = 7;
          ja_vec[9]  = 3; ja_vec[10] = 6;
          ja_vec[11] = 1;
          ja_vec[12] = 2; ja_vec[13] = 5; ja_vec[14] = 7;
          ja_vec[15] = 1; ja_vec[16] = 6;
          ja_vec[17] = 2; ja_vec[18] = 6; ja_vec[19] = 7;

    std::vector<Coeff_Field> a_vec(20);
          a_vec[0]  = (Coeff_Field) 7.0;  a_vec[1]  =  (Coeff_Field) 1.0; a_vec[2]  =  (Coeff_Field) 2.0; a_vec[3]  =  (Coeff_Field) 7.0;
          a_vec[4]  = (Coeff_Field) -4.0; a_vec[5]  =  (Coeff_Field) 8.0; a_vec[6]  =  (Coeff_Field) 2.0;
          a_vec[7]  = (Coeff_Field) 1.0;  a_vec[8]  =  (Coeff_Field) 5.0;
          a_vec[9]  = (Coeff_Field) 7.0;  a_vec[10] =  (Coeff_Field) 9.0;
          a_vec[11] = (Coeff_Field) -4.0;
          a_vec[12] = (Coeff_Field)  7.0; a_vec[13] =  (Coeff_Field) 3.0; a_vec[14] =  (Coeff_Field) 8.0;
          a_vec[15] = (Coeff_Field)  1.0; a_vec[16] =  (Coeff_Field) 11.0;
          a_vec[17] = (Coeff_Field) -3.0; a_vec[18] =  (Coeff_Field) 2.0; a_vec[19] =  (Coeff_Field) 5.0;


    // Setup right hand side

    std::vector<Coeff_Field> b_vec(8);
           b_vec[0]  =  (Coeff_Field) 17.0; b_vec[1]  =   (Coeff_Field) 6.0;  b_vec[2]  =   (Coeff_Field) 6.0;  b_vec[3]  =  (Coeff_Field) 16.0;
           b_vec[4]  =  (Coeff_Field) -4.0; b_vec[5]  =   (Coeff_Field) 18.0; b_vec[6]  =   (Coeff_Field) 12.0; b_vec[7]  =  (Coeff_Field) 4.0;

    std::vector<Coeff_Field> x_exact_vec;         // If you do not have an exact solution, do not bother to allocate memory!!!
    std::vector<Coeff_Field> x_vec(8);            // DO allocate memory here!
                                                  // If you do not allocate memory here, ILU++ will throw an exception.

    // here we set the dropping parameter to 10^{-1.0} = 0.1
    param.set_threshold(1.0);
    // reset other parameters (clear values from previous call):
    abs_tol = 8.0; rel_tol = 8.0; max_iter = 600;

    solve_with_multilevel_preconditioner(iluplusplus::ROW,a_vec,ja_vec,ia_vec,b_vec,x_exact_vec,x_vec,false,rel_tol,abs_tol,max_iter,abs_error,output_dir,"test3",param);

    // look at result: should be (1,...1)^T:
    std::cout<<"Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<") and reduction of the initial residual by 10^(-"<<rel_tol<<")"<<std::endl;
    std::cout<<"Solution using solve_with_multilevel_preconditioner with C++ vectors -- should be (1,...1)^T:"<<std::endl<<std::flush;
    for(unsigned int i=0;i<x_vec.size();i++)std::cout<<"x_vec["<<i<<"] = "<<x_vec[i]<<std::endl<<std::flush;    std::cout<<std::endl<<std::flush;

    // suppose you know the solution, then you can check the accuracy of iluplusplus as follows:

    // setup C++ vector x_exact.
    x_exact_vec.resize(8);
           x_exact_vec[0]  =  (Coeff_Field) 1.0; x_exact_vec[1]  =  (Coeff_Field) 1.0; x_exact_vec[2]  =  (Coeff_Field) 1.0; x_exact_vec[3]  =  (Coeff_Field) 1.0;
           x_exact_vec[4]  =  (Coeff_Field) 1.0; x_exact_vec[5]  =  (Coeff_Field) 1.0; x_exact_vec[6]  =  (Coeff_Field) 1.0; x_exact_vec[7]  =  (Coeff_Field) 1.0;
    abs_tol = 8.0; rel_tol = 8.0; max_iter = 600; 
    // erase
    x_vec.resize(8,(Coeff_Field) 0.0);
    // Note:  solve_with_multilevel_preconditioner returns -log10(rel_tol), -log10(abs_tol), -log10(abs_error) achieved
    solve_with_multilevel_preconditioner(iluplusplus::ROW,a_vec,ja_vec,ia_vec,b_vec,x_exact_vec,x_vec,true,rel_tol,abs_tol,max_iter,abs_error,output_dir,"test4",param);

    // look at result: should be (1,...1)^T:
    std::cout<<"Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^(-"<<abs_tol;
    std::cout<<"), a reduction of the initial residual by 10^(-"<<rel_tol<<") and an absolute error of 10^(-"<<abs_error<<")"<<std::endl;
    std::cout<<"Solution using solve_with_multilevel_preconditioner with C++ vectors arrays -- should be (1,...1)^T:"<<std::endl<<std::flush;
    for(unsigned int i=0;i<x_vec.size();i++)std::cout<<"x_vec["<<i<<"] = "<<x_vec[i]<<std::endl<<std::flush;    std::cout<<std::endl<<std::flush;


  // ***************************************************************************************** //
  // PART IId: Fourth possibility: use C++ vectors from the standard template library.         //
  //       You provide the matrix, the rhs (in C++ vectors),                                   //
  //       the matrix-vector-multiplication and the iterative solver.                          //
  //       ILU++ calculates the preconditioner and provides a routine for applying             //
  //       the preconditioner to C++ vectors.                                                  //
  // ***************************************************************************************** //

    std::cout<<std::endl<<std::flush;
    std::cout<<"   ***  PART IId: user provides a matrix and rhs using C++ vectors."<<std::endl;
    std::cout<<"                  ILU++ calculates preconditioner and applies it to a C++ vector."<<std::endl;
    std::cout<<"                  User provides matrix-vector-multiplication and his own solver"<<std::endl;
    std::cout<<std::endl<<std::flush;

    param.set_threshold(1000.0);

    // setup preconditioner
    // a_vec, ja_vec, ia_vec correspond to the compressed sparse row format,
    // next we have the dimension, the number of non-zero elements, the orientation of the matrix,
    // and the parameter for the preconditioner. The routine setup is for use with C++ vectors.

    Pr.setup(a_vec, ja_vec, ia_vec, iluplusplus::ROW, param );

    std::vector<Coeff_Field> v_vec(8);

           v_vec[0]  =  1.0; v_vec[1]  =  2.0;  v_vec[2]  =  3.0; v_vec[3]  =  4.0;
           v_vec[4]  =  5.0; v_vec[5]  =  6.0;  v_vec[6]  =  7.0; v_vec[7]  =  8.0;


    // NOTE: if you want to use your own iterative solver and have implemented your own matrix-vector-multiplication,
    // then you only need a routine to apply the preconditioner to an arbitrary C++ vector.

    Pr.apply_preconditioner(v_vec);
    // now, v_vec is a vector containing A^{-1}v_vec (because we had calculated an "exact" preconditioner, i.e. the inverse of A)
    std::cout<<"        preconditioned vector using C++ vector"<<std::endl<<std::flush;
    for (unsigned int i=0; i<v_vec.size(); i++) std::cout<<v_vec[i]<<std::endl<<std::flush;
    std::cout<<std::endl<<std::flush;

  // *************************************************************************** //
  // PART III: Testing the preconditioner for varying fill-in:                   //
  //           Results are stored in a file *.out                                //
  // *************************************************************************** //

    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<"PART III: testing preconditioner for particular Harwell-Boeing matrices"<<std::endl;
    std::cout<<"          and varying tau. Information on results is stored in *.out"<<std::endl;
    std::cout<<"**********************************************************************************"<<std::endl;
    std::cout<<std::endl<<std::flush;

    param.set_MEM_FACTOR(3.0); // by default, memory is allocated very generously. It is ok to use less.
    // stopping criteria for iteration (used for both the relative and absolute residual)
    iluplusplus::Real eps = 8.0;
    max_iter = 600;
    bool use_rhs = true;  // if true: check to see if RHS is available. If so, use it. Otherwise, construct artificial RHS. if false: always use artificial RHS

    // test matrices: parameters are:
    //           param (preconditioner parameter),
    //           matrix_dir,
    //           matrix_name,
    //           matrix_ending,
    //           minimal value for dropping parameter tau,
    //           maximal value for dropping parameter tau,
    //           number of tests,
    //           stopping eps (iteration stops if both absolute and relative residual are less than eps),
    //           max. iterations allowed,
    //           use RHS if provided
    // note: the values of tau tested are the equidistant (number of tests) values of tau between min_tau and max_tau.
    // the actual dropping parameter is -log10(tau).
    // Hence, in the first test below, tau takes on 21 different values between 10^1 and 10^(-3) in geometric progression.

#ifndef ILUPLUSPLUS_USES_COMPLEX
    // this matrix has a rhs, hence the exact error in the solution is not known. nan is recorded in *.out
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);

    // this matrix has no rhs, so an artifical rhs is constructed using the solution x = (1,...1)^T; the exact error is known and recorded in *.out
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"west2021",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
#endif

#ifdef ILUPLUSPLUS_USES_COMPLEX
    // this matrix has no rhs, so an artifical rhs is constructed using the solution x = (1,...1)^T; the exact error is known and recorded in *.out
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"young1c",".rb",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
#endif
    std::cout<<std::endl<<std::flush;

    // calls for further testing routines are included for your convenience. 
    // The matrices needed are not provided by ILU++. You need to download them yourself.
    // The first set can be found at the Matrix Market: http://math.nist.gov/MatrixMarket/

#ifndef ILUPLUSPLUS_USES_COMPLEX
/*

    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp___200",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp___400",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp___600",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp___800",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp__1000",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp__1200",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp__1400",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bp__1600",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_541_1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_541_2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_541_3",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_541_4",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_680_1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_680_2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_680_1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_760_1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_760_2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"fs_760_3",".rua", -1.0,3.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"gemat11",".rua",-1.0,5.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"gemat12",".rua",-1.0,5.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"gre__512",".rua",-3.0,2.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"gre_1107",".rua",-3.0,3.0,41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"jpwh_991",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lns__511",".rua",-1.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lns_3937",".rua",0.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lnsp_511",".rua",1.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lnsp3937",".rua",0.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"mahindas",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"mcfe",".rua",-1.0,2.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"nnc666",".rua",-2.0,7.0, 61,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"nnc1374",".rua",0.0,8.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"orani678",".rua",-1.0,2., 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"orsirr_1",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"orsirr_2",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"orsreg_1",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"pde2961",".rua",0.0,2.5, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"pores_2",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"pores_3",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"psmigr_1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"psmigr_2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"psmigr_3",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"saylr3",".rua",0.0,6.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"saylr4",".rua",0.0,5.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman3",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman4",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sherman5",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"shl____0",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"shl__200",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"shl__400",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"steam2",".rua",-1.0,2.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"watt__1",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"watt__2",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"west0655",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"west0989",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"west1505",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"west2021",".rua",-1.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
*/

// The following matrices can be found at the University of Florida Collection: http://www.cise.ufl.edu/research/sparse/matrices/

/*
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bayer10",".rua",-1.0,8.0, 61,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"goodwin",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"graham1",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"hydr1",".rua",0.0,9.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"igbt3",".rua",0.0,10.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lhr01",".rua",-1.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lhr02",".rua",-1.0,5.0, 61,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"lhr04",".rua",2.0,12.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"utm3060",".rua",0.0,6.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"utm5940",".rua",0.0,6.0, 41,eps,max_iter,output_dir,use_rhs);
*/

// The following matrices can be found at the University of Florida Collection: http://www.cise.ufl.edu/research/sparse/matrices/
// Because they are larger, they (usually) require a sparser Schur Complement.

// Configuration for larger matrices, sparser Schur complements

    // Standard configurations of both preprocessing and preconditioning. (Somewhat older, better to use newer routines)
    // normalize rows, columns, use PQ. Pivoting by rows and columns.
    // param.default_configuration(1000);
    // I-matrix preprocessing; pivot by rows and columns
    param.default_configuration(1010);
    // I-matrix preprocessing; Additional preprocessing to make a diagonally dominant submatrix. No pivoting
    // param.default_configuration(1011);

/*
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bayer01",".rua",-1.0,8.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"bcircuit",".rua",2.0,7.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"circuit_4",".rua",3.0,9.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"epb2",".rua",0.0,2.5, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"epb3",".rua",0.0,4.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"hcircuit",".rua",0.0,5.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"nmos3",".rua",2.0,7.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"onetone1",".rua",3.0,9.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"onetone2",".rua",3.0,9.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"raefsky3",".rua",0.0,5.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"scircuit",".rua",0.0,10.0, 41,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sme3Da",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"sme3Db",".rua",0.0,4.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"twotone",".rua",3.0,9.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"venkat01",".rua",0.0,1.5, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"venkat25",".rua",0.0,1.5, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"venkat50",".rua",0.0,1.5, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"wang3",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
    iluplusplus::test_multilevel_preconditioner(param, matrix_dir,"wang4",".rua",0.0,3.0, 21,eps,max_iter,output_dir,use_rhs);
*/
#endif
    return 0;
  }
  catch(iluplusplus::iluplusplus_error ippe){
      std::cerr<<"main: ERROR: "<<ippe.error_message()<<". Returning."<<std::endl;
      return 1;
  }
  catch(std::bad_alloc){
      std::cerr<<"main: Error allocating memory. Returning."<<std::endl;
      return 1;
  }
}
