/***************************************************************************
 *   Copyright (C) 2007 by Jan Mayer                                       *
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



// this program uses only the ILU++ interface to allow easy use in other programs.
// it requires the iluplusplus library including the most important features of ILU++

// include the classes and functions defined in the library iluplusplus(version).a
#include "iluplusplus_interface.h"



int main(int argc, char *argv[]){
  try {
  // *************************************************************************** //
  // PART I: solving linear systems stored in Harwell-Boeing format.             //
  // *************************************************************************** //

    if(argc != 9){
        std::cout<<"main_solve: arguments: <preprocessing number> <preconditioner number> <memory factor> <-log10(threshold)> <matrix file> <output directory> <subdirectory output data> <matrix name>"<<std::endl;
        return 1;
    }
    iluplusplus::preprocessing_sequence L;
    iluplusplus::iluplusplus_precond_parameter param;

    switch ( atoi(argv[1]) ){
        case -1:  L.set_none(); break;
        case  0:  L.set_normalize(); break;
        case  1:  L.set_PQ(); break;
        case  2:  L.set_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case  3:  L.set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG(); break;
        case  4:  L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case  5:  L.set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM(); break;
        case  6:  L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case  7:  L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG(); break;
        case  8:  L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case  9:  L.set_SPARSE_FIRST_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG_DD_MOV_COR_IM(); break;
        case 10:  L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case 11:  L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        #ifdef ILUPLUSPLUS_USES_METIS
        case 12:  L.set_METIS_NODE_ND_ORDERING(); break;
        case 13:  L.set_PQ_METIS_NODE_ND_ORDERING(); break;
        #endif
        case 14: L.set_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case 15: L.set_MAX_WEIGHTED_MATCHING_ORDERING_PQ(); break;
        case 16: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_PQ(); break;
        case 17: L.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(); break;
        case 18: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(); break;
        case 19: L.set_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(); break;
        case 20: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(); break;
        case 21: L.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(); break;
        case 22: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(); break;
        case 23: L.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(); break;
        case 24: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(); break;
        case 25: L.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(); break;
        case 26: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(); break;
        case 27: L.set_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(); break;
        case 28: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(); break;
        case 29: L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case 30: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case 31: L.set_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(); break;
        case 32: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(); break;
        case 33: L.set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(); break;
        case 34: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(); break;
        case 35: L.set_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(); break;
        case 36: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(); break;
        case 37: L.set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(); break;
        case 38: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(); break;
        case 39: L.set_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(); break;
        case 40: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(); break;
        #ifdef ILUPLUSPLUS_USES_METIS
        case 41: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(); break;
        case 42: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(); break;
        case 43: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(); break;
        case 44: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(); break;
        case 45: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(); break;
        case 46: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(); break;
        case 47: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(); break;
        case 48: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(); break;
        case 49: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(); break;
        case 50: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(); break;
        case 51: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(); break;
        case 52: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(); break;
        case 53: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(); break;
        case 54: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(); break;
        case 55: L.set_MAX_WEIGHTED_MATCHING_ORDERING_METIS(); break;
        case 56: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS(); break;
        case 57: L.set_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(); break;
        case 58: L.set_NORM_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(); break;
        case 59: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(); break;
        case 60: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(); break;
        case 61: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(); break;
        case 62: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(); break;
        case 63: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(); break;
        case 64: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(); break;
        case 65: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(); break;
        case 66: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(); break;
        case 67: L.set_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(); break;
        case 68: L.set_NORM_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(); break;
        #endif
        #ifdef ILUPLUSPLUS_USES_PARDISO
        case 69:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case 70:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING(); break;
        case 71:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ(); break;
        case 72:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_PQ(); break;
        case 73:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(); break;
        case 74:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER(); break;
        case 75:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(); break;
        case 76:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_MOVE_CORNER_IM(); break;
        case 77:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(); break;
        case 78:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR(); break;
        case 79:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(); break;
        case 80:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT_MOV_COR_IM(); break;
        case 81:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(); break;
        case 82:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR(); break;
        case 83:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(); break;
        case 84:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_WGT2_MOV_COR_IM(); break;
        case 85:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case 86:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); break;
        case 87:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(); break;
        case 88:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYM_PQ(); break;
        case 89:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(); break;
        case 90:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER(); break;
        case 91:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(); break;
        case 92:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SYMB_MOVE_CORNER_IM(); break;
        case 93:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(); break;
        case 94:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER(); break;
        case 95:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(); break;
        case 96:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_SP_MOVE_CORNER_IM(); break;
        #endif
        #if defined(ILUPLUSPLUS_USES_METIS) && defined(ILUPLUSPLUS_USES_PARDISO)
        case 97:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(); break;
        case 98:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER(); break;
        case 99:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(); break;
        case 100:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_MOVE_CORNER_IM(); break;
        case 101:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(); break;
        case 102:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR(); break;
        case 103:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(); break;
        case 104:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_WGT_MOV_COR_IM(); break;
        case 105:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(); break;
        case 106:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_DD_MOV_COR_IM(); break;
        case 107:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(); break;
        case 108:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR(); break;
        case 109:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(); break;
        case 110:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SP_MOV_COR_IM(); break;
        case 111:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS(); break;
        case 112:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS(); break;
        case 113:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(); break;
        case 114:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_ORDERING_METIS_PQ(); break;
        case 115:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(); break;
        case 116:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SYMMPQ(); break;
        case 117:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(); break;
        case 118:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMC(); break;
        case 119:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(); break;
        case 120:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MC(); break;
        case 121:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(); break;
        case 122:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_SMCI(); break;
        case 123:  L.set_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(); break;
        case 124:  L.set_NORM_PARDISO_MAX_WEIGHTED_MATCHING_METIS_ORDERING_W2MCI(); break;
        #endif  // end PARDISO & METIS available
        default:  std::cout<<"Unknown value for preprocessing type. Using default."<<std::endl; L.set_MAX_WEIGHTED_MATCHING_ORDERING_UNIT_DIAG(); break;
    }

    param.init(L,atoi(argv[2]),""); 
    param.set_MEM_FACTOR(atof(argv[3]));
    param.set_threshold(atof(argv[4]));
    param.set_MAX_FILLIN_IS_INF(true);


    // declare (sparse) matrices and (dense) vectors
    iluplusplus::matrix Acol;
    iluplusplus::vector X,X_exact,B;

    Acol.read_hb(argv[5]);
    B.read_hb(argv[5],1);

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

    // set first tolerance for terminating iterative method (also used as return variable for reduction of residual obtained, i.e. the relative residual)
    iluplusplus::Real rel_tol   = 8.0;  // actual tolerance is 10^{-rel_tol}
    // set second tolerance for terminating iterative method (also used as return variable for residual obtained, i.e. the absolute residual)
    iluplusplus::Real abs_tol   = 8.0;  // actual tolerance is 10^{-abs_tol}
    // variable for returning the absolute error (if exact solution is known -- otherwise nan is returned)
    iluplusplus::Real abs_error;
    // maximal number of iterations. The variable returns the number of iterations required for convergence
    iluplusplus::Integer max_iter = 600;

    std::string output_directory   = argv[6];
    std::string subdirectory_data  = (output_directory + argv[7]);
    std::string matrix_name        = argv[8];

    bool success;
    success = iluplusplus::solve_with_multilevel_preconditioner(Acol,B,X_exact,X,!have_rhs,rel_tol,abs_tol,max_iter,abs_error,output_directory,matrix_name,param,true,subdirectory_data);

    if(success) {
        std::cout<<"        Solved in "<<max_iter<<" iterations and achieved an absolute residual of 10^("<<-abs_tol;
        std::cout<<") and reduction of the initial residual by 10^("<<-rel_tol<<")"<<std::endl;
        //std::cout<<"The calculated solution is "<<std::endl<<x;
    } else {
        std::cout<<"Iteration failed."<<std::endl;
    }


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
