/*! \file 
	\brief This file defines some test functions for AINV preconditioner
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

#include "cublas.h"// might introduce redefine warning
//#include <cutil.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>

#include "gmres.h"



#include <cusp/precond/ainv.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>

#include<iostream>

template <typename Monitor> 
void report_status(Monitor& monitor)
{
	if (monitor.converged())
	{
		std::cout << "Solver converged to " << monitor.tolerance() << " tolerance";
		std::cout << " after " << monitor.iteration_count() << " iterations";
		std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
	}
	else
	{
		std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
		std::cout << " to " << monitor.tolerance() << " tolerance ";
		std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
	}
}

int ainvTest(void)
{
	typedef int                 IndexType;
	typedef float               ValueType;
	typedef cusp::device_memory MemorySpace;

	// create an empty sparse matrix structure
	cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;

	// create 2D Poisson problem
	cusp::gallery::poisson5pt(A, 256, 256);

	// solve without preconditioning
	{
		std::cout << "\nSolving with no preconditioner" << std::endl;

		// allocate storage for solution (x) and right hand side (b)
		cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
		cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

		// set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-6)
		cusp::default_monitor<ValueType> monitor(b, 1000, 1e-6);

		// solve
		cusp::krylov::cg(A, x, b, monitor);

		// report status
		report_status(monitor);
	}

	// solve AINV preconditioner, using standard drop tolerance strategy 
	{
		std::cout << "\nSolving with scaled bridson preconditioner (drop tolerance .1)" << std::endl;

		// allocate storage for solution (x) and right hand side (b)
		cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
		cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

		// set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-6)
		cusp::default_monitor<ValueType> monitor(b, 1000, 1e-6);

		// setup preconditioner
		cusp::precond::scaled_bridson_ainv<ValueType, MemorySpace> M(A, .1);

		// solve
		cusp::krylov::cg(A, x, b, monitor, M);

		// report status
		report_status(monitor);
	}


	// solve AINV preconditioner, using novel dropping strategy 
	{
		std::cout << "\nSolving with scaled bridson preconditioner (10 nonzeroes per row)" << std::endl;

		// allocate storage for solution (x) and right hand side (b)
		cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
		cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

		// set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-6)
		cusp::default_monitor<ValueType> monitor(b, 1000, 1e-6);

		// setup preconditioner
		cusp::precond::scaled_bridson_ainv<ValueType, MemorySpace> M(A, 0, 10);

		// solve
		cusp::krylov::cg(A, x, b, monitor, M);

		// report status
		report_status(monitor);
	}

	return 0;
}


