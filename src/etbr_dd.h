/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_dd.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR_DD header
 *
 */

#ifndef ETBR_DD_H
#define ETBR_DD_H

#include "cs.h"
#include "etbr.h"
#include <iostream>
#include <fstream>
#include <itpp/base/timing.h>
#include <itpp/base/mat.h>
#include "metis.h"

using namespace itpp;
using namespace std;

void partition(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q);

void partition2(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q);

void partition3(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q);

void partition4(idxtype *xadj, idxtype *adjncy, int npart, UF_long m, int nNodes, UF_long* vnode,
				UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q);

void dd_form(int npart, UF_long *part_size, UF_long *node_part, cs_dl *A, double *b, 
			 cs_dl **&As, cs_dl **&E, cs_dl **&F, cs_dl *&At, double **&f, double *&g);

void dd_solve(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			  double **f, double *g, double *z,
			  Real_Timer &symbolic_runtime, Real_Timer &numeric_runtime, Real_Timer &solve_runtime);

void dd_solve2(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			   double **f, double *g, double *z,
			   Real_Timer &symbolic_runtime, Real_Timer &numeric_runtime, Real_Timer &solve_runtime);

void dd_solve3(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			   double **f, double *g, double *z,
			   Real_Timer &symbolic_runtime, Real_Timer &numeric_runtime, Real_Timer &solve_runtime);

void dd_solve_ooc(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
				  double **f, double *g, double *z,
				  Real_Timer &cs_symbolic_runtime, Real_Timer &cs_numeric_runtime, Real_Timer &cs_solve_runtime);

int my_cs_dl_lsolve (const cs_dl *L, double *x);

int my_cs_dl_usolve (const cs_dl *U, double *x);
/*
void etbr_dd(cs_dl *G, cs_dl *C, cs_dl *B, 
			 Source *VS, int nVS, Source *IS, int nIS, 
			 double tstep, double tstop, int q, 
			 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value,
			 int npart, UF_long *part_size, UF_long *node_part, 
			 UF_long *mat_pinv, UF_long *mat_q);
*/
void etbr_dd(cs_dl *B, 
			 Source *VS, int nVS, Source *IS, int nIS, 
			 double tstep, double tstop, int q, 
			 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value,
			 int npart, UF_long *part_size, UF_long *node_part, 
			 UF_long *mat_pinv, UF_long *mat_q);

void dc_dd_solver(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value,
				  int npart, UF_long *part_size, UF_long *node_part, 
				  UF_long *mat_pinv, UF_long *mat_q);

void numeric_dl_save(ofstream &file, cs_dln *N);

void numeric_dl_load(ifstream &file, cs_dln *&N);

void cs_dl_save(ofstream &file, cs_dl *A);

void cs_dl_load(ifstream &file, cs_dl *&A);

#endif
