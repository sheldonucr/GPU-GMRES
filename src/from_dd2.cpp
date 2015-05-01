/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: from_dd2.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:56 $
 *    Authors: Duo Li
 *
 *    Functions: Form As, E, F, At, f, g for Domain Decomposition version 2
 *
 */

#include <iostream>
#include "cs.h"
#include "etbr_dd.h"

void dd_form(int npart, UF_long *part_size, UF_long *node_part, cs_dl *A, double *b, 
			 cs_dl **&As, cs_dl **&E, cs_dl **&F, cs_dl *&At, double **&f, double *&g)
{
  UF_long *Ap = A->p, *Ai = A->i;
  double *Ax = A->x;
  UF_long *part_begin = new UF_long[npart+1];
  UF_long sum = 0;
  for (int i = 0; i < npart+1; i++){
	sum = 0;
	for (int j = 0; j < i; j++){
	  sum += part_size[j];
	}
	part_begin[i] = sum;
  }
  // form As and F
  cs_dl *TAs, *TF, *TE;
  for (int k = 0; k < npart; k++){
	TAs = cs_dl_spalloc(part_size[k], part_size[k], 1, 1, 1);
	TF = cs_dl_spalloc(part_size[npart], part_size[k], 1, 1, 1);
	for (UF_long j = part_begin[k]; j < part_begin[k+1]; j++){
	  for (UF_long p = Ap[j]; p < Ap[j+1]; p++){
		if (Ai[p] >= part_begin[k] && Ai[p] < part_begin[k+1]){
		  cs_dl_entry(TAs, Ai[p]-part_begin[k], j-part_begin[k], Ax[p]);
		}else if (Ai[p] >= part_begin[npart]){
		  cs_dl_entry(TF, Ai[p]-part_begin[npart], j-part_begin[k], Ax[p]);
		}else{
		  std::cout << "part " << k << ": " << "j =  " << j << " Ai[" << p << "] = " << Ai[p] << std::endl;
		}
	  }
	}
	As[k] = cs_dl_compress(TAs);
	cs_dl_spfree(TAs);
	F[k] = cs_dl_compress(TF);
	cs_dl_spfree(TF);
  }
  // form E
  for (int k = 0; k < npart; k++){
	TE = cs_dl_spalloc(part_size[k], part_size[npart], 1, 1, 1);
	for (UF_long j = part_begin[npart]; j < A->n; j++){
	  for (UF_long p = Ap[j]; p < Ap[j+1]; p++){
		if (Ai[p] >= part_begin[k] && Ai[p] < part_begin[k+1]){
		  cs_dl_entry(TE, Ai[p]-part_begin[k], j-part_begin[npart], Ax[p]);
		}
	  }
	}
	E[k] = cs_dl_compress(TE);
	cs_dl_spfree(TE);
  }
  // form At
  cs_dl *TAt = cs_dl_spalloc(part_size[npart], part_size[npart], 1, 1, 1);
  for (UF_long j = part_begin[npart]; j < A->n; j++){
	for (UF_long p = Ap[j]; p < Ap[j+1]; p++){
	  if (Ai[p] >= part_begin[npart]){
		cs_dl_entry(TAt, Ai[p]-part_begin[npart], j-part_begin[npart], Ax[p]);
	  }
	}
  }
  At = cs_dl_compress(TAt);
  cs_dl_spfree(TAt);
  // form f
  for (int k = 0; k < npart; k++){
	f[k] = new double[part_size[k]];
	for (UF_long i = part_begin[k]; i < part_begin[k+1]; i++){
	  f[k][i-part_begin[k]] = b[i];
	}
  }
  // form g
  g = new double[part_size[npart]];
  for (UF_long i = part_begin[npart]; i < A->m; i++){
	g[i-part_begin[npart]] = b[i];
  }
}

void dd_solve(int npart, cs_dl **As, cs_dl **E, cs_dl **F, cs_dl *At, 
			 double **f, double *g, double *z)
{

}
