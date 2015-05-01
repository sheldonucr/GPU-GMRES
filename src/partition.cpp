/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: partition.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Duo Li
 *
 *    Functions: Partition matrix
 *
 */

#include "cs.h"

void partition(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q)
{
  UF_long m = A->m;
  UF_long avg_size = nNodes/npart;
  // int *node_part = new int[m];
  UF_long i = 0, j = 0, p = 0;
  UF_long *Ap = A->p, *Ai = A->i;
  // Initial partitioning
  for (i = 0; i < m; i++){
	node_part[i] = -1;
  }
  for (i = 0; i < nNodes; i++){
	if (i >= j*avg_size && i < (j+1)*avg_size){
	  node_part[i] = j;
	}else if (i >= (j+1)*avg_size){
	  if (j+1 < npart)
		node_part[i] = ++j;
	  else if (j+1 == npart)
		node_part[i] = j;
	  else
		node_part[i] = 0;
	}
  }
  for (int k = 0; k < npart+1; k++){
	part_size[k] = 0;
  }
  for (j = 0; j < m; j++){
	if (node_part[j] != -1)
	  part_size[node_part[j]]++;
  }
  // Adjust partitioning
  for (j = 0; j < m; j++){
	for (p = Ap[j]; p < Ap[j+1]; p++){
	  if (node_part[Ai[p]] == -1){
		node_part[Ai[p]] = node_part[j];
		continue;
	  }
	  if (node_part[Ai[p]] != npart && node_part[j] != npart){
		if (node_part[Ai[p]] != node_part[j]){
		  if (j > Ai[p])
			node_part[j] = npart;
		  else
			node_part[Ai[p]] = npart;
		}
	  }
	}
  }
  // Generate partition results
  // int *part_size = new int[npart+1];
  for (int k = 0; k < npart+1; k++){
	part_size[k] = 0;
  }
  for (j = 0; j < m; j++){
	part_size[node_part[j]]++;
  }  
  // Generate pinv and q
  UF_long *part_begin = new UF_long[npart+1];
  UF_long *part_current = new UF_long[npart+1];
  for (int k = 0; k < npart+1; k++){
	part_current[k] = 0;
  }
  UF_long sum = 0;
  for (i = 0; i < npart+1; i++){
	sum = 0;
	for (j = 0; j < i; j++){
	  sum += part_size[j];
	}
	part_begin[i] = sum;
  }
  for (j = 0; j < m; j++){
	pinv[j] = part_begin[node_part[j]] + part_current[node_part[j]];
	q[part_begin[node_part[j]] + part_current[node_part[j]]] = j;
	part_current[node_part[j]]++;
  }
  delete [] part_begin;
  delete [] part_current;
}

