/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: partition3.cpp,v $
 *    $Revisio$
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Duo Li
 *
 *    Functions: Partition matrix with METIS
 *
 */

#include <set>
#include "cs.h"
#include "metis.h"
//#include "metislib.h"

using namespace std;

void partition3(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q)
{
  UF_long m = A->m;
  UF_long i = 0, j = 0, p = 0;
  UF_long *Ap = A->p, *Ai = A->i;


  idxtype nvtxs, *xadj, *adjncy;
  idxtype nparts, numflag, wgtflag, edgecut, options[10];
  idxtype *part, *vwgt, *adjwgt;

  int nnets = A->nzmax;
  nvtxs= A->m;
  numflag = 0; 
  wgtflag = 0; 
  nparts = npart;
  options[0] = 0;
  xadj = (idxtype *) malloc((nvtxs+1)*sizeof(idxtype));
  adjncy = (idxtype *) malloc(nnets*sizeof(idxtype));
  part = (idxtype *) malloc(nvtxs*sizeof(idxtype));

  idxtype adjncy_index = 0;
  for (j = 0; j < m; j++){
	xadj[j] = adjncy_index;
	for (p = Ap[j]; p < Ap[j+1]; p++){
	  if (Ai[p] != j){
		adjncy[adjncy_index++] = Ai[p];
	  }
	}
  }
  xadj[m] = adjncy_index;
  adjncy = (idxtype *) realloc(adjncy, adjncy_index*sizeof(idxtype));

  printf("  #Vertices: %d, #Edges: %d\n", nvtxs, nnets);

  METIS_PartGraphRecursive(&nvtxs, xadj, adjncy, NULL, NULL, &wgtflag, &numflag,
						   &nparts, options, &edgecut, part);

  printf("%d-way cutsize is: %d\n", nparts, edgecut);

  for (int v = 0; v < nvtxs; v++){
	node_part[v] = part[v];
  }

  for (int k = 0; k < npart+1; k++){
	part_size[k] = 0;
  }
  for (j = 0; j < m; j++){
	part_size[node_part[j]]++;
  }
  // Adjust partitioning
  set <int> toplevel;
  for (j = 0; j < m; j++){
	for (p = Ap[j]; p < Ap[j+1]; p++){
	  if (node_part[Ai[p]] != node_part[j]){
		toplevel.insert(Ai[p]);
		toplevel.insert(j);
	  }
	}
  }
  for(set<int>::iterator iter = toplevel.begin(); iter != toplevel.end(); iter++){
	node_part[*iter] = npart;
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

void partition4(idxtype *xadj, idxtype *adjncy, int npart, UF_long m, int nNodes, UF_long *vnode,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q)
{
  UF_long i = 0, j = 0, p = 0;
  
  idxtype nvtxs;
  idxtype nparts, numflag, wgtflag, edgecut, options[10];
  idxtype *part, *vwgt, *adjwgt;

  nvtxs= nNodes;
  numflag = 0; 
  wgtflag = 0; 
  nparts = npart;
  options[0] = 0;
  part = (idxtype *) malloc(nvtxs*sizeof(idxtype));

  printf("#Vertices: %d, #Edges: %d\n", nvtxs, xadj[nNodes]/2);

  METIS_PartGraphRecursive(&nvtxs, xadj, adjncy, NULL, NULL, &wgtflag, &numflag,
						   &nparts, options, &edgecut, part);

  printf("%d-way cutsize is: %d\n", nparts, edgecut);

  for (int v = 0; v < nvtxs; v++){
	node_part[v] = part[v];
  }
  // Adjust partitioning
  set<int> toplevel;
  for (j = 0; j < nNodes; j++){
	for (p = xadj[j]; p < xadj[j+1]; p++){
	  if (node_part[adjncy[p]] != node_part[j]){
		  toplevel.insert(adjncy[p]);
		  toplevel.insert(j);
	  }
	}
  }
  for(set<int>::iterator iter = toplevel.begin(); iter != toplevel.end(); iter++){
	node_part[*iter] = npart;
  }
  for (j = nNodes; j < m; j++){
	node_part[j] = node_part[vnode[j-nNodes]];
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
  free(part);
}
