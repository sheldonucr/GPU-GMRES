/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: partition2.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Duo Li
 *
 *    Functions: Partition matrix with PaToH
 *
 */

#include <set>
#include "cs.h"
#include "patoh.h"

using namespace std;

void partition2(cs_dl *A, int npart, int nNodes,
			   UF_long *part_size, UF_long *node_part, UF_long *pinv, UF_long *q)
{
  UF_long m = A->m;
  UF_long avg_size = nNodes/npart;
  // int *node_part = new int[m];
  UF_long i = 0, j = 0, p = 0;
  UF_long *Ap = A->p, *Ai = A->i;

  PaToH_Parameters args;
  int _c, _n, _nz, _nconst, *cwghts, *nwghts, 
	*xpins, *pins, *partvec, cut, *partweights;

  _nz = A->nzmax;
  _c = A->m;
  _n = _nz;
  _nconst = 1;
  nwghts = NULL;
  cwghts = (int *) malloc((_c)*sizeof(int));
  for (int c = 0; c < _c; c++){
	cwghts[c] = 1;
  }  
  xpins = (int *) malloc((_n+1)*sizeof(int));
  pins = (int *) malloc(2*_n*sizeof(int));

  int xpins_index = 0;
  int pins_index = 0;
  for (j = 0; j < m; j++){
	for (p = Ap[j]; p < Ap[j+1]; p++){
	  xpins[xpins_index++] = pins_index;
	  pins[pins_index++] = Ai[p];
	  pins[pins_index++] = j;
	}
  }
  xpins[xpins_index] = pins_index;

  printf("#cells = %6d  #nets=%d  #pins=%d\n", _c, _n, xpins[_n]);
 
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, 
							  PATOH_SUGPARAM_DEFAULT);

  args._k = npart;
  // PaToH_Process_Arguments(&args, 3, argc, argv, NULL);
  args.MemMul_CellNet = 4;
  args.MemMul_Pins = 4;
 
  partvec = (int *) malloc(_c*sizeof(int));
  partweights = (int *) malloc(args._k*_nconst*sizeof(int));
  PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, 
			  xpins, pins);

  if (_nconst==1)
	PaToH_Partition(&args, _c, _n, cwghts, nwghts, 
					xpins, pins, partvec, partweights, &cut);
  else
	PaToH_MultiConst_Partition(&args, _c, _n, _nconst, cwghts,
							   xpins, pins, partvec, partweights, &cut);

  printf("%d-way cutsize is: %d\n", args._k, cut);

  // PrintInfo(args._k, partweights,  cut, _nconst);
  
  for (int c = 0; c < _c; c++){
	node_part[c] = partvec[c];
  }
  free(partweights);
  free(partvec); 
  PaToH_Free();

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
