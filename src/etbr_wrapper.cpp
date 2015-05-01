/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_wrapper.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:56 $
 *    Authors: Duo Li and Xue-Xin Liu
 *
 *    Functions: main function wrapper
 *
 */

#include <iostream>
#include <fstream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <itpp/base/timing.h>
#include <itpp/base/smat.h>
#include <itpp/base/math/min_max.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/sort.h>
#include "interp.h"
//#include "parser_sub.h"
#include "parser.h"
#include "circuit.h"
#include "namepool.h"
#include "mna.h"
#include "cs.h"
#include "itpp2csparse.h"
#include "etbr.h"
#include "etbr_dd.h"
#include "metis.h"
#include "etbr_wrapper.h"

using namespace itpp;
using namespace std;

void parser_wrapper(char cktname[], 
		    cs_dl*& Gs, cs_dl*& Cs, cs_dl*& Bs, 
		    Source*& VS, int& nVS, Source*& IS, int& nIS,
		    int& nNodes, int& nport, double& tstep, double& tstop,
		    int& dc_sign,
		    vector<string>& port_name, ivec& port,
		    vector<int>& tc_node, vector<string>& tc_name,
		    gpuETBR *myGPUetbr)
{
  NodeList *nodePool;
  int nL,nsubnode;
  matrix* G, *C, *B; 
  int size_G, size_C, row_B, col_B;
  NodeList::Node* node_tmp;

  if((nodePool = new NodeList)==NULL){
	printf("Out of memory!\n"); exit(1);
  }

  printf("start parser ...\n");
  nsubnode = 0;
  //parser(cktname, tstep, tstop, nVS, nIS, nL, nodePool);
  parser_sub(cktname, tstep, tstop, nsubnode, nVS, nIS, nL, nodePool);
  // nodePool->map_clear();
  //printf("get port information.\n");
  /* get port information */
  if (tstep == 0 && tstop == 0){
	dc_sign = 1;
  } else{
    dc_sign = 0;
  }


  if (dc_sign){
    map<int,string> name_table;
    nodePool->get_name_table(name_table);
    nport = nodePool->numNode();
    port.set_size(nport);
    for (int i = 0; i < nport; i++){
      port(i) = i;
      port_name.push_back(name_table[i]);
    }
    name_table.clear();
  } else{
    nport = nodePool->numPort();
    port.set_size(nport);
    for (int i = 0; i < nport; i++){
      node_tmp = nodePool->getPort(i);
      string pn = nodePool->getPortName(i);
      port(i) = node_tmp->row_no;
      port_name.push_back(pn);
    }
    for (int i = 0; i < nodePool->getTCNum(); i++){
      node_tmp = nodePool->getTCNode(i);
      string tcn = nodePool->getTCName(i);
      tc_node.push_back(node_tmp->row_no);
      tc_name.push_back(tcn);
    }
  }


  /* initialize G,C,B,VS,IS */
  nNodes = nodePool->numNode();
  printf("node number: %d\n", nNodes);
  printf("voltage source number: %d\n", nVS);
  printf("current source number: %d\n", nIS);
  size_G = nNodes + nL + nVS + nsubnode;
  size_C = size_G;
  row_B = size_G;
  col_B = nVS + nIS;

  if((G = new matrix(size_G, size_G)) == NULL){
	printf("Out of memory!\n"); exit(1);
  }
	
  if((C = new matrix(size_C, size_C)) == NULL){
	printf("Out of memory!\n"); exit(1);
  }

  if((B = new matrix(row_B, col_B)) == NULL){
	printf("Out of memory!\n"); exit(1);
  }
  if((VS = new Source[nVS])==NULL){ 
	printf("Out of memory.\n"); exit(1); 
  }
  if((IS = new Source[nIS])==NULL){ 
	printf("Out of memory.\n"); exit(1); 
  }
  
  printf("start stamping circuit...\n");
  if(nsubnode != 0){
    stamp_sub(cktname, nL, nIS, nVS, nNodes, tstep, tstop, VS, IS, G,  C,  B, nodePool, myGPUetbr); // XXLiu
	Bs = B->mat2csdl();
	delete B;
	printf("B matrix done.\n");
	Cs = C->mat2csdl();
	printf("C matrix done.\n");
	delete C;
	Gs = G->mat2csdl();
	delete G;
	printf("G matrix done.\n");
  }
  else{
    stampB(cktname, nL, nIS, nVS, nNodes, tstop, VS, IS, B, nodePool, myGPUetbr); // XXLiu
	Bs = B->mat2csdl();
	delete B;
	printf("B matrix done.\n");
	stampC(cktname, nL, nVS, nNodes, C, nodePool);
	Cs = C->mat2csdl();
	delete C;
	printf("C matrix done.\n");
	stampG(cktname, nL, nVS, nNodes, G, nodePool);
	Gs = G->mat2csdl();
	delete G;
	printf("G matrix done.\n");	
  }
  printf("stamping complete.\n");
  printf("parser complete.\n");
  delete nodePool;

}

void etbr_dc_wrapper(cs_dl* Gs, cs_dl* Bs, 
					 Source* VS, int nVS, Source* IS, int nIS, 
					 int nport, ivec& port,
					 vec& dc_value, vec& dc_port_value)
{
  //dc_solver2(Gs, Bs, VS, nVS, IS, nIS, dc_value);
  dc_solver2(Gs, Bs, VS, nVS, IS, nIS, dc_port_value);
  delete [] VS;
  delete [] IS;
  cs_dl_spfree(Gs);
  cs_dl_spfree(Bs);
  /*
  dc_port_value.set_size(nport);
  if (nport > 0){
	for(int i = 0; i < nport; i++){
	  dc_port_value(i) = dc_value.get(port(i));
	}
  }
  */
}

void partition_wrapper(string& GC_file_name, cs_dl* Gs, cs_dl* Cs, int nNodes, int npart,
					   UF_long* part_size, UF_long* node_part, UF_long* mat_pinv, UF_long* mat_q)
{

  UF_long *Gp, *Gi, *Cp, *Ci;
  UF_long m = Gs->m;
  UF_long nzmax = Gs->nzmax;
  Gp = Gs->p;
  Gi = Gs->i;
  if (Cs != NULL){
	nzmax += Cs->nzmax;
	Cp = Cs->p;
	Ci = Cs->i;
  }
  UF_long i = 0, j = 0, p = 0;
  idxtype *xadj = (idxtype *) malloc((nNodes+1)*sizeof(idxtype));
  idxtype *adjncy = (idxtype *) malloc(nzmax*sizeof(idxtype));
  idxtype adjncy_index = 0;
  set<UF_long> adjncy_set;
  for (j = 0; j < nNodes; j++){
	xadj[j] = adjncy_index;
	for (p = Gp[j]; p < Gp[j+1]; p++){
	  if (Gi[p] < nNodes) 
		adjncy_set.insert(Gi[p]);
	}
	if (Cs != NULL){
	  for (p = Cp[j]; p < Cp[j+1]; p++){
		if (Ci[p] < nNodes)
		  adjncy_set.insert(Ci[p]);
	  }
	}
	for (set<UF_long>::iterator iter = adjncy_set.begin(); iter != adjncy_set.end(); iter++){
	  if (*iter != j){
		adjncy[adjncy_index++] = *iter;
	  }
	}
	adjncy_set.clear();
  }
  xadj[nNodes] = adjncy_index;
  UF_long *vnode = new UF_long[m-nNodes];
  for (j = nNodes; j < m; j++){
	for (p = Gp[j]; p < Gp[j+1]; p++){
	  vnode[j-nNodes] = Gi[p];
	}
  }
  adjncy = (idxtype *) realloc(adjncy, adjncy_index*sizeof(idxtype));	  
  ofstream out_GC_file;
  out_GC_file.open(GC_file_name.c_str(), ios::binary);
  cs_dl_save(out_GC_file, Gs);
  cs_dl_spfree(Gs);
  if (Cs != NULL){
	cs_dl_save(out_GC_file, Cs);
	cs_dl_spfree(Cs);
  }
  out_GC_file.close();
  
  partition4(xadj, adjncy, npart, m, nNodes, vnode, part_size, node_part, mat_pinv, mat_q);
  delete [] vnode;
  free(xadj);
  free(adjncy);

  cout << "**** Partition results ****" << endl;
  UF_long sum = 0;
  for (int i = 0; i < npart+1; i++){
	cout << "part_size[" << i << "] = " << part_size[i] << endl;
	sum += part_size[i];
  }
  cout << "sum of partitions = " << sum << endl;
  cout << "**** ***************** ****" << endl;

}

void etbr_dd_wrapper(string& GC_file_name, cs_dl* Gs, cs_dl* Cs, cs_dl* Bs, 
					 Source* VS, int nVS, Source* IS, int nIS, 
					 vec& dc_value, vec& dc_port_value, int dc_sign,
					 int npart, int nport, ivec& port, int q, double tstep, double tstop,
					 UF_long* part_size, UF_long* node_part, UF_long* mat_pinv, UF_long* mat_q,
					 mat& Gr, mat& Cr, mat& Br, mat& X, mat& sim_value,
					 mat& Xp, mat& sim_port_value)
{  
  if (dc_sign == 1){
	ifstream in_GC_file;
	in_GC_file.open(GC_file_name.c_str(), ios::binary);
	cs_dl_load(in_GC_file, Gs);
	in_GC_file.close();
	dc_dd_solver(Gs, Bs, VS, nVS, IS, nIS, dc_value, 
				 npart, part_size, node_part, mat_pinv, mat_q);
		
	dc_port_value.set_size(nport);
	if (nport > 0){
	  for(int i = 0; i < nport; i++){
		dc_port_value(i) = dc_value.get(port(i));
	  }
	}else{
	  etbr_dd(Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
			  Gr, Cr, Br, X, sim_value, npart, part_size, node_part, mat_pinv, mat_q);
	
	  Xp.set_size(nport, q);
	  sim_port_value.set_size(nport, sim_value.cols());
	  if (nport > 0){
		for(int i = 0; i < nport; i++){
		  Xp.set_row(i, X.get_row(port(i)));
		}
		sim_port_value = Xp * sim_value;
	  }
	}
  }
}

void writer_wrapper(char outFileName[], char outGraphName[],
					int nport, int dc_sign, double tstep, double tstop,
					vector<string>& port_name, ivec& port,
					vec& dc_port_value, mat& sim_port_value)
{
  if (nport == 0){ // XXLiu
    printf("   nport == 0. No output file is generated.\n");
  }

  if (nport >0){
	ofstream outFile;
	outFile.open(outFileName);
#ifdef UCR_EXTERNAL
	outFile.precision(4);
	outFile.setf(std::ios_base::scientific | std::ios_base::showpoint);
#endif
	if (!outFile){
	  cout << "couldn't open " << outFileName << endl;
	  exit(-1);
	}
	cout << "start writing" << endl;
	cout << "       to " << outFileName << endl;
	vec ts;
	form_vec(ts, 0, tstep, tstop);
#ifdef UCR_EXTERNAL
	if (dc_sign == 1){
	  for (int i = 0; i < nport; i++){
	    outFile << port_name[i] << " " << dc_port_value(i) << endl;
	  }
	  outFile << endl;
	}else{
	  for (int j = 0; j < nport; j++){
	    outFile << "NODE: " << port_name[j] << endl;
	    vec pv = sim_port_value.get_row(j);
	    for(int i = 0; i < ts.length(); i++){
		outFile << ts(i) << " " << pv(i) << endl;
		//printf("%.4e %.4e\n", ts(i), pv(i));
	    }
	    outFile << "END: " << port_name[j] << endl;
	  }
	  outFile << endl;
	}
#else
	if (dc_sign == 1){
	  for (int i = 0; i < nport; i++){
		outFile << port_name[i] << "  ";
		outFile << dc_port_value(i) << endl;
	  }
	  outFile << endl;
	}else{
	vec pv = sim_port_value.get_row(0);
        for( int i = 0; i < nport; i++)
        {
	  outFile << endl;
	  outFile <<"Node: "<< port_name[i] << "\t" <<endl;
	  outFile << endl;
	  pv = sim_port_value.get_row(i);
	  for(int j = 0; j < ts.length(); j++){
	    outFile.precision(3);
	    outFile << scientific << " " <<ts(j);
	    outFile.precision(6);
	    outFile << scientific <<  " "<<pv(j)<<endl ;
	  }
	  outFile << "END: " << port_name[i] << endl;
 
        }




	  /*outFile << "Time\t";
	  for (int i = 0; i < nport; i++)
		outFile << port_name[i] << "\t";
	  outFile << endl;
	  for(int i = 0; i < ts.length(); i++){
		vec pv = sim_port_value.get_col(i);
		outFile << ts(i) << "\t";
		for(int j =0; j < nport; j++){
		  outFile << pv(j) << "\t";
		}
		outFile << endl;
	  } */
	}	
#endif
	cout << "** " << outFileName << " dumped" << endl;
	outFile.close();
#ifndef UCR_EXTERNAL
	if (!dc_sign)
	  write_xgraph(outGraphName, ts, sim_port_value, port, port_name);
#endif
  }
}
