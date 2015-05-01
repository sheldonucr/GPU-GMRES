/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_dd_cmd.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li, Ning Mi
 *
 *    Functions: Main function
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
#include "interp.h"
#include "parser.h"
#include "circuit.h"
#include "namepool.h"
#include "mna.h"
#include "cs.h"
#include "itpp2csparse.h"
#include "etbr.h"
#include "etbr_dd.h"
#include "metis.h"

using namespace itpp;
using namespace std;

int main(int argc, char* argv[]){
  
    if (argc > 6){
        cout << "usage: etbr_dd_cmd circuit_name [-nq reduced_order] [-np partition_number]\n";
		exit(-1);
	}
  
	Real_Timer total_run_time, parser_run_time, partition_run_time, etbr_run_time, write_run_time;
	CPU_Timer total_cpu_time, parser_cpu_time, partition_cpu_time, etbr_cpu_time, write_cpu_time;
	total_run_time.start();
	total_cpu_time.start();
	parser_run_time.start();
	parser_cpu_time.start();
	int q = 0;
	int npart = 0;
    if (argc == 2){
	  q = 10;
	  npart = 1;
	}else if (argc == 4){
	  if (strcmp(argv[2],"-nq") == 0){
		q = atoi(argv[3]);
		npart = 1;
	  }else if (strcmp(argv[2],"-np") == 0){
		q = 10;
		npart = atoi(argv[3]);
	  }
	}else if (argc == 6){
	  if (strcmp(argv[2],"-nq") == 0 && strcmp(argv[4],"-np") == 0){
		q = atoi(argv[3]);
		npart = atoi(argv[5]);
	  }else if (strcmp(argv[2],"-np") == 0 && strcmp(argv[4],"-nq") == 0){
		q = atoi(argv[5]);
		npart = atoi(argv[3]);
	  }
	}else{
      cout << "usage: etbr_dd_cmd circuit_name [-nq reduced_order] [-np partition_number]\n";
	  exit(-1);
	}
	if (npart <= 1){
	  cout << "np value should be larger than 1\n" ;
	  exit(-1);
	}

	int thread_version = 0;

	char cktname[100];
	strcpy(cktname, argv[1]);

	Source* VS, * IS;
	NodeList *nodePool;
	int nVS,nIS,nport,nNodes,nL;
	double tstep = 0, tstop = 0;
	int dc_sign = 0;
	ivec port;
	vector<string> port_name;
	matrix* G, *C, *B; 
	int size_G, size_C, row_B, col_B;
	NodeList::Node* node_tmp;

	if((nodePool = new NodeList)==NULL){
	  printf("Out of memory!\n"); exit(1);
	}

	printf("Parser ...\n");
	parser(cktname, tstep, tstop, nVS, nIS, nL, nodePool);
	// nodePool->map_clear();
	printf("get port information.\n");
	/* get port information */
	nport = nodePool->numPort();
	port.set_size(nport);
	for (int i = 0;i < nport; i++){
	  node_tmp = nodePool->getPort(i);
	  string pn = nodePool->getPortName(i);
	  port(i) = node_tmp->row_no;
	  port_name.push_back(pn);
	}
	if (tstep == 0 && tstop == 0){
	  dc_sign = 1;
	}

	/* initialize G,C,B,VS,IS */
	nNodes = nodePool->numNode();
	printf("node number: %d\n", nNodes);
	printf("voltage source number: %d\n", nVS);
	printf("current source number: %d\n", nIS);
	size_G = nNodes + nL + nVS;
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
  
	printf("stamp circuit...\n");
	cs_dl* Gs, *Cs, *Bs;
	stampB(cktname, nL, nVS, nNodes, tstop, VS, IS, B, nodePool);
	Bs = B->mat2csdl();
	//B->matrix::~matrix();
	delete B;
	printf("B cs done.\n");
	stampC(cktname, nL, nVS, nNodes, C, nodePool);
	Cs = C->mat2csdl();
	delete C;
	printf("C cs done.\n");
	stampG(cktname, nL, nVS, nNodes, G, nodePool);
	Gs = G->mat2csdl();
	delete G;
	printf("G cs done.\n");
	printf("Finish stamp.\n");
	
	//nodePool->NodeList::~NodeList();
	delete nodePool;

	parser_run_time.stop();
	parser_cpu_time.stop();
	
	/* ETBR */
	cout << "**** ETBR starts ****" << endl;
	mat Gr, Cr, Br, X, sim_value;
	mat Xp, sim_port_value;
	vec dc_value, dc_port_value;
	if (npart > 1){
	  etbr_run_time.start();
	  etbr_cpu_time.start();
	  partition_run_time.start();
	  partition_cpu_time.start();
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
	  idxtype *xadj = (idxtype *) malloc((m+1)*sizeof(idxtype));
	  idxtype *adjncy = (idxtype *) malloc(nzmax*sizeof(idxtype));
	  idxtype adjncy_index = 0;
	  set<UF_long> adjncy_set;
	  for (j = 0; j < m; j++){
		xadj[j] = adjncy_index;
		for (p = Gp[j]; p < Gp[j+1]; p++){
		  adjncy_set.insert(Gi[p]);
		}
		if (Cs != NULL){
		  for (p = Cp[j]; p < Cp[j+1]; p++){
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
	  xadj[m] = adjncy_index;
	  adjncy = (idxtype *) realloc(adjncy, adjncy_index*sizeof(idxtype));	  
	  ofstream out_GC_file;
	  string GC_file_name = "temp/GC_file";
	  out_GC_file.open(GC_file_name.c_str(), ios::binary);
	  cs_dl_save(out_GC_file, Gs);
	  cs_dl_spfree(Gs);
	  if (Cs != NULL){
		cs_dl_save(out_GC_file, Cs);
		cs_dl_spfree(Cs);
	  }
	  out_GC_file.close();
	  UF_long *node_part = new UF_long[m];
	  UF_long *part_size = new UF_long[npart+1];
	  UF_long *mat_pinv = new UF_long[m];
	  UF_long *mat_q = new UF_long[m];
	  partition4(xadj, adjncy, npart, m, part_size, node_part, mat_pinv, mat_q);
	  partition_cpu_time.stop();
	  partition_run_time.stop();
	  cout << "**** Partition results ****" << endl;
	  UF_long sum = 0;
	  for (int i = 0; i < npart+1; i++){
		cout << "part_size[" << i << "] = " << part_size[i] << endl;
		sum += part_size[i];
	  }
	  cout << "sum of partitions = " << sum << endl;
	  cout << "**** ***************** ****" << endl;
	  if (dc_sign == 1){
		ifstream in_GC_file;
		in_GC_file.open(GC_file_name.c_str(), ios::binary);
		cs_dl_load(in_GC_file, Gs);
		in_GC_file.close();
		dc_dd_solver(Gs, Bs, VS, nVS, IS, nIS, dc_value, 
					 npart, part_size, node_part, mat_pinv, mat_q);
		delete [] VS;
		delete [] IS;
		delete [] node_part;
		delete [] part_size;
		delete [] mat_pinv;
		delete [] mat_q;
		cs_dl_spfree(Bs);
		
		dc_port_value.set_size(nport);
		if (nport > 0){
		  for(int i = 0; i < nport; i++){
			dc_port_value(i) = dc_value.get(port(i));
		  }
		}
	  }else{
		/*
		  etbr_dd(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
		  Gr, Cr, Br, X, sim_value, npart, part_size, node_part, mat_pinv, mat_q);
		*/
		etbr_dd(Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
				Gr, Cr, Br, X, sim_value, npart, part_size, node_part, mat_pinv, mat_q);

		delete [] VS;
		delete [] IS;
		delete [] node_part;
		delete [] part_size;
		delete [] mat_pinv;
		delete [] mat_q;

		// cs_dl_spfree(Gs);
		// cs_dl_spfree(Cs);
		cs_dl_spfree(Bs);
	
		Xp.set_size(nport, q);
		sim_port_value.set_size(nport, sim_value.cols());
		if (nport > 0){
		  for(int i = 0; i < nport; i++){
			Xp.set_row(i, X.get_row(port(i)));
		  }
		  sim_port_value = Xp * sim_value;
		}
	  }
	  etbr_run_time.stop();
	  etbr_cpu_time.stop();
	}
	cout << "**** ETBR ends ****" << endl;

	/* Write simulation value */
	write_run_time.start();
	write_cpu_time.start();

	if (nport >0){
	  char outFileName[100];
	  strcpy(outFileName, cktname);
	  strcat(outFileName, ".output");
	  ofstream outFile;
	  outFile.open(outFileName);
	  if (!outFile){
		cout << "couldn't open " << outFileName << endl;
		exit(-1);
	  }
	  // outFile.setf(std::ios::fixed,std::ios::floatfield); 
	  // outFile.precision(10);
	  if (dc_sign == 1){
		for (int i = 0; i < nport; i++)
		  outFile << port_name[i] << "\t";
		outFile << endl;
		for (int i = 0; i < nport; i++)
		  outFile << dc_port_value(i) << "\t";
		outFile << endl;
	  }else{
		outFile << "Time\t";
		for (int i = 0; i < nport; i++)
		  outFile << port_name[i] << "\t";
		outFile << endl;
		vec ts;
		form_vec(ts, 0, tstep, tstop);
		for(int i = 0; i < ts.length(); i++){
		  vec pv = sim_port_value.get_col(i);
		  outFile << ts(i) << "\t";
		  for(int j =0; j < nport; j++){
			outFile << pv(j) << "\t";
		  }
		  outFile << endl;
		}
	  }
	  cout << "** " << outFileName << " dumped" << endl;
	  outFile.close();	
	}
	write_run_time.stop();
	write_cpu_time.stop();
	total_run_time.stop();
	total_cpu_time.stop();

	cout << endl;
	cout << "****** Runtime Statistics (seconds) ******  " << endl;
	std::cout.setf(std::ios::fixed,std::ios::floatfield); 
	std::cout.precision(2);
	cout << "parse      \t: " << parser_run_time.get_time() << " (CPU: " << parser_cpu_time.get_time() << ")" << endl;
	cout << "partition  \t: " << partition_run_time.get_time() << " (CPU: " << partition_cpu_time.get_time() << ")" << endl;
	cout << "etbr       \t: " << etbr_run_time.get_time() << " (CPU: " << etbr_cpu_time.get_time() << ")" << endl;
	cout << "write      \t: " << write_run_time.get_time() << " (CPU: " << write_cpu_time.get_time() << ")" << endl;
	cout << "total      \t: " << total_run_time.get_time() << " (CPU: " << total_cpu_time.get_time() << ")" << endl;
}
