/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: mna_cmd.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:59 $
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

using namespace itpp;
using namespace std;

int main(int argc, char* argv[]){
  gpuETBR myGPUetbr; // XXLiu
  myGPUetbr.PWLcurExist = 0;  myGPUetbr.PULSEcurExist = 0; // XXLiu
  myGPUetbr.PWLvolExist = 0;  myGPUetbr.PULSEvolExist = 0; // XXLiu

    if (argc == 1 || argc > 3){
        cout << "usage: mna_cmd circuit_name [-ir]\n";
		exit(-1);
	}
  
	Real_Timer total_run_time, parser_run_time, partition_run_time, mna_run_time, write_run_time;
	CPU_Timer total_cpu_time, parser_cpu_time, partition_cpu_time, mna_cpu_time, write_cpu_time;
	Real_Timer ir_run_time;
	CPU_Timer ir_cpu_time;
	total_run_time.start();
	total_cpu_time.start();
	parser_run_time.start();
	parser_cpu_time.start();
	
	char cktname[100];
	strcpy(cktname, argv[1]);

	int ir_info = 0;
	for (int i = 2; i < argc;){
	  if (strcmp(argv[i],"-ir") == 0){
		ir_info = 1;
		i++;
	  }else{
        cout << "usage: etbr_cmd circuit_name [-ir]\n";
		exit(-1);
	  }
	}

	int display_ir_num = 20;

	Source* VS, * IS;
	NodeList *nodePool;
	int nVS,nIS,nport,nNodes,nL,nsubnode;
	double tstep = 0, tstop = 0;
	int dc_sign = 0;
	ivec port;
	vector<string> port_name;
	vector<int> tc_node;
	vector<string> tc_name;
	matrix* G, *C, *B; 
	int size_G, size_C, row_B, col_B;
	NodeList::Node* node_tmp;

	if((nodePool = new NodeList)==NULL){
	  printf("Out of memory!\n"); exit(1);
	}
	
	printf("Parser ...\n");
	nsubnode = 0;
	//parser(cktname, tstep, tstop, nVS, nIS, nL, nodePool);
	parser_sub(cktname, tstep, tstop, nsubnode, nVS, nIS, nL, nodePool);
	// nodePool->map_clear();
	printf("get port information.\n");
	/* get port information */
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
	if (tstep == 0 && tstop == 0){
	  dc_sign = 1;
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
  
	printf("stamp circuit...\n");
	cs_dl* Gs, *Cs, *Bs;
	if(nsubnode != 0){
	  stamp_sub(cktname, nL, nIS, nVS, nNodes, tstep, tstop, VS, IS, G,  C,  B, nodePool,
		    &myGPUetbr); // XXLiu
	  Bs = B->mat2csdl();
	  delete B;
	  printf("B cs done.\n");
	  Cs = C->mat2csdl();
	  printf("C cs done.\n");
	  delete C;
	  Gs = G->mat2csdl();
	  delete G;
	  printf("G cs done.\n");
	}
	else{
	  stampB(cktname, nL, nIS, nVS, nNodes, tstop, VS, IS, B, nodePool, &myGPUetbr);
	  Bs = B->mat2csdl();
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
	}
	delete nodePool;

	parser_run_time.stop();
	parser_cpu_time.stop();
		
	/* MNA solver */
	cout << "**** MNA solver starts ****" << endl;
	mat sim_port_value;	
	mna_run_time.start();
	mna_cpu_time.start();
	char ir_name[100];
	strcpy(ir_name, cktname);
	strcat(ir_name, ".ir.mna");
	mna_solve(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
			  port, sim_port_value, tc_node, tc_name, display_ir_num, ir_info, ir_name);
	delete [] VS;
	delete [] IS;
	cs_dl_spfree(Gs);
	cs_dl_spfree(Cs);
	cs_dl_spfree(Bs);
	mna_run_time.stop();
	mna_cpu_time.stop();
	cout << "**** MNA solver ends ****" << endl;

	/* Write simulation value */
	write_run_time.start();
	write_cpu_time.start();

	if (nport >0){
	  char outFileName[100];
	  strcpy(outFileName, cktname);
	  strcat(outFileName, ".output.mna");
	  ofstream outFile;
	  outFile.open(outFileName);
	  if (!outFile){
		cout << "couldn't open " << outFileName << endl;
		exit(-1);
	  }
	  // outFile.setf(std::ios::fixed,std::ios::floatfield); 
	  // outFile.precision(10);
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
	  cout << "** " << outFileName << " dumped" << endl;
	  outFile.close();	
	  char outGraphName[100];
	  strcpy(outGraphName, cktname);
	  strcat(outGraphName, ".xgraph.mna");
	  write_xgraph(outGraphName, ts, sim_port_value, port, port_name);
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
	cout << "MNA        \t: " << mna_run_time.get_time() << " (CPU: " << mna_cpu_time.get_time() << ")" << endl;
	cout << "write      \t: " << write_run_time.get_time() << " (CPU: " << write_cpu_time.get_time() << ")" << endl;
	// cout << "IR analysis\t: " << ir_run_time.get_time() << " (CPU: " << ir_cpu_time.get_time() << ")" << endl;
	cout << "total      \t: " << total_run_time.get_time() << " (CPU: " << total_cpu_time.get_time() << ")" << endl;
}
