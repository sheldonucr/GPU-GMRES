/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_cmd_short.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li, Ning Mi
 *             Xue-Xin Liu, Zao Liu
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
#include "etbr_wrapper.h"
#include "metis.h"

#include "gpuData.h"

using namespace itpp;
using namespace std;

extern long interp1_sum;
extern long interp2_sum;

#define ETBR_VER "2.0"  /* ETBR version */
#define DEFAULT_R_ORDER  20	/* default reduction order */
#define DEFAULT_IR_PERCENTAGE 0.05	 /* default allowed IR drop percentage */
	
void help_message();
/* for release version */
void help_message_rel();
void banner();

int main(int argc, char* argv[]){
  
    if (argc == 1){
		//help_message();
		help_message_rel();
        //cout << "Usage: etbr_cmd circuit_name [-nq reduced_order] [-np partition_number] [-th threshold] [-ir] [-cd]\n";
		exit(-1);
	}
  
	Real_Timer total_run_time, parser_run_time, partition_run_time, etbr_run_time, write_run_time;
	CPU_Timer total_cpu_time, parser_cpu_time, partition_cpu_time, etbr_cpu_time, write_cpu_time;
	Real_Timer ir_run_time, simu_run_time;
	CPU_Timer ir_cpu_time, simu_cpu_time;
	total_run_time.start();
	total_cpu_time.start();
	
	int q = DEFAULT_R_ORDER;
	
	int npart = 1;
	int ir_info = 0;
	int use_gpu = 0, use_cuda_double=1, use_cuda_single=0;
	int cd_info = 0;
	int thread_version = 0;
	int mna_version = 1;
	int etbr_version = 0;
	int error_control = 0;
        int use_gmres = 0, use_iluPackage = 0;
	// error percentgae allowed
	double threshold_percentage = DEFAULT_IR_PERCENTAGE;	

	for (int i = 2; i < argc;){
	  if (strcmp(argv[i],"-fast") == 0){
	    etbr_version = 1;
	    mna_version = 0;
	    i++;
	  }else if (strcmp(argv[i],"-nq") == 0){
	    if (!etbr_version) {
	      cout << "Error: missing -fast option" << endl;
	      exit(-1);
	    }
	    q = atoi(argv[i+1]);
	    i += 2;
	  }else if (strcmp(argv[i],"-np") == 0){
	    if (!etbr_version) {
	      cout << "Error: missing -fast option" << endl;
	      exit(-1);
	    }
	    npart = atoi(argv[i+1]);
	    i += 2;
	  }else if (strcmp(argv[i],"-ec") == 0){
	    if (!etbr_version) {
	      cout << "Error: missing -fast option" << endl;
	      exit(-1);
	    }
	    error_control = 1;
	    i++;
	  }else if (strcmp(argv[i],"-th") == 0){
	    if (!etbr_version) {
	      cout << "Error: missing -fast option" << endl;
	      exit(-1);
	    }
	    threshold_percentage = atof(argv[i+1]);
	    i += 2;
	  }else if (strcmp(argv[i],"-mt") == 0){
	    if (!etbr_version) {
	      cout << "Error: missing -fast option" << endl;
	      exit(-1);
	    }
	    thread_version = 1;
	    i++;
	  }else if (strcmp(argv[i],"-ir") == 0){
	    ir_info = 1;
	    i++;	  
	  }
	  else if (strcmp(argv[i],"-gpu") == 0){
	    use_gpu = 1;
	    i++;
	  }
	  else if (strcmp(argv[i],"-double") == 0){
	    use_cuda_single = 0;
	    use_cuda_double = 1;
	    i++;
	  }
	  else if (strcmp(argv[i],"-single") == 0){
	    use_cuda_single = 1;
	    use_cuda_double = 0;
	    i++;
	  }
	  else if (strcmp(argv[i],"-cd") == 0){
	    cd_info = 1;
	    i++;
	  }
          else if(strcmp(argv[i],"-gmres") == 0){
            use_gmres = 1;
            i++;
          }
          else if(strcmp(argv[i],"-ilu") == 0){
            use_iluPackage = 1;
            i++;
          }
	  else{
	    //help_message();
	    help_message_rel();
		  
	    exit(-1);
	  }
	}
	
	// print the banner
	banner();
	
	// print some parameter used for etbr
	
	//cout <<"****  ETBR parameter setting **** " << endl;
	
	//cout <<"The reduction order = " << q << endl;	
	
	//cout <<"The allowed IR drop error in percentage = " << threshold_percentage << endl;
	
	//cout << endl;
	
	int display_ir_num = 20;

	char cktname[100];
	char cktname_cd[100];

	strcpy(cktname, argv[1]);
	if(cd_info){
	  char *dir = strrchr(argv[1], '/');
	  strcpy(cktname_cd, dir+1);
	}else{
	  strcpy(cktname_cd, argv[1]);
	}

	Source* VS, * IS;
	int nVS, nIS;
	cs_dl* Gs, *Cs, *Bs;
	double tstep = 0, tstop = 0;
	int dc_sign = 0;
	int nport = 0;
	int nNodes = 0;	
	vector<string> port_name;
	ivec port;
	vector<int> tc_node; 
	vector<string> tc_name;

	/* Parser */
	parser_run_time.start();
	parser_cpu_time.start();

	gpuETBR myGPUetbr; // XXLiu
        gpuRelatedDataInit(&myGPUetbr);
        myGPUetbr.q = q;
	myGPUetbr.PWLcurExist = 0;  myGPUetbr.PULSEcurExist = 0; // XXLiu
	myGPUetbr.PWLvolExist = 0;  myGPUetbr.PULSEvolExist = 0; // XXLiu
	myGPUetbr.use_cuda_single = use_cuda_single;
	myGPUetbr.use_cuda_double = use_cuda_double; // XXLiu

	parser_wrapper(cktname, 
		       Gs, Cs, Bs, 
		       VS, nVS, IS, nIS,
		       nNodes, nport, tstep, tstop,
		       dc_sign,
		       port_name, port,
		       tc_node, tc_name,
		       &myGPUetbr);

	parser_run_time.stop();
	parser_cpu_time.stop();
		
	mat Gr, Cr, Br, X, sim_value;
	mat Xp, sim_port_value;
	vec dc_value, dc_port_value;
	vec max_value, min_value, ir_value;
	double max_ir, min_ir, avg_ir;
	int max_ir_idx, min_ir_idx;
	ivec sorted_ir_value_idx;
	
	char ir_name[100];
	if (cd_info){
	  strcpy(ir_name, cktname_cd);
	}else{
	  strcpy(ir_name, cktname);
	}
	strcat(ir_name, ".ir");

        /* XXLiu: GMRES sparse solver on circuit MNA equation. */
	if (mna_version){
	  if (dc_sign == 1){ 
            cout << "dc_sign = 1, solve dc"<<endl;
	    etbr_dc_wrapper(Gs, Bs, VS, nVS, IS, nIS, nport, port, dc_value, dc_port_value);
	  }else{
	    simu_run_time.start();
	    simu_cpu_time.start();
            
            if(use_gmres) /* XXLiu: Iterative solvers will be used. */
              if(use_gpu) // -gmres -gpu -single
                mna_solve_gpu_gmres(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
                                    port, sim_port_value, tc_node, tc_name,
                                    display_ir_num, ir_info, ir_name, &myGPUetbr);
              else if(use_iluPackage) // -gmres -ilu
                mna_solve_cpu_ilu_gmres(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
                                        port, sim_port_value, tc_node, tc_name,
                                        display_ir_num, ir_info, ir_name);
              else // -gmres
                mna_solve_cpu_gmres(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
                                    port, sim_port_value, tc_node, tc_name,
                                    display_ir_num, ir_info, ir_name, &myGPUetbr);
                
            else if(use_gpu) // -gpu
              mna_solve_gpu(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
                            port, sim_port_value, tc_node, tc_name,
                            display_ir_num, ir_info, ir_name, &myGPUetbr);
            else
              mna_solve(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, 
                        port, sim_port_value, tc_node, tc_name,
                        display_ir_num, ir_info, ir_name);

	    simu_run_time.stop();
	    simu_cpu_time.stop();
	  }
	  etbr_version = 0;
	}
        
	if (etbr_version && npart == 1){
          /* ETBR */
          cout << "**** starting ETBR ****" << endl;
	  etbr_run_time.start();
	  etbr_cpu_time.start();

	  if (dc_sign == 1){  
	    etbr_dc_wrapper(Gs, Bs, VS, nVS, IS, nIS, nport, port, dc_value, dc_port_value);
	  }else{

	    double max_i = 0;
	    int max_i_idx = 0;

	    cout << "**** starting reduction ****" << endl;
	    cout << "# reduced order: " << q << endl;
	    if (thread_version){
	      if(use_gpu)
	      	gpu_etbr_thread(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
	      			Gr, Cr, Br, X, max_i, max_i_idx, &myGPUetbr);
	      else
		etbr2_thread(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
			     Gr, Cr, Br, X, max_i, max_i_idx);
	    }else{
	      etbr2(Gs, Cs, Bs, VS, nVS, IS, nIS, tstep, tstop, q, 
		    Gr, Cr, Br, X, max_i, max_i_idx);
	    }
	    cout << "**** reduction complete ****" << endl;

	    etbr_run_time.stop();
	    etbr_cpu_time.stop();
            cout << "**** ETBR complete ****" << endl;

	    simu_run_time.start();
	    simu_cpu_time.start();

	    std::cout.precision(10);
	    //cout << "max_i = " << max_i << endl;
		
	    cout << "**** starting simulation  ****" << endl;

	    if(!error_control){
	      if(use_gpu)
		gpu_transim(Gs, Cs, Bs, VS, nVS, IS, nIS, 
			    tstep, tstop, q, max_i, max_i_idx, threshold_percentage,
			    Gr, Cr, Br, X, sim_port_value,
			    port, tc_node, tc_name, 
			    display_ir_num, ir_info, ir_name,
			    &myGPUetbr);
	      else
		reduced_transim2(Gs, Cs, Bs, VS, nVS, IS, nIS, 
				 tstep, tstop, q, max_i, max_i_idx, threshold_percentage,
				 Gr, Cr, Br, X, sim_port_value,
				 port, tc_node, tc_name, 
				 display_ir_num, ir_info, ir_name);
	    }else{
	      mixed_transim2(Gs, Cs, Bs, VS, nVS, IS, nIS, 
			     tstep, tstop, q, max_i, max_i_idx, threshold_percentage,
			     Gr, Cr, Br, X, sim_port_value,
			     port, tc_node, tc_name, 
			     display_ir_num, ir_info, ir_name);
	    }
		
	    cout << "**** simulation complete ****" << endl;
	    simu_run_time.stop();
	    simu_cpu_time.stop();
		
	  }
	}else if (etbr_version && npart > 1){
	  etbr_run_time.start();
	  etbr_cpu_time.start();
	  UF_long m = Gs->m;
	  UF_long *node_part = new UF_long[m];
	  UF_long *part_size = new UF_long[npart+1];
	  UF_long *mat_pinv = new UF_long[m];
	  UF_long *mat_q = new UF_long[m];	
	  partition_run_time.start();
	  partition_cpu_time.start();
	  string GC_file_name = "temp/GC_file";

	  partition_wrapper(GC_file_name, Gs, Cs, nNodes, npart,
						part_size, node_part, mat_pinv, mat_q);
	  partition_cpu_time.stop();
	  partition_run_time.stop();
	  
	  etbr_dd_wrapper(GC_file_name, Gs, Cs, Bs, 
					  VS, nVS, IS,  nIS, 
					  dc_value, dc_port_value, dc_sign,
					  npart, nport, port, q, tstep, tstop,
					  part_size, node_part, mat_pinv, mat_q,
					  Gr, Cr, Br, X, sim_value,
					  Xp, sim_port_value);

	  delete [] node_part;
	  delete [] part_size;
	  delete [] mat_pinv;
	  delete [] mat_q;
	  etbr_run_time.stop();
	  etbr_cpu_time.stop();
	}
       
        if (dc_sign != 1){  
	delete [] VS;
	delete [] IS;
        
	cs_dl_spfree(Gs);
	cs_dl_spfree(Cs);
	cs_dl_spfree(Bs);
        }

	/* Write simulation value */
	write_run_time.start();
	write_cpu_time.start();

	char outFileName[100];
	char outGraphName[100];

	if (cd_info){
	  strcpy(outFileName, cktname_cd);
	  strcpy(outGraphName, cktname_cd);
	}else{
	  strcpy(outFileName, cktname);
	  strcpy(outGraphName, cktname);
	}
	strcat(outFileName, ".output");
	strcat(outGraphName, ".xgraph");

	writer_wrapper(outFileName, outGraphName,
				   nport, dc_sign, tstep, tstop,
				   port_name, port,
				   dc_port_value, sim_port_value);

	write_run_time.stop();
	write_cpu_time.stop();
	cout << "**** " << endl;

	total_run_time.stop();
	total_cpu_time.stop();
	cout << "****** Runtime Statistics (seconds) ******  " << endl;
	std::cout.setf(std::ios::fixed,std::ios::floatfield); 
	std::cout.precision(2);
	//cout << "interp1_sum = " << interp1_sum << endl;
	//cout << "interp2_sum = " << interp2_sum << endl;
	cout << "parse      \t: " << parser_run_time.get_time() << " (CPU: " << parser_cpu_time.get_time() << ")" << endl;	
	if (npart > 1)
	  cout << "partition  \t: " << partition_run_time.get_time() << " (CPU: " << partition_cpu_time.get_time() << ")" << endl;
        if(etbr_version)
          cout << "reduction  \t: " << etbr_run_time.get_time() << " (CPU: " << etbr_cpu_time.get_time() << ")" << endl;
	cout << "simulation \t: " << simu_run_time.get_time() << " (CPU: " << simu_cpu_time.get_time() << ")" << endl;
	cout << "write      \t: " << write_run_time.get_time() << " (CPU: " << write_cpu_time.get_time() << ")" << endl;
	if (ir_info)
	  cout << "IR analysis\t: " << ir_run_time.get_time() << " (CPU: " << ir_cpu_time.get_time() << ")" << endl;
	cout << "total      \t: " << total_run_time.get_time() << " (CPU: " << total_cpu_time.get_time() << ")" << endl;

        gpuRelatedDataFree(&myGPUetbr);
}

void banner()
{
	cout <<"\n";
	printf("######################################################\n");
	printf("GPU ETBR - Extended TBR based power grid analysis tool\n");
	printf("Version: %s, developed by MSLAB (www.mscad.ee.ucr.edu) at UCRiveside.\n", ETBR_VER);
        printf("Author: Duo Li, Xue-Xin Liu, Zao Liu\n");
        printf("    In this update, iterative solvers are added for DC and transient analysis.\n");
	printf("Copyright 2012 Regents of the University of California.  All rights reserved.\n");
	printf("######################################################\n");
	
}

void help_message()
{		
	
	printf("Usage: etbr_cmd circuit_name [ -fast [-nq reduced_order] [-np partition_number] [-ec] [-th threshold] [-mt] ] [-ir] [-gpu] [-cd]\n");
	//cout << "******************* general options ***********************\n";
	printf("  [-fast -- fast version using reduction method]\n");
	printf("  [-nq <int> -- reduced order, default: %d]\n", (int)DEFAULT_R_ORDER);
	printf("  [-np <int> -- partition_number]\n");
	printf("  [-ec -- use dynamic error control technique]\n");
	printf("  [-th <double> -- allowed IR drop error in percentage (wrt the lartgest IR drop), default: %g]\n", (float)DEFAULT_IR_PERCENTAGE);
	printf("  [-mt -- use multi-threading simulation]\n");		
	printf("  [-ir -- perform IR drop analysis and print out 20 nodes with largest IR drops]\n");
	printf("  [-gpu -- GPU acceleration]\n");
	printf("  [-cd -- dump the output files into current directory]\n");

	cout <<"\n";
}

void help_message_rel()
{		
	
	printf("Usage: etbr_cmd circuit_name "
               "[ -fast [-nq reduced_order] [-np partition_number] [-mt] ] "
               "[-gpu -single|-double] [-cd]\n");
	//cout << "******************* general options ***********************\n";
	printf("  [-fast -- fast version using reduction method]\n");
	printf("  [-nq <int> -- reduced order, default: %d]\n", (int)DEFAULT_R_ORDER);
	printf("  [-np <int> -- partition_number]\n");
	printf("  [-mt -- use multi-threading simulation]\n");
	printf("  [-gpu -- GPU acceleration]\n");
	printf("  [-single|-double -- GPU float point precision]\n");
	printf("  [-cd -- dump the output files into current directory]\n");

	cout <<"\n";
}
