/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR header
 *
 */

#ifndef ETBR_H
#define ETBR_H


#include <itpp/base/smat.h>
#include <itpp/base/mat.h>
#include <vector>
#include "cs.h"
#include "gpuData.h"

using namespace itpp;
using namespace std;

typedef struct{
  vec time;
  vec value;
} Source;

typedef struct{
  vec samples;
  cs_dl *G;
  cs_dl *C;
  cs_dl *B;
  mat *us;
  vec *zvec;
}AXBDATA;

// #define NUM_THREADS 20
// pthread_t threads[NUM_THREADS];
//pthread_mutex_t mutexz;



void etbr(sparse_mat &G, sparse_mat &C, sparse_mat &B, 
		  Source *VS, int nVS, Source *IS, int nIS, 
		  double tstep, double tstop, int q, 
		  mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value);

void etbr(cs_dl *G, cs_dl *C, cs_dl *B, 
		  Source *VS, int nVS, Source *IS, int nIS, 
		  double tstep, double tstop, int q, 
		  mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value);

void etbr2(cs_dl *G, cs_dl *C, cs_dl *B, 
		   Source *VS, int nVS, Source *IS, int nIS, 
		   double tstep, double tstop, int q, 
		   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value);

void etbr2(cs_dl *G, cs_dl *C, cs_dl *B, 
		   Source *VS, int nVS, Source *IS, int nIS, 
		   double tstep, double tstop, int q, 
		   mat &Gr, mat &Cr, mat &Br, mat &X, double &max_i, int &max_i_idx);


void etbr2(cs_dl *G, cs_dl *C, cs_dl *B,
			Source *VS, int nVS, Source *IS, int nIS,
			double tstep, double tstop, int q,
			mat &Gr, mat &Cr, mat &Br, mat &X, double &max_i,
			vec *u_col);

void etbr_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
				 Source *VS, int nVS, Source *IS, int nIS, 
				 double tstep, double tstop, int q, 
				 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_value);

void etbr2_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
				 Source *VS, int nVS, Source *IS, int nIS, 
				 double tstep, double tstop, int q, 
				 mat &Gr, mat &Cr, mat &Br, mat &X,
				 double &max_i, int &max_i_idx);

void gpu_etbr_thread(cs_dl *G, cs_dl *C, cs_dl *B, 
		     Source *VS, int nVS, Source *IS, int nIS, 
		     double tstep, double tstop, int q, 
		     mat &Gr, mat &Cr, mat &Br, mat &X,
		     double &max_i, int &max_i_idx, gpuETBR *myGPUetbr);

void dc_solver(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value);

void dc_solver2(cs_dl *G, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, vec &dc_value);

void mna_solve(cs_dl *G, cs_dl *C, cs_dl *B, 
			   Source *VS, int nVS, Source *IS, int nIS, 
			   double tstep, double tstop, const ivec &port, 
			   mat &sim_port_value);

void mna_solve(cs_dl *G, cs_dl *C, cs_dl *B, 
			   Source *VS, int nVS, Source *IS, int nIS, 
			   double tstep, double tstop, const ivec &port, mat &sim_port_value, 
			   vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info, char *ir_name);

void ir_analysis(int display_num, vector<int> &tc_node,
				 vector<string> &tc_name, mat &X, mat &sim_value, char *ir_name);

void write_xgraph(char *outGraphName, vec &ts, mat &sim_port_value, 
				  ivec &port, vector<string> &port_name);

void mixed_transim(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
				   double tstep, double tstop, int q, double max_i, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name);

void mixed_transim2(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
				   double tstep, double tstop, int q, double max_i, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name);

void mixed_transim2(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
					double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name);
void reduced_transim2(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
					double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name);

// XXLiu
// #ifdef __cplusplus
// extern "C"
// #endif

void gpu_transim(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
		 double tstep, double tstop, int q, double max_i, int max_i_idx, double threshold_percentage,
		 mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
		 const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
		 int num, int ir_info, char *ir_name,
		 gpuETBR *myGPUetbr);

void mna_solve_gpu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num,
                         int ir_info, char *ir_name, gpuETBR *myGPUetbr);

void mna_solve_cpu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num,
                         int ir_info, char *ir_name, gpuETBR *myGPUetbr);

void mna_solve_cpu_ilu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num,
                         int ir_info, char *ir_name);//, gpuETBR *myGPUetbr

void mna_solve_gpu(cs_dl *G, cs_dl *C, cs_dl *B, 
                   Source *VS, int nVS, Source *IS, int nIS, 
                   double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                   vector<int> &tc_node, vector<string> &tc_name, int num,
                   int ir_info, char *ir_name, gpuETBR *myGPUetbr);

// // XXLiu
// #ifdef __cplusplus
// extern "C"
// #endif
// void cudaTranSim(gpuETBR *myGPUetbr);


void mixed_transim3(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
				   double tstep, double tstop, int q, double max_i, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name);

void mixed_transim(cs_dl *G, cs_dl *C, cs_dl *B, Source *VS, int nVS, Source *IS, int nIS, 
				   double tstep, double tstop, int q, double max_i, double threshold_percentage,
				   mat &Gr, mat &Cr, mat &Br, mat &X, mat &sim_port_value,
				   const ivec &port, vector<int> &tc_node, vector<string> &tc_name, 
				   int num, int ir_info, char *ir_name, vec *u_col);

void multiply(mat& a, vec& x, vec& b);

void multiply(mat& a, double* x, double* b);

#endif
