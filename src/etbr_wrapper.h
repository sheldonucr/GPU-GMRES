/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: etbr_wrapper.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:56 $
 *    Authors: Duo Li
 *
 *    Functions: ETBR wrapper header
 *
 */

#ifndef ETBR_WRAPPER_H
#define ETBR_WRAPPER_H


#include <itpp/base/smat.h>
#include <itpp/base/mat.h>
#include <vector>
#include <string>
#include "cs.h"
#include "etbr.h"

using namespace itpp;
using namespace std;

void parser_wrapper(char cktname[], 
		    cs_dl*& Gs, cs_dl*& Cs, cs_dl*& Bs, 
		    Source*& VS, int& nVS, Source*& IS, int& nIS,
		    int& nNodes, int& nport, double& tstep, double& tstop,
		    int& dc_sign,
		    vector<string>& port_name, ivec& port,
		    vector<int>& tc_node, vector<string>& tc_name,
		    gpuETBR *myGPUetbr);

void etbr_dc_wrapper(cs_dl* Gs, cs_dl* Bs, 
					 Source* VS, int nVS, Source* IS, int nIS, 
					 int nport, ivec& port,
					 vec& dc_value, vec& dc_port_value);

void partition_wrapper(string& GC_file_name, cs_dl* Gs, cs_dl* Cs, int nNodes, int npart,
					   UF_long* part_size, UF_long* node_part, UF_long* mat_pinv, UF_long* mat_q);

void etbr_dd_wrapper(string& GC_file_name, cs_dl* Gs, cs_dl* Cs, cs_dl* Bs, 
					 Source* VS, int nVS, Source* IS, int nIS, 
					 vec& dc_value, vec& dc_port_value, int dc_sign,
					 int npart, int nport, ivec& port, int q, double tstep, double tstop,
					 UF_long* part_size, UF_long* node_part, UF_long* mat_pinv, UF_long* mat_q,
					 mat& Gr, mat& Cr, mat& Br, mat& X, mat& sim_value,
					 mat& Xp, mat& sim_port_value);

void writer_wrapper(char outFileName[], char outGraphName[],
					int nport, int dc_sign, double tstep, double tstop,
					vector<string>& port_name, ivec& port,
					vec& dc_port_value, mat& sim_port_value);


#endif
