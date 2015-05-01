/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: parser.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Ning Mi 
 * 
 *    Functions: header file for parser 
 *
 */


#ifndef __PARSER_H
#define __PARSER_H

#include "circuit.h"
#include "etbr.h"
#include "matrix.h"
#include <itpp/base/vec.h>
#include <itpp/base/smat.h>
#include <itpp/base/mat.h>
#include <itpp/base/math/elem_math.h>

#define READ_BLOCK_SIZE 1000
#define NAME_BLOCK_SIZE 200
#define VALUE_BLOCK_SIZE 40
#define PWL_SIZE 800

using namespace itpp;

/*typedef struct{
  vec time;
  vec value;
} Source;*/

typedef struct{
  char type;
  char* node1;
  char* node2;
  double value;
} subelement;

void psource(wave* waveform, circuit* ckt, Source *VS, int nVS, Source *IS, int nIS);

void parser(const char* filename, double& tstep, double& tstop, int& nIS, int& nVS, int& nL, NodeList* nodePool);

void parser_sub(const char* filename, double& tstep, double& tstop, int& num_subnode, int& nVS, int& nIS, int& nL, NodeList* nodePool);


void stamp(const char* filename, int nL, int nVS, int& nNodes, double& tstep, double& tstop, Source *VS, Source *IS, matrix* G, matrix* C, matrix* B, NodeList* nodePool);

void stamp_sub(const char* filename, int nL, int nIS, int nVS, int& nNodes,
	       double& tstep, double& tstop, Source *VS, Source *IS,
	       matrix* G, matrix* C, matrix* B, NodeList* nodePool,
	       gpuETBR *myGPUetbr);

void stampG(const char* filename, int nL, int nVS, int nNodes, matrix* G, NodeList* nodePool);

void stampC(const char* filename, int nL, int nVS, int nNodes, matrix* C, NodeList* nodePool);

void stampB(const char* filename, int nL, int nIS, int nVS, int nNodes, double tstop,
	    Source *VS, Source *IS, matrix* B, NodeList* nodePool,
	    gpuETBR *myGPUetbr);

void parser_old(const char* filename, circuit* cir, wave* waveform); 
// read node and branch information from file

void printsource(Source* VS, int nVS, Source* IS, int nIS);

void printmatrix(sparse_mat *smatrix);

#endif
