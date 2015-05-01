/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#ifndef __SPMV__INSPECT_H__
#define __SPMV__INSPECT_H__

struct indStruct {
    unsigned int entry;
    unsigned int pos;
};

typedef struct indStruct index_ins;

struct rowIndStruct {
    unsigned int nnz_row;
    unsigned int rownum;
};

typedef struct rowIndStruct rowIndex_ins;

int inspectBlock(SpMatrix *m, unsigned int **rowIndices, unsigned int **indices,
                  unsigned int *numblocks, unsigned int **nnzCount_block, unsigned int **yCount_block,
                  unsigned int bsx, unsigned int bsy);

int inspectVarBlock(SpMatrix *m, float **valFill, unsigned int **indicesFill, unsigned int **rowIndicesFill, unsigned int **rowIndices,
                  unsigned int **indices, unsigned int *numblocks, unsigned int **nnzCount_block, unsigned int **yCount_block,
                  unsigned int *nnz_fill, unsigned int bsx, unsigned int bsy, unsigned int varC);

int inspectInputBlock(SpMatrix *m, unsigned int **inputList, unsigned int **rowIndices, unsigned int **indices,
                  unsigned int *numblocks, unsigned int *inputListCount, unsigned int bsx, unsigned int bsy);

#define ERR_INSUFFICIENT_MEMORY -3

#endif /* __SPMV__INSPECT_H__ */

