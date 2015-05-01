/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "SpMV.h"
#include "SpMV_inspect.h"

int isPresent(unsigned *arr, unsigned arrsize, unsigned val)
{
    for (unsigned i=0;i<arrsize;i++)
	if (arr[i]==val) return 1;
    return 0;	
}

// Bubble sort: returned is a nondecresing sequence
void sort(unsigned *a, unsigned size) {
  unsigned switched = 1;
  unsigned hold;
  size -= 1;

  for(unsigned i = 0; i < size && switched; i++) {
    switched = 0;
    for(unsigned j = 0; j < size - i; j++)
      if(a[j] > a[j+1]) {
	switched = 1;
	hold = a[j];
	a[j] = a[j + 1];
	a[j + 1] = hold;
      }
  }
}

int inspectBlock(SpMatrix *m, unsigned **rowIndices, unsigned **indices,
                  unsigned *numblocks, unsigned **nnzCount_block, unsigned **yCount_block,
                  unsigned bsx, unsigned bsy)
{
    unsigned nblocks=0;
    unsigned nrows = m->numRows;
    unsigned ncols = m->numCols;
    unsigned *yflag = (unsigned *)malloc(sizeof(int)*bsy);
    unsigned *yBFlag = (unsigned *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
    unsigned *lb = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *ub = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *bptr = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *colFlag = (unsigned *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
    unsigned *nblocksRow = (unsigned *)malloc(sizeof(int)*(int)ceild(nrows,bsx));
    unsigned **indRows = (unsigned **)malloc(sizeof(int*)*(int)ceild(nrows,bsx));

    *rowIndices = (unsigned *)malloc(sizeof(int)*((int)ceild(nrows,bsx)+1));
    if (*rowIndices == NULL) return ERR_INSUFFICIENT_MEMORY;


    // for each block of row
    unsigned it, iti;
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        // start of a row block
        (*rowIndices)[iti]=nblocks;
        nblocksRow[iti]=0;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
	    lb[i-it] = (m->rowPtrs)[i];
            if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1;
            else ub[i-it] = (m->rowPtrs)[i+1]-1;
            bptr[i-it] = lb[i-it];
        }
        // for each block of column within a row block
        for (unsigned jt = 0, jti = 0; jt < ncols; jt += bsy, jti++) {
            colFlag[jti]=0;
            yBFlag[jti]=0;
            for (unsigned k=0;k<bsy;k++)
                yflag[k]=0;
            unsigned blockStart = nblocks;
            for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
                unsigned j = bptr[i-it];
                for (; j <= ub[i-it]; j++) {
                    unsigned cInd = (m->nzentries)[j].colNum;
                    if (cInd >= jt+bsy)
                        break;
                    else {
                        if (!yflag[cInd-jt]) yflag[cInd-jt]++;
                        colFlag[jti]++; //colFlag[jti] stores the #nnzs in the block
                        if (blockStart == nblocks) {
                            nblocks++;
                            nblocksRow[iti]++;
                        }
                    }
                }
                bptr[i-it] = j;
            }
            for (unsigned k=0;k<bsy;k++)
                yBFlag[jti] += yflag[k];
        }
        indRows[iti] = (unsigned *)malloc(sizeof(int)*3*nblocksRow[iti]);
        if (indRows[iti] == NULL) return ERR_INSUFFICIENT_MEMORY;
        for (unsigned k=0, indRowk=0; k < ceild(ncols,bsy); k++) {
            if (colFlag[k]) {
                indRows[iti][3*indRowk]=k;
                indRows[iti][3*indRowk+1]=colFlag[k];
                indRows[iti][3*indRowk+2]=yBFlag[k];
                indRowk++;
            }
	    // indRowk at the end of the for loop will be equal to nblocksRow[iti]
        }

    }
    (*rowIndices)[iti]=nblocks;

    *numblocks = nblocks;
    *indices = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*indices == NULL) return ERR_INSUFFICIENT_MEMORY;
    *nnzCount_block = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*nnzCount_block == NULL) return ERR_INSUFFICIENT_MEMORY;
    *yCount_block = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*yCount_block == NULL) return ERR_INSUFFICIENT_MEMORY;

    // Merge all indRows to generate indices, nnzCount_block and yCount_block
    nblocks=0;
    for (unsigned k=0; k < ceild(nrows,bsx); k++) {
        for (unsigned l=0; l<nblocksRow[k]; l++) {
            (*indices)[nblocks]=indRows[k][3*l];
            (*nnzCount_block)[nblocks]=indRows[k][3*l+1];
            (*yCount_block)[nblocks]=indRows[k][3*l+2];
            nblocks++;
        }
    }
    for (unsigned k=0; k < ceild(nrows,bsx); k++)
        free(indRows[k]);
    free(nblocksRow);
    free(indRows);
    free(lb);
    free(ub);
    free(bptr);
    free(colFlag);
    free(yflag);
    free(yBFlag);
    return 0;
}

int inspectVarBlock(SpMatrix *m, float **valFill, unsigned **indicesFill, unsigned **rowIndicesFill, unsigned **rowIndices, 
		  unsigned **indices, unsigned *numblocks, unsigned **nnzCount_block, unsigned **yCount_block,
                  unsigned *nnz_fill, unsigned bsx, unsigned bsy, unsigned varC)
{

    unsigned nblocks=0, nnz_filled=0;
    unsigned nrows = m->numRows;
    unsigned ncols = m->numCols;
    unsigned *yflag = (unsigned *)malloc(sizeof(int)*bsy);
    unsigned *yBFlag = (unsigned *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
    unsigned *lb = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *ub = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *bptr = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *colFlag = (unsigned *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
    unsigned *startCol = (unsigned *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
    unsigned *nblocksRow = (unsigned *)malloc(sizeof(int)*(int)ceild(nrows,bsx));
    unsigned **indRows = (unsigned **)malloc(sizeof(int*)*(int)ceild(nrows,bsx));

    *rowIndices = (unsigned *)malloc(sizeof(int)*((int)ceild(nrows,bsx)+1));
    if (*rowIndices == NULL) return ERR_INSUFFICIENT_MEMORY;

    // Assume bsy is a multiple of varC
    // Do blocking as if block size along column is varC
    // Then combine blocks of size bsy

    // for each block of row
    unsigned it, iti;
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        // start of a row block
        (*rowIndices)[iti]=nblocks;
        nblocksRow[iti]=0;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
            lb[i-it] = (m->rowPtrs)[i];
            if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1;
            else ub[i-it] = (m->rowPtrs)[i+1]-1;
            bptr[i-it] = lb[i-it];
        }
        // for each block of column within a row block
        for (unsigned jt = 0, jti = 0; jt < ncols; jt += bsy, jti++) {

    	    // Assume bsy is a multiple of varC
    	    // Skip zero column blocks (of size varC)

	    unsigned minjt=jt;
	    for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
                unsigned j = bptr[i-it];
                unsigned cInd = (m->nzentries)[j].colNum;
                for (unsigned jtv=jt; ; jtv+=varC) {
                    if (cInd < jtv+varC) {
			if (i==it) minjt = jtv; 
			else {
			    if (jtv < minjt) minjt = jtv;
			}
                        break;
		    }
                }
            }
	    jt=minjt;
	    startCol[jti]=jt;
            colFlag[jti]=0;
            yBFlag[jti]=0;
            for (unsigned k=0;k<bsy;k++)
                yflag[k]=0;
            unsigned blockStart = nblocks;

            for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
                unsigned j = bptr[i-it];
		unsigned rownnz = 0;
                for (; j <= ub[i-it]; j++) {
                    unsigned cInd = (m->nzentries)[j].colNum;
                    if (cInd >= jt+bsy) {
			//if ((rownnz%HALFWARP) || (!rownnz)) nnz_filled += (HALFWARP - (rownnz%HALFWARP));
			if (rownnz%HALFWARP) nnz_filled += (HALFWARP - (rownnz%HALFWARP));
                        break;
		    }
                    else {
                        if (!yflag[cInd-jt]) yflag[cInd-jt]++;
                        colFlag[jti]++; //colFlag[jti] stores the #nnzs in the block
			nnz_filled++;
			rownnz++;
                        if (blockStart == nblocks) {
                            nblocks++;
                            nblocksRow[iti]++;
                        }
                    }
                }
                bptr[i-it] = j;
		//if ( (j==(ub[i-it]+1)) && ((rownnz%HALFWARP) || (!rownnz)) ) nnz_filled += (HALFWARP - (rownnz%HALFWARP));
		if ( (j==(ub[i-it]+1)) && (rownnz%HALFWARP)  ) nnz_filled += (HALFWARP - (rownnz%HALFWARP));
            }
            for (unsigned k=0;k<bsy;k++)
                yBFlag[jti] += yflag[k];
        }
        indRows[iti] = (unsigned *)malloc(sizeof(int)*3*nblocksRow[iti]);
        if (indRows[iti] == NULL) return ERR_INSUFFICIENT_MEMORY;
        for (unsigned indRowk=0; indRowk < nblocksRow[iti]; indRowk++) {
            indRows[iti][3*indRowk]=startCol[indRowk];
            indRows[iti][3*indRowk+1]=colFlag[indRowk];
            indRows[iti][3*indRowk+2]=yBFlag[indRowk];
        }

    }
    (*rowIndices)[iti]=nblocks;

    *numblocks = nblocks;
    *indices = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*indices == NULL) return ERR_INSUFFICIENT_MEMORY;
    *nnzCount_block = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*nnzCount_block == NULL) return ERR_INSUFFICIENT_MEMORY;
    *yCount_block = (unsigned *)malloc(sizeof(int)*nblocks);
    if (*yCount_block == NULL) return ERR_INSUFFICIENT_MEMORY;

    // Merge all indRows to generate indices, nnzCount_block and yCount_block
    nblocks=0;
    for (unsigned k=0; k < ceild(nrows,bsx); k++) {
        for (unsigned l=0; l<nblocksRow[k]; l++) {
            (*indices)[nblocks]=indRows[k][3*l];
            (*nnzCount_block)[nblocks]=indRows[k][3*l+1];
            (*yCount_block)[nblocks]=indRows[k][3*l+2];
            nblocks++;
        }
    }

    // Fill in value

    *valFill = (float *)malloc(sizeof(float)*(nnz_filled));
    *indicesFill = (unsigned *)malloc(sizeof(int)*(nnz_filled));
    *rowIndicesFill = (unsigned *)malloc(sizeof(int)*(nrows+1));
    // One more loop to fill in val
    nnz_filled=0;
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        unsigned lbb = (*rowIndices)[iti];
        unsigned ubb = (*rowIndices)[iti+1]-1;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
	    (*rowIndicesFill)[i]=nnz_filled;
	    unsigned lbi, ubi;
            lbi = (m->rowPtrs)[i];
            if (i==(nrows-1)) ubi = m->numNZEntries-1;
            else ubi = (m->rowPtrs)[i+1]-1;
	    unsigned j=lbi;
	    unsigned rownnz=0;
            unsigned jti=(*indices)[lbb];
            for (unsigned jb = lbb; jb <= ubb; jb++) {
                jti = (*indices)[jb];
		rownnz=0;
                for (; j <= ubi; j++) {
                    unsigned cInd = (m->nzentries)[j].colNum;
                    if (cInd >= (jti+bsy)) {
			//if ((rownnz%HALFWARP) || (!rownnz)) {
			if (rownnz%HALFWARP) {
			    for (unsigned p=0;p<(HALFWARP - (rownnz%HALFWARP));p++) {
				(*valFill)[nnz_filled]=0;
				//(*indicesFill)[nnz_filled]=ncols+jti;
				(*indicesFill)[nnz_filled]=jti;
				nnz_filled++;
			    }
			}
			break;
		    }
                    else {
			(*valFill)[nnz_filled]=(m->nzentries)[j].val;
			(*indicesFill)[nnz_filled]=cInd;
			nnz_filled++;
			rownnz++;
		    }
                }
            }
	    //if ((rownnz%HALFWARP) || (!rownnz)) {
	    if (rownnz%HALFWARP) {
                for (unsigned p=0;p<(HALFWARP - (rownnz%HALFWARP));p++) {
                    (*valFill)[nnz_filled]=0;
		    //(*indicesFill)[nnz_filled]=ncols+jti;
		    (*indicesFill)[nnz_filled]=jti;
		    nnz_filled++;
		}
            }
        }
    }
    (*rowIndicesFill)[nrows]=nnz_filled;
    *nnz_fill = nnz_filled;

    for (unsigned k=0; k < ceild(nrows,bsx); k++)
        free(indRows[k]);
    free(nblocksRow);
    free(indRows);
    free(lb);
    free(ub);
    free(bptr);
    free(colFlag);
    free(startCol);
    free(yflag);
    free(yBFlag);
    return 0;

}

int inspectInputBlock(SpMatrix *m, unsigned **inputList, unsigned **rowIndices, unsigned **indices,
                  unsigned *numblocks, unsigned *inputListCount, unsigned bsx, unsigned bsy)
{
    unsigned nblocks=0, nblocksPerRowBlock;
    unsigned nrows = m->numRows;
    unsigned ncols = m->numCols;
    unsigned *lb = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *ub = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *bptr = (unsigned *)malloc(sizeof(int)*bsx);
    unsigned *inpListRowBlock = (unsigned *)malloc(sizeof(int)*bsx*bsy);

    *rowIndices = (unsigned *)malloc(sizeof(int)*((int)ceild(nrows,bsx)+1));

    unsigned it, iti;
    unsigned maxNZPerRowBlock;
    // for each block of row
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        // start of a row block
        (*rowIndices)[iti]=nblocks;
	maxNZPerRowBlock=0;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
            if (i==(nrows-1)) 
                maxNZPerRowBlock =Max(maxNZPerRowBlock,(m->numNZEntries-(m->rowPtrs)[i]));
	    else
                maxNZPerRowBlock =Max(maxNZPerRowBlock,((m->rowPtrs)[i+1]-(m->rowPtrs)[i]));
        }
	nblocks += (int) ceild(maxNZPerRowBlock,bsy);
    }
    (*rowIndices)[iti]=nblocks;
    *numblocks = nblocks;
    *indices = (unsigned *)malloc(sizeof(int)*(nblocks+1));

    unsigned countPerBlock,countInputList=0;
    nblocks=0;
    // for each block of row
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        // start of a row block
	maxNZPerRowBlock=0;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
	    lb[i-it] = (m->rowPtrs)[i];
            if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1; else ub[i-it] = (m->rowPtrs)[i+1]-1;
	    maxNZPerRowBlock =Max(maxNZPerRowBlock,ub[i-it]-lb[i-it]+1);
	    bptr[i-it] = lb[i-it];
        }
	nblocksPerRowBlock = (int) ceild(maxNZPerRowBlock,bsy);
        // for each block of column within a row block
        for (unsigned jt = 0, jti = 0; jt < nblocksPerRowBlock; jt += bsy, jti++,nblocks++) {
	    (*indices)[nblocks]=countInputList;
	    countPerBlock=0;
            for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
                unsigned j = bptr[i-it];
                for (; j <=Min(bptr[i-it]+bsy-1,ub[i-it]); j++) {
                    unsigned cInd = (m->nzentries)[j].colNum;
		    if (i==it) inpListRowBlock[countPerBlock++]=cInd;
		    else {
			if (!isPresent(inpListRowBlock,countPerBlock,cInd))
			    inpListRowBlock[countPerBlock++]=cInd;
		    }
                }
                bptr[i-it] = j;
            }
	    countInputList += countPerBlock;
        }
    }
    (*indices)[nblocks]=countInputList;

    *inputList = (unsigned *)malloc(sizeof(int)*countInputList);
    *inputListCount = countInputList;
    countInputList = 0;
    nblocks=0;
    // for each block of row
    for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
        // start of a row block
	maxNZPerRowBlock=0;
        for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
	    lb[i-it] = (m->rowPtrs)[i];
            if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1; else ub[i-it] = (m->rowPtrs)[i+1]-1;
	    maxNZPerRowBlock =Max(maxNZPerRowBlock,ub[i-it]-lb[i-it]+1);
	    bptr[i-it] = lb[i-it];
        }
	nblocksPerRowBlock = (int) ceild(maxNZPerRowBlock,bsy);
        // for each block of column within a row block
        for (unsigned jt = 0, jti = 0; jt < nblocksPerRowBlock; jt += bsy, jti++,nblocks++) {
	    countPerBlock=0;
            for (unsigned i = it; i <Min(it+bsx,nrows); i++) {
                unsigned j = bptr[i-it];
                for (; j <=Min(bptr[i-it]+bsy-1,ub[i-it]); j++) {
                    unsigned cInd = (m->nzentries)[j].colNum;
		    if (i==it) inpListRowBlock[countPerBlock++]=cInd;
		    else {
			if (!isPresent(inpListRowBlock,countPerBlock,cInd))
			    inpListRowBlock[countPerBlock++]=cInd;
		    }
                }
                bptr[i-it] = j;
            }
	    //Sort inpListRowBlock
	    sort(inpListRowBlock,countPerBlock);
	    if ( (countPerBlock > (bsx*bsy/2)) && (inpListRowBlock[countPerBlock-1] - inpListRowBlock[0] <512) ) {
	        for (int k=0;k<countPerBlock;k++)
		    (*inputList)[countInputList+k]=inpListRowBlock[k];
	    }
	    else {
	        for (int k=0;k<countPerBlock;k++)
		    //(*inputList)[countInputList+k]=inpListRowBlock[k];
		    (*inputList)[countInputList+k]=ncols;
	    }
	    countInputList += countPerBlock;
        }
    }

    free(inpListRowBlock);
    free(lb);
    free(ub);
    free(bptr);

    return 0;
}

