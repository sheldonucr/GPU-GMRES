/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "SpMV.h"
#include "config.h"
#include "defs.h"

#include <iostream>
using namespace std;

#define  PRINTER_WIDTH  80

int cmpRow(const void *e1, const void *e2) {
	if (((NZEntry *)e1)->rowNum < ((NZEntry *)e2)->rowNum) return -1;
	if (((NZEntry *)e1)->rowNum > ((NZEntry *)e2)->rowNum) return 1;
	if (((NZEntry *)e1)->colNum < ((NZEntry *)e2)->colNum) return -1;
	if (((NZEntry *)e1)->colNum > ((NZEntry *)e2)->colNum) return 1;
	return 0;
}

int cmpCol(const void *e1, const void *e2) {
	if (((NZEntry *)e1)->colNum < ((NZEntry *)e2)->colNum) return -1;
	if (((NZEntry *)e1)->colNum > ((NZEntry *)e2)->colNum) return 1;
	if (((NZEntry *)e1)->rowNum < ((NZEntry *)e2)->rowNum) return -1;
	if (((NZEntry *)e1)->rowNum > ((NZEntry *)e2)->rowNum) return 1;
	return 0;
}

void readInputVector(float *y, const char *filename, int numCols)
{
	FILE *f;
	f = fopen(filename,"r");
	if (!f) {
		fprintf(stderr,"Cannot open file: %s\n",filename);
		exit(-1);
	}

	int i=0;
	char line[256];
	while ( ((fgets(line, 256, f)) != NULL) && (i<numCols) ) {
		sscanf(line,"%f", &(y[i]));
		i++;
	}
	if (i<numCols) exit(-1);
	fclose(f);
}

void writeOutputVector(float *x, const char *filename, int numRows)
{
	FILE *f;
	f = fopen(filename,"w");
	if (!f) {
		fprintf(stderr,"Cannot open file: %s\n",filename);
		exit(-1);
	}

	int i;
	for (i = 0; i < numRows; i++)
		fprintf(f,"%20.18e\n", x[i]);

	fclose(f);
}

void loadVectorFromFile(ifstream &fin_u_vec, int u_vec_numElements, float *h_u_vec){
	assert(!fin_u_vec.fail());

	for(int i=0; i<u_vec_numElements; ++i){
		fin_u_vec >> h_u_vec[i];
	}
}

void writeOutputVector(float *x, FILE *f, int numRows)
{
	if (f == NULL) {
		fprintf(stderr,"Not a valid FILE pointer!\n");
		exit(-1);
	}

	int i;
	for (i = 0; i < numRows; i++)
		fprintf(f, "%20.18e\n", x[i]);

	fprintf(f, "\n");
}

void readSparseMatrix(SpMatrix *m, const char *filename, int format)
{
	trace;

	FILE *f;
	f = fopen(filename,"r");
	if (!f) {
		fprintf(stderr,"Cannot open file: %s\n",filename);
		fprintf(stdout,"Cannot open file: %s\n",filename);
		exit(-1);
	}
	trace;

	char line[256];
	while ( (fgets(line, 256, f)) != NULL) {
		if (line[0] != '%') break;  
	}
	trace;

	float fnumRow, fnumCols, fnumNZEntries;
	if ( (sscanf(line,"%f %f %f", &fnumRow, &fnumCols, &fnumNZEntries)) != 3)
		exit(-1);
	m->numRows = (int)fnumRow;
	m->numCols = (int)fnumCols;
	m->numNZEntries = (int)fnumNZEntries;

	trace;

	m->nzentries = (NZEntry *) malloc(sizeof(NZEntry) * (m->numNZEntries));
	m->rowPtrs = (int *) malloc(sizeof(int) * (m->numRows));
	m->colPtrs = (int *) malloc(sizeof(int) * (m->numCols));

	NZEntry e;
	int i;
	float temp_row, temp_col, temp_val;
	for (i = 0; i < m->numNZEntries; i++) {
		//fscanf(f,"%d %d %f\n", &(e.rowNum), &(e.colNum), &(e.val));
		// attention, reading parttern modified by zky, so that the program can read in index represented with floats
		fscanf(f,"%f %f %f\n", &(temp_row), &(temp_col), &(temp_val));

		e.rowNum = (int)temp_row;
		e.colNum = (int)temp_col;
		e.val = temp_val;

		// row and column indices begin with 1 in MatrixMarket
		// in our storage, row and column indices begin with 0!!!
		e.rowNum--; e.colNum--;
		(m->nzentries)[i]= e;
	}

	prval(format);

	/*
	   printf("Before sort\n");
	   for(int ii=0; ii<m->numNZEntries; ++ii){
	   printf("index: %d, Val: %f, row: %u, col: %u\n", ii, m->nzentries[ii].val, m->nzentries[ii].rowNum, m->nzentries[ii].colNum);
	   }
	 */

	// sort into row-major order or column major order based on the format
	if (format == 0) { // ROW-MAJOR
		qsort(m->nzentries, m->numNZEntries, sizeof(NZEntry), cmpRow);
		// set index of first elt in each row
		// relies on at least one item in each row
		m->rowPtrs[0]=0;
		int row, prevrow=0;
		for (i = 1; i < m->numNZEntries; i++) {
			row = (m->nzentries)[i].rowNum;
			if (row != prevrow) {
				prevrow = row;
				m->rowPtrs[prevrow]=i;
			}
		}

		trace;

	}
	else if (format == 1) { // COLUMN-MAJOR
		qsort(m->nzentries, m->numNZEntries, sizeof(NZEntry), cmpCol);
		// set index of first elt in each col
		// relies on at least one item in each col
		m->colPtrs[0]=0;
		int col, prevcol=0;
		for (i = 1; i < m->numNZEntries; i++) {
			col = (m->nzentries)[i].colNum;
			if (col != prevcol) {
				prevcol = col;
				m->colPtrs[prevcol]=i;
			}
		}
	}
	else { }

	fclose(f);
}

void genCSRFormat(SpMatrix * m, float *val, int *rowIndices, int *indices)
{
	int numRows = m->numRows;
	for (int i = 0; i < m->numNZEntries; i++) {
		val[i] = (m->nzentries)[i].val;
		indices[i] = (m->nzentries)[i].colNum;
	}

	for (int i = 0; i < numRows; i++) {
		rowIndices[i] = m->rowPtrs[i];
	}
	rowIndices[numRows] = m->numNZEntries;
}

void genCSCFormat(SpMatrix * m, float *val, int *colIndices, int *indices)
{
	int numCols = m->numCols;
	for ( int i = 0; i < m->numNZEntries; i++) {
		val[i] = (m->nzentries)[i].val;
		indices[i] = (m->nzentries)[i].rowNum;
	}

	for ( int i = 0; i < numCols; i++) {
		colIndices[i] = m->colPtrs[i];
	}
	colIndices[numCols] = m->numNZEntries;
}

void genBCSRFormat(SpMatrix *m, float **val,  int **rowIndices,  int **indices,
		 int *numblocks,  int bsx,  int bsy)
{
	 int nblocks=0;
	 int nrows = m->numRows;
	 int ncols = m->numCols;
	 int *lb = ( int *)malloc(sizeof(int)*bsx);
	 int *ub = ( int *)malloc(sizeof(int)*bsx);
	 int *bptr = ( int *)malloc(sizeof(int)*bsx);
	 int *colFlag = ( int *)malloc(sizeof(int)*(int)ceild(ncols,bsy));
	 int *nblocksRow = ( int *)malloc(sizeof(int)*(int)ceild(nrows,bsx));
	 int **indRows = ( int **)malloc(sizeof(int*)*(int)ceild(nrows,bsx));

	*rowIndices = ( int *)malloc(sizeof(int)*((int)ceild(nrows,bsx)+1));


	// for each block of row 
	 int it, iti;
	for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
		// start of a row block
		(*rowIndices)[iti]=nblocks;
		nblocksRow[iti]=0;
		for ( int i = it; i < min(it+bsx,nrows); i++) {
			lb[i-it] = (m->rowPtrs)[i];
			if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1;
			else ub[i-it] = (m->rowPtrs)[i+1]-1;
			bptr[i-it] = lb[i-it];
		}
		// for each block of column within a row block
		for ( int jt = 0, jti = 0; jt < ncols; jt += bsy, jti++) {
			colFlag[jti]=0;
			 int blockStart = nblocks;
			for ( int i = it; i < min(it+bsx,nrows); i++) {
				 int j = bptr[i-it];
				for (; j <= ub[i-it]; j++) {
					 int cInd = (m->nzentries)[j].colNum;
					if (cInd >= jt+bsy)  
						break;
					if (blockStart == nblocks) {
						nblocks++;
						nblocksRow[iti]++;
						colFlag[jti]=1;
					}
				}
				bptr[i-it] = j; 
			}
		}
		indRows[iti] = ( int *)malloc(sizeof(int)*nblocksRow[iti]);
		for ( int k=0, indRowk=0; k < ceild(ncols,bsy); k++) {
			if (colFlag[k]) {
				indRows[iti][indRowk]=k;
				indRowk++;
			}
		}

	}
	(*rowIndices)[iti]=nblocks;

	*numblocks = nblocks;
	*indices = ( int *)malloc(sizeof(int)*nblocks);
	*val = (float *)malloc(sizeof(float)*(nblocks*bsx*bsy));

	// Merge all indRows to generate indices
	nblocks=0;
	for ( int k=0; k < ceild(nrows,bsx); k++) {
		for ( int l=0; l<nblocksRow[k]; l++) {
			(*indices)[nblocks]=indRows[k][l];
			nblocks++;
		}
	}
	for ( int k=0; k < ceild(nrows,bsx); k++)
		free(indRows[k]);
	free(nblocksRow);
	free(indRows);

	// One more loop to fill in val 
	nblocks=0;
	for (it = 0, iti = 0; it < nrows; it += bsx, iti++) {
		 int lbb = (*rowIndices)[iti];
		 int ubb = (*rowIndices)[iti+1]-1;
		for ( int i = it; i < min(it+bsx,nrows); i++) {
			lb[i-it] = (m->rowPtrs)[i];
			if (i==(nrows-1)) ub[i-it] = m->numNZEntries-1;
			else ub[i-it] = (m->rowPtrs)[i+1]-1;
			bptr[i-it] = lb[i-it];
		}
		for ( int jb = lbb; jb <= ubb; jb++) {
			 int jti = (*indices)[jb];
			for ( int i = it; i < min(it+bsx,nrows); i++) {
				for ( int k = 0; k < bsy; k++)
					(*val)[nblocks*bsx*bsy+(i-it)*bsy+k]=0;
				 int j = bptr[i-it];
				for (; j <= ub[i-it]; j++) {
					 int cInd = (m->nzentries)[j].colNum;
					if (cInd >= ((jti*bsy)+bsy)) break;
					else (*val)[nblocks*bsx*bsy+(i-it)*bsy+(cInd-(jti*bsy))]=(m->nzentries)[j].val;
				}
				bptr[i-it] = j;
			}
			nblocks++;
		}
	}

	free(lb);
	free(ub);
	free(bptr);
	free(colFlag);
}

void genPaddedCSRFormat(SpMatrix * m, float **val, int **rowIndices, int **indices)
{

	/*-----------------------------------------------
	  Calculate how many zeros are needed for padding
	  -----------------------------------------------*/
	// initialization
	// int numRows = m->numRows;
	int prevRowNum = -1;
	int padNZ=0, nnzRow;

	for (int i = 0; i < m->numRows-1; i++) {
		nnzRow = m->rowPtrs[i+1]-m->rowPtrs[i]; // nnz in each row
		if (nnzRow%HALFWARP) {
			padNZ += nnzRow + HALFWARP-(nnzRow%HALFWARP); 
			// printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",
			//         i,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
		}
		else {
			padNZ += nnzRow;
			//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",i,nnzRow,nnzRow);
		}
	}
	// last row of the matrix
	nnzRow = m->numNZEntries - m->rowPtrs[m->numRows-1];
	//printf("nnz last row: %d\n",nnzRow); //XXLiu
	if (nnzRow%HALFWARP) {
		padNZ += nnzRow + HALFWARP-(nnzRow%HALFWARP);
		//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",m->numRows-1,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
	}
	else {
		padNZ += nnzRow;
		//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",m->numRows-1,nnzRow,nnzRow);
	}


	/*-----------------------------------------------
	  formulate the new padded matrix
	  -----------------------------------------------*/
	// allocate memory for CPU storage of the padded matrix
	(*val) = (float *)malloc(sizeof(float)*padNZ);
	(*indices) = (int *)malloc(sizeof(int)*padNZ);
	// allocate memory for row indices (integer type)
	(*rowIndices) = (int *)malloc(sizeof(int)*(m->numRows+1));

	int padNZCnt=0;
	for (int i = 0; i < m->numNZEntries; i++) {
		int currRowNum = (m->nzentries)[i].rowNum;

		if (currRowNum != prevRowNum) { //start of a row
			if (currRowNum && (padNZCnt%HALFWARP)) {// if currRowNum == 0, won't pad zero element!!!
				//Not first row and padNZCnt not a multiple of HALFWARP
				int fillCount = HALFWARP-(padNZCnt%HALFWARP);
				for (int j=0; j<fillCount; j++) {
					(*val)[padNZCnt]=0;
					(*indices)[padNZCnt]=0;
					padNZCnt++;
				}
			}

			(*rowIndices)[currRowNum]=padNZCnt;

			prevRowNum = currRowNum;
		}

		(*val)[padNZCnt] = (m->nzentries)[i].val;
		(*indices)[padNZCnt] = (m->nzentries)[i].colNum;
		padNZCnt++;
	}

	(*rowIndices)[m->numRows] = padNZCnt;

	/*
	   for(int i=0; i<m->numRows; ++i){
	   int lb = (*rowIndices)[i];
	   int ub = (*rowIndices)[i+1];
	   for(int j=lb; j<ub; ++j){
	   printf("(%d, %d, %f)\t", i, (*indices)[j], (*val)[j]);
	   }
	   printf("\n");
	   }
	 */


}

// added by XXLiu for matrix display. UNFINISHED
void printSparseMatrix(SpMatrix *m, int Data, int Header)
{
	/* Print header. */
	if (Header) {
		printf("MATRIX SUMMARY\n\n");
		printf("\nsparse matrix %d-by-%d: nnz=%d\n",
				m->numRows, m->numCols, m->numNZEntries);

		// if ( Matrix->Reordered && PrintReordered )
		//   printf("Matrix has been reordered.\n");
		// putchar('\n');
		// 
		// if ( Matrix->Factored )
		//   printf("Matrix after factorization:\n");
		// else
		//   printf("Matrix before factorization:\n");
		// 
		// SmallestElement = LARGEST_REAL;
		// SmallestDiag = SmallestElement;
	}

	int I, J, K, Col;
	int StartCol = 1, StopCol, Columns, ElementCount = 0;
	int PrintReordered = 1, Printer_Width=PRINTER_WIDTH;

	// sort the columns in each row
	int rowVec[ m->numRows ], rowVecAccum[ m->numRows ];
	NZEntry nz_tmp;
	for (I = 0; I < m->numRows; ++I) {
		rowVec[I] = 0;  rowVecAccum[I] = 0;
	}
	for (I = 0; I < m->numNZEntries; ++I) {
		rowVec[ (m->nzentries)[I].rowNum ] += 1;
	}
	for (I = 1; I < m->numRows; ++I) {
		rowVecAccum[ I ] = rowVecAccum[ I-1 ] + rowVec[ I-1 ];
	}
	for (I = 0; I < m->numRows; ++I)
		if (rowVec[I] > 0) {
			for (J = 0; J < rowVec[I]-1; ++J) {
				for (K = J+1; K < rowVec[I]; ++K)
					if ( (m->nzentries)[ J+rowVecAccum[I] ].colNum
							> (m->nzentries)[ K+rowVecAccum[I] ].colNum ) {
						nz_tmp = (m->nzentries)[ J+rowVecAccum[I] ];
						(m->nzentries)[ J+rowVecAccum[I] ] = (m->nzentries)[ K+rowVecAccum[I] ];
						(m->nzentries)[ K+rowVecAccum[I] ] = nz_tmp;
					}
			}
		}


	/* Determine how many columns to use. */
	Columns = Printer_Width;
	if (Header) Columns -= 5;
	if (Data) Columns = (Columns+1) / 10;

	for (I = 0; I < m->numNZEntries; ++I) {
		printf("%d %d %f\n",(m->nzentries)[I].rowNum+1,(m->nzentries)[I].colNum+1,
				(m->nzentries)[I].val);
	}


	J = 0;
	while ( J <= m->numCols ) {
		/* Calculatestrchr of last column to printed in this group. */
		StopCol = StartCol + Columns - 1;
		if (StopCol > m->numCols) StopCol = m->numCols;

		/* Label the columns. */
		if (Header) {
			printf("Columns %1d to %1d.\n",StartCol,StopCol);
		}

		/* ... in each column of the group. */
		for (J = StartCol; J <= StopCol; J++) {
			if (PrintReordered)
				Col = J-1;
			//else
			//  Col = PrintOrdToIntColMap[J];

			printf("%6.4e  ", (m->nzentries)[J].val);

			// pElement = Matrix->FirstInCol[Col];
			// while(pElement != NULL && pElement->Row != Row)
			//   pElement = pElement->NextInCol;
			// 
			// if (Data) pImagElements[J - StartCol] = pElement;
			// 
			// if (pElement != NULL) {
			//   /* Case where element exists */
			//   if (Data)
			//     printf(" %9.3g", (double)pElement->Real);
			//   else
			//     putchar('x');
			// 
			//   /* Update status variables */
			//   if ( (Magnitude = ELEMENT_MAG(pElement)) > LargestElement )
			//     LargestElement = Magnitude;
			//   if ((Magnitude < SmallestElement) && (Magnitude != 0.0))
			//     SmallestElement = Magnitude;
			// 
			//   ElementCount++;
			// }
			// /* Case where element is structurally zero */
			// else {
			//   if (Data)
			//     printf("       ...");
			//   else
			//     putchar('.');
			// }
		}

		putchar('\n');

		// if (Matrix->Complex && Data) {
		// 	printf("    ");
		// 	for (J = StartCol; J <= StopCol; J++) {
		// 	  if (pImagElements[J - StartCol] != NULL)
		// 	    printf(" %8.2gj",(double)pImagElements[J-StartCol]->Imag);
		// 	  else printf("          ");
		// 	}
		// 	putchar('\n');
		// }
		//   }
		// 
		//   /* Calculatestrchr of first column in next group. */
		//   StartCol = StopCol;
		//   StartCol++;
		//   putchar('\n');
}

}
