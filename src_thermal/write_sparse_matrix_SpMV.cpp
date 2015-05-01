/*
   XXL: not finished
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int main( int argc, char** argv)
{
  // read in number of rows and number of columns
  if (argc < 4 || argc > 5) {
    printf("Correct Usage:\n"
	   "  write_sparse_matrix_SpMv <filename> <numRows> [<numCols>] <numNZentries>\n");
    exit(-1);
  }

  char spmfileName[256];
  // input matrix file
  strcpy(spmfileName,argv[1]);
  FILE* f;
  f = fopen(spmfileName, "w");
  if ( f == NULL) {
    printf("Cannot write input matrix file\n");
    exit(-1);
  }
  
  unsigned numRows, numCols, numNonZeroElements;
  if (argc == 4) {
    numRows = atoi(argv[2]);
    numCols = numRows;
    numNonZeroElements = atoi(argv[3]);
  }
  else {
    numRows = atoi(argv[2]);
    numCols = atoi(argv[3]);
    numNonZeroElements = atoi(argv[4]);
  }
  fprintf(f,"%d %d %d\n",numRows, numCols, numNonZeroElements);

  unsigned idxRow, idxCol;
  float val;
  for (unsigned i = 0; i < numNonZeroElements; i++) {
  // The following two lines are modified by zky, the index should start with 1, not the orginal 0
    idxRow = int( rand() / (float)RAND_MAX * (float)numRows ) + 1;
    idxCol = int( rand() / (float)RAND_MAX * (float)numCols ) + 1;
    val =  rand() / (float)RAND_MAX;
    fprintf(f,"%d %d %f\n", idxRow, idxCol, val);
  }


  fclose(f);

  exit(0);
}
