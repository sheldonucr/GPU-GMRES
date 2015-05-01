#include "gpuData.h"

void sort(int *col_idx, float *a, int start, int end)
{
  int i, j, it;
  float dt;

  for (i=end-1; i>start; i--)
    for(j=start; j<i; j++)
      if (col_idx[j] > col_idx[j+1]){

	if (a){
	  dt=a[j]; 
	  a[j]=a[j+1]; 
	  a[j+1]=dt;
        }
	it=col_idx[j]; 
	col_idx[j]=col_idx[j+1]; 
	col_idx[j+1]=it;
	  
      }
}

void sortDouble(int *col_idx, double *a, int start, int end)
{
  int i, j, it;
  double dt;

  for (i=end-1; i>start; i--)
    for(j=start; j<i; j++)
      if (col_idx[j] > col_idx[j+1]){

	if (a){
	  dt=a[j]; 
	  a[j]=a[j+1]; 
	  a[j+1]=dt;
        }
	it=col_idx[j]; 
	col_idx[j]=col_idx[j+1]; 
	col_idx[j+1]=it;
	  
      }
}


/* converts COO format to CSR format, in-place,
 * if SORT_IN_ROW is defined, each row is sorted in column index.
 * On return, i_idx contains row_start position */
void coo2csr_in(int n, int nz, float *a, int *i_idx, int *j_idx)
{
  int *row_start;
  row_start = (int *)malloc((n+1)*sizeof(int));
  if (!row_start){
    printf ("coo2csr_in: cannot allocate temporary memory\n");
    exit (1);
  }

  int i, j;
  int init, i_next, j_next, i_pos;
  float dt, a_next;

  for (i=0; i<=n; i++) row_start[i] = 0;
  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
  for (i=0; i<n; i++) row_start[i+1] += row_start[i]; // csrRowPtr

  for (init=0; init<nz; ){
    dt = a[init];
    i = i_idx[init];
    j = j_idx[init];
    i_idx[init] = -1; // flag
    while (1){
      i_pos = row_start[i];
      a_next = a[i_pos];
      i_next = i_idx[i_pos];
      j_next = j_idx[i_pos];

      a[i_pos] = dt;
      j_idx[i_pos] = j;
      i_idx[i_pos] = -1;
      row_start[i]++;
      if (i_next < 0) break;
      dt = a_next;
      i = i_next;
      j = j_next;

    }
    init++;
    while ( (init < nz) && (i_idx[init] < 0))  init++;
  }
  /* shift back row_start */
  for (i=0; i<n; i++) i_idx[i+1] = row_start[i];
  i_idx[0] = 0;

  for (i=0; i<n; i++){
    sort (j_idx, a, i_idx[i], i_idx[i+1]);
  }

  free(row_start);
}

void coo2csrDouble_in(int n, int nz, double *a, int *i_idx, int *j_idx)
{
  int *row_start;
  row_start = (int *)malloc((n+1)*sizeof(int));
  if (!row_start){
    printf ("coo2csr_in: cannot allocate temporary memory\n");
    exit (1);
  }

  int i, j;
  int init, i_next, j_next, i_pos;
  double dt, a_next;

  for (i=0; i<=n; i++) row_start[i] = 0;
  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
  for (i=0; i<n; i++) row_start[i+1] += row_start[i]; // csrRowPtr

  for (init=0; init<nz; ){
    dt = a[init];
    i = i_idx[init];
    j = j_idx[init];
    i_idx[init] = -1; // flag
    while (1){
      i_pos = row_start[i];
      a_next = a[i_pos];
      i_next = i_idx[i_pos];
      j_next = j_idx[i_pos];

      a[i_pos] = dt;
      j_idx[i_pos] = j;
      i_idx[i_pos] = -1;
      row_start[i]++;
      if (i_next < 0) break;
      dt = a_next;
      i = i_next;
      j = j_next;

    }
    init++;
    while ( (init < nz) && (i_idx[init] < 0))  init++;
  }
  /* shift back row_start */
  for (i=0; i<n; i++) i_idx[i+1] = row_start[i];
  i_idx[0] = 0;

  for (i=0; i<n; i++){
    sortDouble(j_idx, a, i_idx[i], i_idx[i+1]);
  }

  free(row_start);
}


// /* converts COO format to CSR format, not in-place,
//  * if SORT_IN_ROW is defined, each row is sorted in column index */
// void coo2csr(int n, int nz, double *a, int *i_idx, int *j_idx,
// 	     double *csr_a, int *col_idx, int *row_start)
// {
//   int i, l;
// 
//   for (i=0; i<=n; i++) row_start[i] = 0;
//   /* determine row lengths */
//   for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;
//   for (i=0; i<n; i++) row_start[i+1] += row_start[i];
// 
//   /* go through the structure  once more. Fill in output matrix. */
//   for (l=0; l<nz; l++){
//     i = row_start[i_idx[l]];
//     csr_a[i] = a[l];
//     col_idx[i] = j_idx[l];
//     row_start[i_idx[l]]++;
//   }
// 
//   /* shift back row_start */
//   for (i=n; i>0; i--) row_start[i] = row_start[i-1];
//   row_start[0] = 0;
// 
//   for (i=0; i<n; i++){
//     sort (col_idx, csr_a, row_start[i], row_start[i+1]);
//   }
// }

void LDcsc2csr(long int nnz, long int m, long int n,
               long int *cscColPtr, long int *cscRowIdx, double *cscVal,
               int **csrRowPtrIn, int **csrColIdxIn, float **csrValIn)
{
  *csrRowPtrIn=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
  *csrColIdxIn=(int*)malloc(nnz*sizeof(int));
  *csrValIn=(float*)malloc(nnz*sizeof(float));

  int *csrRowPtr = *csrRowPtrIn;
  int *csrColIdx = *csrColIdxIn;
  float *csrVal = *csrValIn;

  for(int j=0; j<n; j++) { // Convert to COO format first.
    int lb=cscColPtr[j], ub=cscColPtr[j+1];
    for(int i=lb; i<ub; i++)
      csrColIdx[i] = j;
  }
  //int *rowIdx=(int*)malloc(nnz*sizeof(int));
  for(int i=0; i<nnz; i++) {
    csrRowPtr[i] = (int)cscRowIdx[i];
    csrVal[i] = (float)cscVal[i];
  }

  coo2csr_in(m, nnz, csrVal, csrRowPtr, csrColIdx);

  // for(int i=0; i<nnz-1; i++) {
  //   for(int j=i+1; j<nnz; j++) {
  //     if(rowIdx[i] > rowIdx[j]) {
  //       int tmpRowIdx=rowIdx[i], tmpColIdx=csrColIdx[i];
  //       double tmpVal=cscVal[i];
  //       rowIdx[i] = rowIdx[j];  rowIdx[j] = tmpRowIdx;
  //       csrColIdx[i] = csrColIdx[j];  csrColIdx[j] = tmpColIdx;
  //       csrVal[i] = csrVal[j];  csrVal[j] = tmpVal;
  //     }
  //   }
  // }
  // 
  // int i=0;
  // csrRowPtr[0] = 0;
  // for(int j=0; j<nnz; j++) {
  //   if(rowIdx[j] > i) { /* Empty rows are considered. */
  //     for(int k=i+1; k<=rowIdx[j]; k++)
  //       csrRowPtr[k] = j;
  //     i = rowIdx[j];
  //   }
  // }
  // for( ; i<m+1; i++)
  //   csrRowPtr[i] = nnz;
  // free(rowIdx);
}


void LDcsc2csrMySpMatrix(MySpMatrix *mySpM, ucr_cs_dl *M)
{
  int nnz = M->nzmax;
  int m = M->m;
  int n = M->n;

  mySpM->numRows = m;
  mySpM->numCols = n;
  mySpM->numNZEntries = nnz;

  mySpM->rowIndices=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc(nnz*sizeof(int));
  mySpM->val=(float*)malloc(nnz*sizeof(float));

  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  float *csrVal = mySpM->val;

  for(int j=0; j<n; j++) { // Convert to COO format first.
    int lb=M->p[j], ub=M->p[j+1];
    for(int i=lb; i<ub; i++)
      csrColIdx[i] = j;
  }
  //int *rowIdx=(int*)malloc(nnz*sizeof(int));
  for(int i=0; i<nnz; i++) {
    csrRowPtr[i] = (int) M->i[i];
    csrVal[i] = (float) M->x[i];
    // if( csrVal[i] == 0 && M->x[i] != 0 )
    //   printf(" Accuracy loss M->x[%d]=%6.4e\n",i,M->x[i]);
  }

  coo2csr_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
}

void LDcsc2csrMySpMatrixDouble(MySpMatrixDouble *mySpM, ucr_cs_dl *M)
{
  int nnz = M->nzmax;
  int m = M->m;
  int n = M->n;

  mySpM->numRows = m;
  mySpM->numCols = n;
  mySpM->numNZEntries = nnz;

  mySpM->rowIndices=(int*)malloc((nnz > (m+1) ? nnz : (m+1))*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc(nnz*sizeof(int));
  mySpM->val=(double*)malloc(nnz*sizeof(double));

  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  double *csrVal = mySpM->val;

  for(int j=0; j<n; j++) { // Convert to COO format first.
    int lb=M->p[j], ub=M->p[j+1];
    for(int i=lb; i<ub; i++)
      csrColIdx[i] = j;
  }
  //int *rowIdx=(int*)malloc(nnz*sizeof(int));
  for(int i=0; i<nnz; i++) {
    csrRowPtr[i] = (int) M->i[i];
    csrVal[i] = M->x[i];
  }

  coo2csrDouble_in(m, nnz, csrVal, csrRowPtr, csrColIdx);
}

/* for (k = 0 ; k < n ; k++)
 *     x [p ? p [k] : k] = b [k] ; */
void vec2csrMySpMatrix(MySpMatrix *mySpM, int *p, int n)
{
  mySpM->numRows = n;
  mySpM->numCols = n;
  mySpM->numNZEntries = n;

  mySpM->rowIndices=(int*)malloc((n+1)*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc(n*sizeof(int));
  mySpM->val=(float*)malloc(n*sizeof(float));

  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  float *csrVal = mySpM->val;
  for(int i=0; i<n; i++) {
    csrVal[i] = 1.0;
    csrColIdx[p[i]] = i;
  }
  for(int i=0; i<n+1; i++)
    csrRowPtr[i] = i;
}

void writeCSR(int m, int n, int nnz,
              int *rowPtr, int *colIdx, float *val,
              const char *filename)
{
  FILE *f;
  f = fopen(filename,"wb");
  if(!f) {
    fprintf(stdout,"Cannot open file: %s\n",filename);
    exit(-1);
  }

  fwrite(&m, sizeof(int), 1, f);
  fwrite(&n, sizeof(int), 1, f);
  fwrite(&nnz, sizeof(int), 1, f);

  fwrite(rowPtr, sizeof(int), m+1, f);
  fwrite(colIdx, sizeof(int), nnz, f);
  fwrite(val, sizeof(float), nnz, f);
  
  fclose(f);
}

void writeCSRmySpMatrix(MySpMatrix *M, const char *filename)
{
  FILE *f;
  f = fopen(filename,"wb");
  if(!f) {
    fprintf(stdout,"Cannot open file: %s\n",filename);
    exit(-1);
  }

  fwrite(&(M->numRows), sizeof(int), 1, f);
  fwrite(&(M->numCols), sizeof(int), 1, f);
  fwrite(&(M->numNZEntries), sizeof(int), 1, f);

  fwrite(M->rowIndices, sizeof(int), M->numRows + 1, f);
  fwrite(M->indices, sizeof(int), M->numNZEntries, f);
  fwrite(M->val, sizeof(float), M->numNZEntries, f);
  
  fclose(f);
}

void writeCSRmySpMatrixDouble(MySpMatrixDouble *M, const char *filename)
{
  FILE *f;
  f = fopen(filename,"wb");
  if(!f) {
    fprintf(stdout,"Cannot open file: %s\n",filename);
    exit(-1);
  }

  fwrite(&(M->numRows), sizeof(int), 1, f);
  fwrite(&(M->numCols), sizeof(int), 1, f);
  fwrite(&(M->numNZEntries), sizeof(int), 1, f);

  fwrite(M->rowIndices, sizeof(int), M->numRows + 1, f);
  fwrite(M->indices, sizeof(int), M->numNZEntries, f);
  fwrite(M->val, sizeof(double), M->numNZEntries, f);
  
  fclose(f);
}
