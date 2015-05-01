/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: matrix.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:59 $
 *    Authors: Ning Mi 
 *
 *    Functions: matrix class
 *
 */


#include "matrix.h"
#include "UFconfig.h"
#include <stdlib.h>
#include <algorithm>

bool entry_comp (Entry a, Entry b){
  return a.i<b.i;
}

matrix::matrix(int m, int n)
{
        int i;

        rowsize = m;
	colsize = n;
	nnz = 0;
	//rowIndex = (Entry**)malloc(m*sizeof(Entry*));
	colIndex = (Entry**)malloc(n*sizeof(Entry*));
	/*for( i=0 ; i<rowsize ; i++ ){
		rowIndex[i] = NULL;
		}*/
	num_per_col = (int*)malloc(n*sizeof(int));
	for( i=0;i<colsize;i++ ){
	  num_per_col[i] = 0;
	  colIndex[i] = NULL;
	}
	uncompact_data = NULL;
	compact_data = NULL;	
}

matrix::~matrix()
{
  /*Entry *temp_pr, *temp_pr1;
  for (int i=0;i<rowsize;i++){
	temp_pr = rowIndex[i];
	while(temp_pr != NULL){
	  temp_pr1 = temp_pr;
	  temp_pr = temp_pr->rowNext;
	  free(temp_pr1);
	}
	}*/

  for(int i=0;i<colsize;i++){
    if(colIndex[i]!=NULL)
      free(colIndex[i]);
  }
        /*
	for (int j=0;j<colsize;j++){
	  if(colIndex[j] != NULL)
		free(colIndex[j]);
	}
	*/
	//free(rowIndex);
  free(colIndex);
  free(num_per_col);
  if( compact_data != NULL) free(compact_data);
  if( uncompact_data != NULL) delete uncompact_data;
}

/*void matrix::pushEntry(int i, int j, double value)
{
	Entry* temp_entry = (Entry*)malloc(sizeof(Entry));	
	temp_entry->i = i;
	temp_entry->j = j;
	temp_entry->value = value;
	temp_entry->rowNext = rowIndex[i];
	rowIndex[i] = temp_entry;
	}
*/

void matrix::pushEntry(int i, int j, double value)
{
  int k,num;
  int mark = 0;
  int nblock;
  if(num_per_col[j]==0)
    {
      colIndex[j] = (Entry*)malloc(ROW_NUM * sizeof(Entry));
      colIndex[j][0].i = i;
      colIndex[j][0].value = value;    
      num_per_col[j]++;
      nnz++;
    }
  else
    {
      for (k = 0; k < num_per_col[j]; k++){
	if(colIndex[j][k].i == i){
	  colIndex[j][k].value += value;
	  mark = 1;
	  break;
	}
      }
      if(mark != 1){
	num = num_per_col[j];
	if(num%ROW_NUM==0){
	  nblock = num/ROW_NUM + 1;
	  colIndex[j] = (Entry*)realloc(colIndex[j], nblock*ROW_NUM * sizeof(Entry));
	}	
	colIndex[j][num].i = i;
	colIndex[j][num].value = value;
	num_per_col[j]++;
	nnz++;
      }
    }
}

//bool Entry::operator<(const Entry& a, const Entry& b){
//  return a.i<b.i;
//}

void matrix::sort()
{
  int j;
  for(j = 0;j < colsize; j++){
    colIndex[j] = (Entry*)realloc(colIndex[j], num_per_col[j] * sizeof(Entry));
    std::sort(colIndex[j], colIndex[j]+num_per_col[j], entry_comp);
  }
}

/*void matrix::sort()
{
	int i;
	Entry* temp_pr, *temp_pr1, * temp_pc;

	for(i=rowsize-1,nnz=0;i>=0;i--)
	{
		temp_pr = rowIndex[i];
		temp_pr1 = temp_pr;
		while(temp_pr!=NULL){
			temp_pc = colIndex[temp_pr->j];
			if(temp_pc == NULL || temp_pc->i != i){
				temp_pr->colNext = temp_pc;
				colIndex[temp_pr->j] = temp_pr;
				nnz++;
				temp_pr1 = temp_pr;
				temp_pr = temp_pr->rowNext;
			}
			else{
				temp_pc->value += temp_pr->value;
				if (temp_pr != rowIndex[i]){
				  temp_pr1->rowNext = temp_pr->rowNext;
				  free(temp_pr);
				  temp_pr = temp_pr1->rowNext;
				}else{
				  rowIndex[i] = temp_pr->rowNext;
				  free(temp_pr);
				  temp_pr = rowIndex[i];
				}
			}
		}
		
	}

	for(i=rowsize-1;i>=0;i--) rowIndex[i]=NULL;
	
	for(i=colsize-1;i>=0;i--){
		temp_pc = colIndex[i];
		while(temp_pc!=NULL){
			temp_pc->rowNext = rowIndex[temp_pc->i];
			rowIndex[temp_pc->i] = temp_pc;
			temp_pc = temp_pc->colNext;
		}
	} 
   
}
*/

/*void matrix::compact()
{
	int i;
	Entry* temp_old, *temp_new, ** temp_pp;
	Entry* temp_data = (Entry*)malloc(nnz* sizeof(Entry));
	for (i=0;i<colsize;i++) colIndex[i] = NULL;
	for (i=rowsize-1, temp_new=temp_data;i>=0;i--)
	{
		temp_old = rowIndex[i];
		temp_pp = &(rowIndex[i]);
		while(temp_old!=NULL)
		{
			temp_new->i = temp_old->i;
			temp_new->j = temp_old->j;
			temp_new->value = temp_old->value;
			temp_new->colNext = colIndex[temp_new->j];
			colIndex[temp_new->j] = temp_new;
			(*temp_pp) = temp_new;
			temp_pp = &(temp_new->rowNext);
			temp_old = temp_old->rowNext;
			temp_new++;
		}
		(*temp_pp) = NULL;
	}
	if(uncompact_data != NULL){
		delete uncompact_data;
		uncompact_data = NULL;
	}
	if(compact_data != NULL){
		free(compact_data);
		compact_data = temp_data;
	}
	}*/


void matrix::printmatrix(FILE* fid)
{
  int i,j;
	
  if(fid = NULL)
    {
      for (i=0;i<colsize;i++){
	for (j=0;j<num_per_col[i];j++){
	  printf("(%d, %d)%.3e\t", colIndex[i][j].i, i, colIndex[i][j].value);
	}  
	printf("\n");
      }
      printf("\n");
    }
  else
    {
      for (i=0;i<colsize;i++){
	for (j=0;j<num_per_col[i];j++){
	  printf("(%d, %d)%.3e\t", colIndex[i][j].i, i, colIndex[i][j].value);
	}  
	printf("\n");
      }
      printf("\n");
    }

}


void matrix::printmatrix()
{
  int i,j;

  for (i=0;i<colsize;i++){
    for (j=0;j<num_per_col[i];j++){
      printf("(%d, %d)%.3e\t", colIndex[i][j].i, i, colIndex[i][j].value);
    }  
    printf("\n");
  }
  printf("\n");
}
	
/*void matrix::trans(sparse_mat* smatrix)
{
  Entry* temp_entry;
  int i;

  for( i=0; i<colsize; i++ ){
    temp_entry = colIndex[i];
    while(temp_entry != NULL){
      smatrix->add_elem(temp_entry->i,i,temp_entry->value);
      temp_entry = temp_entry->colNext;
    }
  }

  }*/

/*cs*  matrix::mat2cs()
{
  cs *T = cs_spalloc(rowsize, colsize, nnz, 1, 0);
  Entry* temp_entry;
  int i, nnz_tmp;
               
  nnz_tmp = 0;
  for( i=0; i<colsize; i++ ){
    temp_entry = colIndex[i];
    T->p[i] = nnz_tmp;
    while(temp_entry != NULL){
      T->i[nnz_tmp] = temp_entry->i;
      T->x[nnz_tmp] = temp_entry->value;
      nnz_tmp++;
      temp_entry = temp_entry->colNext;
    }
  }
  T->p[colsize] = nnz;
  return T;
  }*/

/*cs_dl*  matrix::mat2csdl()
{
  cs_dl *T = cs_dl_spalloc((UF_long) rowsize, (UF_long) colsize, (UF_long) nnz, 1, 0);
  Entry* temp_entry;
  UF_long i, nnz_tmp;
               
  nnz_tmp = 0;
  for( i=0; i<colsize; i++ ){
    temp_entry = colIndex[i];
    T->p[i] = nnz_tmp;
    while(temp_entry != NULL){
      T->i[nnz_tmp] = temp_entry->i;
      T->x[nnz_tmp] = temp_entry->value;
      nnz_tmp++;
      temp_entry = temp_entry->colNext;
    }
  }
  T->p[colsize] = nnz;
  return T;
  }
*/

cs*  matrix::mat2cs()
{
  if (nnz == 0)
	return NULL;
  cs *T = cs_spalloc(rowsize, colsize, nnz, 1, 0);
  Entry* temp_entry;
  int i, j, nnz_tmp;
               
  nnz_tmp = 0;
  for( i=0; i<colsize; i++ ){
    T->p[i] = nnz_tmp;
    for( j=0; j<num_per_col[i]; j++){
      T->i[nnz_tmp] = colIndex[i][j].i;
      T->x[nnz_tmp] = colIndex[i][j].value;
      nnz_tmp++;
    }
  }
  T->p[colsize] = nnz;
  return T;
}

cs_dl*  matrix::mat2csdl()
{
  if (nnz == 0)
	return NULL;
  cs_dl *T = cs_dl_spalloc((UF_long) rowsize, (UF_long) colsize, (UF_long) nnz, 1, 0);
  Entry* temp_entry;
  UF_long i, j, nnz_tmp;
               
  nnz_tmp = 0;
  for( i=0; i<colsize; i++ ){
    T->p[i] = nnz_tmp;
    for( j=0; j<num_per_col[i]; j++){
      T->i[nnz_tmp] = colIndex[i][j].i;
      T->x[nnz_tmp] = colIndex[i][j].value;
      nnz_tmp++;
    }
  }
  T->p[colsize] = nnz;
  return T;
}
