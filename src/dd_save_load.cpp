#include <iostream>
#include <fstream>
#include "cs.h"
#include "etbr_dd.h"

using namespace std;

void numeric_dl_save(ofstream &file, cs_dln *N)
{
  cs_dl *L = N->L;
  cs_dl *U = N->U;
  UF_long n = L->n;
  cs_dl_save(file, L);
  cs_dl_save(file, U);
  file.write((char *)N->pinv, sizeof(UF_long)*n);

}

void numeric_dl_load(ifstream &file, cs_dln *&N)
{
  N = (cs_dln *)cs_dl_calloc(1, sizeof(cs_dln));
  cs_dl_load(file, N->L);
  cs_dl_load(file, N->U);
  UF_long n = N->L->n;
  N->pinv = (UF_long *) cs_malloc (n, sizeof (UF_long));
  file.read((char *)N->pinv, sizeof(UF_long)*n);
}

void cs_dl_save(ofstream &file, cs_dl *A)
{
  file.write((char *)&A->nzmax, sizeof(UF_long));
  file.write((char *)&A->m, sizeof(UF_long));
  file.write((char *)&A->n, sizeof(UF_long));
  file.write((char *)A->p, sizeof(UF_long)*(A->n+1));
  file.write((char *)A->i, sizeof(UF_long)*(A->nzmax));
  file.write((char *)A->x, sizeof(double)*(A->nzmax));
  file.write((char *)&A->nz, sizeof(UF_long));
}

void cs_dl_load(ifstream &file, cs_dl *&A)
{
  char *nzmax, *m, *n, *nz;
  nzmax = new char [sizeof(UF_long)];
  m = new char [sizeof(UF_long)];
  n = new char [sizeof(UF_long)];
  nz = new char [sizeof(UF_long)];
  file.read(nzmax, sizeof(UF_long));
  file.read(m, sizeof(UF_long));
  file.read(n, sizeof(UF_long));  
  A = cs_dl_spalloc(*(UF_long*)m, *(UF_long*)n, *(UF_long*)nzmax, 1, 0);
  file.read((char *)A->p, sizeof(UF_long)*(A->n+1));
  file.read((char *)A->i, sizeof(UF_long)*(A->nzmax));
  file.read((char *)A->x, sizeof(double)*(A->nzmax));
  file.read(nz, sizeof(UF_long));
  A->nz = *(UF_long*) nz;
  delete [] m;
  delete [] n;
  delete [] nzmax;
  delete [] nz;
}
