#include <iostream>
#include <fstream>

#include <itpp/base/timing.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/sort.h>
#include "umfpack.h"
#include "cs.h"

#include "etbr.h"
#include "interp.h"
#include "gpuData.h"
#include "SpMV.h"
#include "iluplusplus.h"

double threshold=1.2, factor=1.0; // threshold=3.0

typedef iluplusplus::Real Real;
typedef iluplusplus::matrix_sparse<Real> Matrix;
typedef iluplusplus::vector_dense<Real> Vector;
  

#include "gmres_interface_pg.h"
/* for(i=0;i<number_rows;i++)
 *     x[offset+i]=w[perm.get(i)];  */
void index_list2csrMySpMatrix(MySpMatrix *mySpM, iluplusplus::index_list &p, int n)
{
  mySpM->isCSR = 1;

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
    csrColIdx[i] = p.get(i);
  }
  for(int i=0; i<n+1; i++)
    csrRowPtr[i] = i;
}

void ILUPPmat2csrMySpMatrix(MySpMatrix *mySpM, const Matrix &M)
{
  mySpM->isCSR = 1;

  mySpM->numRows = M.rows();
  mySpM->numCols = M.columns();

  // if( M.orient() == iluplusplus::ROW) {
  //   mySpM->numNZEntries = M.get_pointer( mySpM->numRows );
  // }
  // else {
  //   mySpM->numNZEntries = M.get_pointer( mySpM->numCols );
  // }
  mySpM->numNZEntries = M.get_pointer( M.get_pointer_size()-1 );

  mySpM->rowIndices=(int*)malloc( (mySpM->numRows + 1)*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc( (mySpM->numNZEntries)*sizeof(int));
  mySpM->val=(float*)malloc( (mySpM->numNZEntries)*sizeof(float));
  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  float *csrVal = mySpM->val;

  if( M.orient() == iluplusplus::COLUMN) {
    for(int j=0; j<mySpM->numCols; j++) { // Convert to COO format first.
      int lb=M.get_pointer(j), ub=M.get_pointer(j+1);
      for(int i=lb; i<ub; i++)
        csrColIdx[i] = j;
    }
    //int *rowIdx=(int*)malloc(nnz*sizeof(int));
    for(int i=0; i<mySpM->numNZEntries; i++) {
      csrRowPtr[i] = M.get_index(i);
      csrVal[i] = (float) M.get_data(i);
      // if( csrVal[i] == 0 && M->x[i] != 0 )
      //   printf(" Accuracy loss M->x[%d]=%6.4e\n",i,M->x[i]);
    }
    coo2csr_in(mySpM->numRows, mySpM->numNZEntries, csrVal, csrRowPtr, csrColIdx);
  }
  else { // iluplusplus::ROW
    for(int j=0; j<=mySpM->numRows; j++)
      csrRowPtr[j] = M.get_pointer(j);
    for(int i=0; i<mySpM->numNZEntries; i++) {
      csrColIdx[i] = M.get_index(i);
      csrVal[i] = (float) M.get_data(i);
    }
  }
}

void ILUPPmat2csrMySpMatrixDouble(MySpMatrixDouble *mySpM, const Matrix &M)
{
  mySpM->isCSR = 1;

  mySpM->numRows = M.rows();
  mySpM->numCols = M.columns();

  // if( M.orient() == iluplusplus::ROW) {
  //   mySpM->numNZEntries = M.get_pointer( mySpM->numRows );
  // }
  // else {
  //   mySpM->numNZEntries = M.get_pointer( mySpM->numCols );
  // }
  mySpM->numNZEntries = M.get_pointer( M.get_pointer_size()-1 );

  mySpM->rowIndices=(int*)malloc( ((mySpM->numRows + 1)>mySpM->numNZEntries ?
                                   (mySpM->numRows + 1) : mySpM->numNZEntries)*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc( (mySpM->numNZEntries)*sizeof(int));
  mySpM->val=(double*)malloc( (mySpM->numNZEntries)*sizeof(double));
  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  double *csrVal = mySpM->val;

  if( M.orient() == iluplusplus::COLUMN) {
    for(int j=0; j<mySpM->numCols; j++) { // Convert to COO format first.
      int lb=M.get_pointer(j), ub=M.get_pointer(j+1);
      for(int i=lb; i<ub; i++)
        csrColIdx[i] = j;
    }
    //int *rowIdx=(int*)malloc(nnz*sizeof(int));
    for(int i=0; i<mySpM->numNZEntries; i++) {
      csrRowPtr[i] = M.get_index(i);
      csrVal[i] = M.get_data(i);
      // if( csrVal[i] == 0 && M->x[i] != 0 )
      //   printf(" Accuracy loss M->x[%d]=%6.4e\n",i,M->x[i]);
    }
    coo2csrDouble_in(mySpM->numRows, mySpM->numNZEntries, csrVal, csrRowPtr, csrColIdx);
  }
  else { // iluplusplus::ROW
    for(int j=0; j<=mySpM->numRows; j++)
      csrRowPtr[j] = M.get_pointer(j);
    for(int i=0; i<mySpM->numNZEntries; i++) {
      csrColIdx[i] = M.get_index(i);
      csrVal[i] = M.get_data(i);
    }
  }
}

void ILUPPvec2csrMySpMatrix(MySpMatrix *mySpM, const Vector &M)
{
  mySpM->isCSR = 1;

  mySpM->numRows = M.dim();
  mySpM->numCols = M.dim();
  mySpM->numNZEntries = M.dim();

  mySpM->rowIndices=(int*)malloc( (mySpM->numRows + 1)*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc( (mySpM->numNZEntries)*sizeof(int));
  mySpM->val=(float*)malloc( (mySpM->numNZEntries)*sizeof(float));
  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  float *csrVal = mySpM->val;

  for(int j=0; j<=mySpM->numRows; j++)
    csrRowPtr[j] = j;
  for(int i=0; i<mySpM->numNZEntries; i++) {
    csrColIdx[i] = i;
    csrVal[i] = (float) M.get(i);
  }
}

void ILUPPvec2csrMySpMatrixDouble(MySpMatrixDouble *mySpM, const Vector &M)
{
  mySpM->isCSR = 1;

  mySpM->numRows = M.dim();
  mySpM->numCols = M.dim();
  mySpM->numNZEntries = M.dim();

  mySpM->rowIndices=(int*)malloc( (mySpM->numRows + 1)*sizeof(int)); // Only first m+1 elements are useful on return.
  mySpM->indices=(int*)malloc( (mySpM->numNZEntries)*sizeof(int));
  mySpM->val=(double*)malloc( (mySpM->numNZEntries)*sizeof(double));
  int *csrRowPtr = mySpM->rowIndices;
  int *csrColIdx = mySpM->indices;
  double *csrVal = mySpM->val;

  for(int j=0; j<=mySpM->numRows; j++)
    csrRowPtr[j] = j;
  for(int i=0; i<mySpM->numNZEntries; i++) {
    csrColIdx[i] = i;
    csrVal[i] = M.get(i);
  }
}

void mna_solve_gpu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info,
                         char *ir_name, gpuETBR *myGPUetbr)
{
  printf("             mna_solve_gpu_gmres()\n");
  setGPUdevice();
  
  int useDoubleILU=0;

  Real_Timer interp2_run_time;
  Real_Timer ir_run_time;
  Real_Timer lufact_time;
  Real_Timer gmresCPUilu_time;
   
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, avg_ir; // min_ir, 
  int max_ir_idx; // , min_ir_idx
  ivec sorted_max_value_idx, sorted_min_value_idx, 
	sorted_avg_value_idx, sorted_ir_value_idx;
  int nNodes = tc_node.size();
  int display_num = num<tc_node.size()?num:tc_node.size();
  max_value.set_size(nNodes);
  min_value.set_size(nNodes);
  avg_value.set_size(nNodes);
  sorted_max_value_idx.set_size(nNodes);
  sorted_min_value_idx.set_size(nNodes);
  sorted_avg_value_idx.set_size(nNodes);
  sorted_ir_value_idx.set_size(nNodes);
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vector<int> const_v, const_i, var_v, var_i;
  for(int j = 0; j < nVS; j++){
	if (VS[j].time.size() == 1)
	  const_v.push_back(j);
	else
	  var_v.push_back(j);
  }
  for(int j = 0; j < nIS; j++){
	if (IS[j].time.size() == 1)
	  const_i.push_back(j);
	else
	  var_i.push_back(j);
  }
  /* DC simulation */
  for(vector<int>::iterator it = const_v.begin(); it != const_v.end(); ++it){
	u_col(*it) = VS[*it].value(0);
  }
  for(vector<int>::iterator it = const_i.begin(); it != const_i.end(); ++it){
	u_col(nVS+(*it)) = IS[*it].value(0);
  }
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
  vec x(n);
  x.zeros();

  // ----------- LU part start ----------------
  // cs_dls *SymbolicG, *SymbolicA;
  // cs_dln *NumericG, *NumericA;
  // int order = 2;
  // double tol = 1e-10; // XXLiu: was 1e-14
  // lufact_time.start();
  // SymbolicG = cs_dl_sqr(order, G, 0);
  // NumericG = cs_dl_lu(G, SymbolicG, tol);
  // lufact_time.stop();
  // cs_dl_ipvec(NumericG->pinv, w._data(), x._data(), n);
  // cs_dl_lsolve(NumericG->L, x._data());
  // cs_dl_usolve(NumericG->U, x._data());
  // cs_dl_ipvec(SymbolicG->q, x._data(), xres._data(), n);
  // for (int j = 0; j < port.size(); j++)  sim_port_value.set(j, 0, xres(port(j)));
  // ----------- LU part finish ----------------

  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xres(tc_node[j]);
	  min_value(j) = xres(tc_node[j]);
	  avg_value(j) = xres(tc_node[j]);
	}
	ir_run_time.stop();
  }

  /* Transient simulation */
  cs_dl *right = cs_dl_spalloc(C->m, C->n, C->nzmax, 1, 0);
  for (UF_long i = 0; i < C->n+1; i++){
	right->p[i] = C->p[i];
  }
  for (UF_long i = 0; i < C->nzmax; i++){
	right->i[i] = C->i[i];
	right->x[i] = 1/tstep*C->x[i];
  }
  cs_dl *left = cs_dl_add(G, right, 1, 1);
  // ----------- LU part start ----------------
  // lufact_time.start();
  // SymbolicA = cs_dl_sqr(order, left, 0);
  // NumericA = cs_dl_lu(left, SymbolicA, tol);
  // lufact_time.stop();
  // ----------- LU part finish ----------------

  cout << endl;
  // cout<<"*****************************************************************************"<<endl;
  // cout<<" ILU++ "<<endl;
  // cout<<"*****************************************************************************"<<endl;
  iluplusplus::preprocessing_sequence L;
  L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
  cout<<"Preprocessing selected:"<<endl;
  L.print();
  cout<<endl;
  iluplusplus::iluplusplus_precond_parameter param;
  param.init(L,11,"test"); // setup some other default values. Has no effect on preprocessing (except choice of PQ-Algorithm)
  // if(n > 1000000) {
  //   threshold = 1.7;//8
  //   factor = 1.0;
  // }
  param.set_threshold(threshold);
  param.set_MEM_FACTOR(factor);
  param.set_MAX_LEVELS(1);

  printf("    threshold=%f,  factor=%f\n", threshold, factor);
  
  //Matrix Arow;
  ucr_cs_di Gcs_di, Acs_di;
  Gcs_di.convertFromCS_DL(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  Acs_di.convertFromCS_DL(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  Matrix Gcol(Gcs_di.x, Gcs_di.i, Gcs_di.p, Gcs_di.m, Gcs_di.n, iluplusplus::COLUMN);
  Matrix Acol(Acs_di.x, Acs_di.i, Acs_di.p, Acs_di.m, Acs_di.n, iluplusplus::COLUMN);
  iluplusplus::multilevelILUCDPPreconditioner<Real,Matrix,Vector> PrG, PrA;
  PrG.make_preprocessed_multilevelILUCDP(Gcol,param);
  Real_Timer precondBuild_time;
  precondBuild_time.start();
  PrA.make_preprocessed_multilevelILUCDP(Acol,param);
  precondBuild_time.stop();
  cout<<"Information on the preconditioner G:"<<endl;
  PrG.print_info();
  cout<<"fill-in: "<<((Real) PrG.total_nnz())/(Real)Gcol.non_zeroes()<<endl;
  cout<<"Information on the preconditioner A:"<<endl;
  PrA.print_info();
  cout<<"fill-in: "<<((Real) PrA.total_nnz())/(Real)Acol.non_zeroes()<<endl;
  
  int min_iter =0;
  int reach_max_iter = 6000;
  Real reach_rel_tol = 3.0;
  Real reach_abs_tol = 3.0;
  
  Real rel_tol =  reach_rel_tol;
  Real abs_tol = reach_abs_tol;
  int max_iter = reach_max_iter;
  int restart = 32;
  
  cout<<"*****************************************************************************"<<endl;
  cout<<"GMRES GPU"<<endl;
  cout<<"*****************************************************************************"<<endl;
  Vector b(n), xgmres(n); // xbicgstab,xcgs,
  //gmresCPUilu_time.start();
  // xgmres.resize(Acol.rows(),0.0);
  // b.value( w._data(), 0, n );
  // iluplusplus::gmres<Real,Matrix,Vector>
  //   (PrG,iluplusplus::SPLIT,Gcol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
  //gmresCPUilu_time.stop();
  // cout<<"Iterations "<<max_iter<<endl;
  // //cout<<"error: "<<(xgmres-x_exact).norm_max()<<endl<<flush;
  // cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
  // cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;// cout<<endl;
  // for (int j = 0; j < port.size(); j++)  sim_port_value.set(j, 0, xgmres[port(j)]);


  ucr_cs_dl leftUCR, rightUCR, G_UCR, B_UCR;
  leftUCR.shallowCpy(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  rightUCR.shallowCpy(right->nzmax, right->m, right->n, right->p, right->i, right->x, right->nz);
  G_UCR.shallowCpy(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  B_UCR.shallowCpy(B->nzmax, B->m, B->n, B->p, B->i, B->x, B->nz);
  MySpMatrix GmySpM, //PrLeft_GmySpM, PrRight_GmySpM,
    PrMiddle_GmySpM,
    PrPermRow_GmySpM, PrPermCol_GmySpM;
  LDcsc2csrMySpMatrix( &GmySpM, &G_UCR );

  MySpMatrixDouble GmySpMdouble, AmySpMdouble;
  LDcsc2csrMySpMatrixDouble( &GmySpMdouble, &G_UCR );
  LDcsc2csrMySpMatrixDouble( &AmySpMdouble, &leftUCR );
  //writeCSRmySpMatrixDouble(&GmySpMdouble, "csrFileGdouble.dat");

  // Test the ILU
  /////////////////////////////////////////////////////
  Matrix PrecondG_left, PrecondG_right;
  Vector PrecondG_middle, PrecondG_lscale, PrecondG_rscale;
  iluplusplus::index_list PrecondG_perm_rows, PrecondG_perm_columns;
  PrecondG_left = PrG.extract_left_matrix(0);
  PrecondG_right = PrG.extract_right_matrix(0);
  PrecondG_middle = PrG.extract_middle_matrix(0);
  PrecondG_perm_rows = PrG.extract_permutation_rows(0);
  PrecondG_perm_columns = PrG.extract_inverse_permutation_columns(0);
  PrecondG_lscale = PrG.extract_left_scaling(0);
  PrecondG_rscale = PrG.extract_right_scaling(0);

  ILUPPvec2csrMySpMatrix( &PrMiddle_GmySpM, PrecondG_middle );
  index_list2csrMySpMatrix( &PrPermRow_GmySpM, PrecondG_perm_rows, n);
  index_list2csrMySpMatrix( &PrPermCol_GmySpM, PrecondG_perm_columns, n);  

  MySpMatrixDouble PrLeft_GmySpMdouble, PrRight_GmySpMdouble,
    PrLscale_GmySpMdouble, PrRscale_GmySpMdouble;
  MySpMatrix PrLscale_GmySpM, PrRscale_GmySpM;
  ILUPPmat2csrMySpMatrixDouble( &PrLeft_GmySpMdouble, PrecondG_left );
  ILUPPmat2csrMySpMatrixDouble( &PrRight_GmySpMdouble, PrecondG_right ); // UPPER_TRIANGULAR ROW ID
  // writeCSRmySpMatrix( &PrPermRow_GmySpM, "PrG_perm_row_mySpM.dat" );
  // writeCSRmySpMatrix( &PrPermCol_GmySpM, "PrG_perm_col_mySpM.dat" );

  if(useDoubleILU == 1) {
    ILUPPvec2csrMySpMatrixDouble( &PrLscale_GmySpMdouble, PrecondG_lscale);
    ILUPPvec2csrMySpMatrixDouble( &PrRscale_GmySpMdouble, PrecondG_rscale);
    // writeCSRmySpMatrix( &PrLscale_GmySpM, "PrG_lscale_mySpM.dat" );
    // writeCSRmySpMatrix( &PrRscale_GmySpM, "PrG_rscale_mySpM.dat" );
  }
  else {
    ILUPPvec2csrMySpMatrix( &PrLscale_GmySpM, PrecondG_lscale);
    ILUPPvec2csrMySpMatrix( &PrRscale_GmySpM, PrecondG_rscale);
  }


  // MySpMatrix PrLeft_GmySpM, PrRight_GmySpM,
  // // ILUPPmat2csrMySpMatrix( &PrLeft_GmySpM, PrecondG_left );
  // // ILUPPmat2csrMySpMatrix( &PrRight_GmySpM, PrecondG_right ); // UPPER_TRIANGULAR ROW ID
  // ILUPPveccsc2csrMySpMatrix( &PrLscale_GmySpM, PrecondG_lscale);
  // ILUPPveccsc2csrMySpMatrix( &PrRscale_GmySpM, PrecondG_rscale);
  // //writeCSRmySpMatrix( &PrLeft_GmySpM, "PrG_left_mySpM.dat" );
  // //writeCSRmySpMatrix( &PrRight_GmySpM, "PrG_right_mySpM.dat" );

  //////////////////////////////////////////////////
  Matrix PrecondA_left, PrecondA_right;
  Vector PrecondA_middle, PrecondA_lscale, PrecondA_rscale;
  iluplusplus::index_list PrecondA_perm_rows, PrecondA_perm_columns;
  PrecondA_left = PrA.extract_left_matrix(0);
  PrecondA_right = PrA.extract_right_matrix(0);
  PrecondA_middle = PrA.extract_middle_matrix(0);
  PrecondA_perm_rows = PrA.extract_permutation_rows(0);
  PrecondA_perm_columns = PrA.extract_inverse_permutation_columns(0);
  PrecondA_lscale = PrA.extract_left_scaling(0);
  PrecondA_rscale = PrA.extract_right_scaling(0);
  
  MySpMatrix AmySpM, PrMiddle_AmySpM, PrPermRow_AmySpM, PrPermCol_AmySpM;
  ILUPPvec2csrMySpMatrix( &PrMiddle_AmySpM, PrecondA_middle );
  index_list2csrMySpMatrix( &PrPermRow_AmySpM, PrecondA_perm_rows, n);
  index_list2csrMySpMatrix( &PrPermCol_AmySpM, PrecondA_perm_columns, n);

  LDcsc2csrMySpMatrix( &AmySpM, &leftUCR );

  MySpMatrixDouble PrLeft_AmySpMdouble, PrRight_AmySpMdouble,
    PrLscale_AmySpMdouble, PrRscale_AmySpMdouble;
  MySpMatrix PrLeft_AmySpM, PrRight_AmySpM, PrLscale_AmySpM, PrRscale_AmySpM;
  if(useDoubleILU == 1) {
    ILUPPmat2csrMySpMatrixDouble( &PrLeft_AmySpMdouble, PrecondA_left );
    ILUPPmat2csrMySpMatrixDouble( &PrRight_AmySpMdouble, PrecondA_right ); // UPPER_TRIANGULAR ROW ID
    ILUPPvec2csrMySpMatrixDouble( &PrLscale_AmySpMdouble, PrecondA_lscale);
    ILUPPvec2csrMySpMatrixDouble( &PrRscale_AmySpMdouble, PrecondA_rscale);
  }
  else {
    ILUPPmat2csrMySpMatrixDouble( &PrLeft_AmySpMdouble, PrecondA_left );
    ILUPPmat2csrMySpMatrixDouble( &PrRight_AmySpMdouble, PrecondA_right );
    // ILUPPmat2csrMySpMatrix( &PrLeft_AmySpM, PrecondA_left );
    // ILUPPmat2csrMySpMatrix( &PrRight_AmySpM, PrecondA_right ); // UPPER_TRIANGULAR ROW ID
    ILUPPvec2csrMySpMatrix( &PrLscale_AmySpM, PrecondA_lscale);
    ILUPPvec2csrMySpMatrix( &PrRscale_AmySpM, PrecondA_rscale);
  }

  // int *ptr=(int*)malloc((n+1)*sizeof(int));
  // int nnz = PrecondG_left.get_pointer(n);
  // int *idx=(int*)malloc(nnz*sizeof(int));
  // float *val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_left.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_left.get_index(i);
  //   val[i] = PrecondG_left.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_left_mySpM.dat");
  //free(idx); free(val);
  
  // nnz = PrecondG_right.get_pointer(n);
  // idx=(int*)malloc(nnz*sizeof(int));
  // val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_right.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_right.get_index(i);
  //   val[i] = PrecondG_right.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_right_mySpM.dat");
  
  // for(int i=0; i<=n; i++)  ptr[i] = i;
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = i;
  //   val[i] = PrecondG_middle.get(i);
  // }
  // writeCSR(n, n, n, ptr, idx, val, "PrG_middle_mySpM.dat");
  // free(idx); free(val);
  // Test finished

  gmresInterfacePG GmyInterfacePG, AmyInterfacePG;
  gmresInterfacePGfloat GmyInterfacePGfloat, AmyInterfacePGfloat;
  if(useDoubleILU == 1) {
    GmyInterfacePG.setPrecondPG
      (&GmySpM, &PrLeft_GmySpMdouble, &PrRight_GmySpMdouble, &PrMiddle_GmySpM,
       &PrPermRow_GmySpM, &PrPermCol_GmySpM, &PrLscale_GmySpMdouble, &PrRscale_GmySpMdouble);
    AmyInterfacePG.setPrecondPG
      (&AmySpM, &PrLeft_AmySpMdouble, &PrRight_AmySpMdouble, &PrMiddle_AmySpM,
       &PrPermRow_AmySpM, &PrPermCol_AmySpM, &PrLscale_AmySpMdouble, &PrRscale_AmySpMdouble);
  }
  else {
    GmyInterfacePGfloat.setPrecondPG
      ( &GmySpM, &PrLeft_GmySpMdouble, &PrRight_GmySpMdouble, &PrMiddle_GmySpM,
        &PrPermRow_GmySpM, &PrPermCol_GmySpM, &PrLscale_GmySpM, &PrRscale_GmySpM );
    AmyInterfacePGfloat.setPrecondPG
      ( &AmySpM, &PrLeft_AmySpMdouble, &PrRight_AmySpMdouble, &PrMiddle_AmySpM,
        &PrPermRow_AmySpM, &PrPermCol_AmySpM, &PrLscale_AmySpM, &PrRscale_AmySpM );
  }
  for(int i=0; i<n; i++) {
    GmyInterfacePGfloat.xgmres_h[i] = 0.0;
    GmyInterfacePGfloat.rhs_h[i] = *(w._data()+i); // 1.0
  }
  printf("DC simulation:  ");
  gmresCPUilu_time.start();
  GmyInterfacePGfloat.GMRES_dev_PG();
  gmresCPUilu_time.stop();
  cout<<"Iterations: "<< GmyInterfacePGfloat.max_it
      <<"  Residual: "<< GmyInterfacePGfloat.tol
      <<"  Time: " << gmresCPUilu_time.get_time() << endl;
  gmresCPUilu_time.reset();
  for(int j = 0; j < port.size(); j++)
    sim_port_value.set(j, 0, GmyInterfacePGfloat.xgmres_h[port(j)]);

  
  if(useDoubleILU == 1)
    for(int j=0; j<n; j++)  AmyInterfacePG.xgmres_h[j] = GmyInterfacePG.xgmres_h[j];
  else
    for(int j=0; j<n; j++)  AmyInterfacePGfloat.xgmres_h[j] = GmyInterfacePGfloat.xgmres_h[j];

  int iterTotal=0;
  /* GMRES solver part finishes. */
  
  vec xn(n), xnr(n), xn1(n), xn1t(n);
  // ----------- LU part start ----------------
  //xn = xres;
  // ----------- LU part finish ----------------
  //for(int j=0; j<n; j++)  xn._data()[j] = GmyInterfacePG.xgmres_h[j];
  //for(int j=0; j<n; j++)  xn._data()[j] = xgmres[j]; // for ILU++ gmres
  
  if(useDoubleILU == 1)
    for(int j=0; j<n; j++)  xn._data()[j] = GmyInterfacePG.xgmres_h[j]; // for UCRilu gmres
  else
    for(int j=0; j<n; j++)  xn._data()[j] = GmyInterfacePGfloat.xgmres_h[j]; // for UCRilu gmres
    
  xn1.zeros();
  xn1t.zeros();
  printf("   ts.size() = %d.\n",ts.size());
  for (int i = 1; i < ts.size(); i++){//
        /*
        for(int j = 0; j < nVS; j++){
          interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
          u_col(j) = temp;
        }
        for(int j = 0; j < nIS; j++){
          interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
          u_col(nVS+j) = temp;
        }
        */
        interp2_run_time.start();
        for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
          interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
          u_col(*it) = temp;
        }
        for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
          interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
          u_col(nVS+(*it)) = temp;
        }
        interp2_run_time.stop();
  
        w.zeros();
        cs_dl_gaxpy(B, u_col._data(), w._data());
        xnr.zeros();
        // cs_dl_gaxpy(C, xn._data(), xnr._data());
        // w += 1/tstep*xnr;
        cs_dl_gaxpy(right, xn._data(), xnr._data());
        w += xnr;
        
        //-----------------------------
        // rel_tol =  reach_rel_tol;
        // abs_tol = reach_abs_tol;
        // max_iter = reach_max_iter;
        // restart = 100;
        // b.value( w._data(), 0, n );
        // iluplusplus::gmres<Real,Matrix,Vector>(PrA,iluplusplus::SPLIT,Acol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
        // for(int j=0; j<n; j++)  xn1._data()[j] = xgmres[j];
        // cout<<"Iterations "<<max_iter
        //     <<" rel resid "<<exp(-rel_tol*log(10.0))
        //     <<" abs resid "<<exp(-abs_tol*log(10.0))<<endl;
        // iterTotal += max_iter;
        //-----------------------------
        
        if(useDoubleILU == 1) {
          for(int j=0; j<n; j++)  AmyInterfacePG.rhs_h[j] = *(w._data()+j);
          //for(int j=0; j<n; j++)  AmyInterfacePG.xgmres_h[j] = 0.0;
          AmyInterfacePG.GMRES_host_PG();
          for(int j=0; j<n; j++)  xn1._data()[j] = AmyInterfacePG.xgmres_h[j];
        }
        else {
          for(int j=0; j<n; j++)  AmyInterfacePGfloat.rhs_h[j] = *(w._data()+j);
          gmresCPUilu_time.start();
          AmyInterfacePGfloat.GMRES_dev_PG();
          gmresCPUilu_time.stop();
          for(int j=0; j<n; j++)  xn1._data()[j] = AmyInterfacePGfloat.xgmres_h[j];
          iterTotal += AmyInterfacePGfloat.max_it;
        }
        //-----------------------------
  
        // ----------- LU part start ----------------
        // cs_dl_ipvec(NumericA->pinv, w._data(), xn1t._data(), n);
        // cs_dl_lsolve(NumericA->L, xn1t._data());
        // cs_dl_usolve(NumericA->U, xn1t._data());
        // cs_dl_ipvec(SymbolicA->q, xn1t._data(), xn1._data(), n);  
        // ----------- LU part finish ----------------
        for (int j = 0; j < port.size(); j++)
          sim_port_value.set(j, i, xn1(port(j)));

        if (ir_info){
          ir_run_time.start();
          for (int j = 0; j < nNodes; j++){
        	if (max_value(j) < xn1(tc_node[j])){
        	  max_value(j) = xn1(tc_node[j]);
        	}
        	if (xn1(tc_node[j]) < min_value(j)){
        	  min_value(j) = xn1(tc_node[j]);
        	}
        	avg_value(j) += xn1(tc_node[j]);
          }
          ir_run_time.stop();
        }
        xn = xn1;
  }
  //------------------------------------------

  int nport = port.size();
  int *invPort=(int*)malloc(nport*sizeof(int));
  for (int j = 0; j < port.size(); j++){
    //sim_port_value.set(j, 0, xgmres[port(j)]);
    // printf("port index: %d\n",port(j));
    invPort[j] = port(j);
  }


  /********************* GPU preparation ******************************/
  Real_Timer GPUtime;
  printf("*** GPU accelerated transient simulation ***\n");
  myGPUetbr->n = B->m;
  myGPUetbr->m = B->n;
  myGPUetbr->numPts = ts.size();
  myGPUetbr->ldUt = (((myGPUetbr->numPts-1) +31)/32)*32;
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;
  myGPUetbr->tstep = tstep;
  myGPUetbr->tstop = tstop;

  if(myGPUetbr->PWLcurExist && myGPUetbr->PULSEcurExist) {
    printf("                    Do not support PWL and PULSE current sources at the same time.\n");
    while( !getchar() ) ;
  }
  if(myGPUetbr->PWLcurExist) {
    if(myGPUetbr->PWLcurExist == nIS)
      printf("       All PWL current sources.\n");
    else {
      printf("       Error: There are non-PWL current sources mingled with PWL.\n");
      while(!getchar()) ;
    }
  }
  if(myGPUetbr->PULSEcurExist) {
    if(myGPUetbr->PULSEcurExist == nIS)
      printf("       All PULSE current sources.\n");
    else {
      printf("       Error: There are non-PULSE current sources mingled with PULSE.\n");
      while(!getchar()) ;
    }
  }
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;

  if(myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m) {
    printf("                    myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m\n");
    while( !getchar() ) ;
  }

  if(myGPUetbr->PWLvolExist || myGPUetbr->PULSEvolExist) {
    printf("                    Do not support PWL and PULSE voltage sources.\n");
    while( !getchar() ) ;
  }

  
  /* The following section need CPU generated source info. */

  if(BLK_SIZE_UTGEN < MAX_PWL_PTS) {
    printf("       Error: BLK_SIZE_UTGEN should be no less than MAX_PWL_PTS\n");
    while(!getchar()) ;
  }

  /* The following section prepares the source info in order to use GPU evaluation. */
  // store DC voltage source info.
  myGPUetbr->dcVt_host=(double*)malloc(nVS*sizeof(double));
  for(int i=0; i<nVS; i++)
    myGPUetbr->dcVt_host[i] = VS[i].value(0);

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    myGPUetbr->dcVt_single_host=(float*)malloc(nVS*sizeof(float));
    myMemcpyD2S(myGPUetbr->dcVt_single_host,  myGPUetbr->dcVt_host, nVS);
  }

  if(myGPUetbr->PWLcurExist) {
    myGPUetbr->PWLnumPts_host=(int*)malloc(nIS*sizeof(int));
    myGPUetbr->PWLtime_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    myGPUetbr->PWLval_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    for(int i=0; i<nIS; i++) {
      int herePWLnumPts=IS[i].time.size();
      //printf(" size: %d\n",herePWLnumPts);
      if(herePWLnumPts > MAX_PWL_PTS) {
	printf("       Error: More PWL points than allowed. %d > %d at source-%d\n",herePWLnumPts, MAX_PWL_PTS, i);
	while(!getchar()) ;
      }
      myGPUetbr->PWLnumPts_host[i] = herePWLnumPts;
      for(int j=0; j<herePWLnumPts; j++) {
	myGPUetbr->PWLtime_host[i*MAX_PWL_PTS+j] = IS[i].time(j);
	myGPUetbr->PWLval_host[i*MAX_PWL_PTS+j] = IS[i].value(j);
      }
    }

    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      myGPUetbr->PWLtime_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myGPUetbr->PWLval_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myMemcpyD2S(myGPUetbr->PWLtime_single_host, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS);
      myMemcpyD2S(myGPUetbr->PWLval_single_host, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS);
    }
  }

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    if(nport > 0)
      myGPUetbr->x_single_host=(float*)malloc(nport*(myGPUetbr->numPts)*sizeof(float));
  }

  // if(nport > 0)
  //   wrapperGMRESforPG(&leftUCR, &rightUCR, &G_UCR, &B_UCR, //w._data(),
  //                     invPort, nport, myGPUetbr);
  // for(int i=0; i<nport; i++)
  //   for(int j=0; j<ts.size(); j++)
  //     sim_port_value.set(i, j, (double)myGPUetbr->x_single_host[i+j*nport]);

  free(invPort);
  
  free(myGPUetbr->dcVt_host); // if(myGPUetbr->dcVt_host != NULL)
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION  
    free(myGPUetbr->dcVt_single_host); // if(myGPUetbr->dcVt_single_host != NULL)
    if(nport > 0)
      free(myGPUetbr->x_single_host); //if(myGPUetbr->x_single_host != NULL)
  }

  //------------------------------------------

  cs_dl_spfree(left);
  cs_dl_spfree(right);
  // cs_dl_sfree(SymbolicA);
  // cs_dl_nfree(NumericA);
  // cs_dl_sfree(SymbolicG);
  // cs_dl_nfree(NumericG);
  delete [] cur;

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	/*
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	*/
	ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	ir_run_time.stop();
  }

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation2  \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "IR analysis     \t: " << ir_run_time.get_time() << std::endl;
  std::cout << "LU factorization\t: " << lufact_time.get_time() << std::endl;
  std::cout << "ILU++ construct \t: " << precondBuild_time.get_time() << std::endl;
  std::cout << "ILU++ GMRES GPU \t: " << gmresCPUilu_time.get_time()
            << "    Avg iter per point: " << (int)ceil(1.0*iterTotal/ts.size())
            << "    Time per point: " << gmresCPUilu_time.get_time() / ts.size() << std::endl;

  mySpMatrixFree(&GmySpM); //PrLeft_GmySpM, PrRight_GmySpM,
  mySpMatrixFree(&PrMiddle_GmySpM);
  mySpMatrixFree(&PrPermRow_GmySpM);
  mySpMatrixFree(&PrPermCol_GmySpM);

  mySpMatrixDoubleFree(&PrLeft_GmySpMdouble);
  mySpMatrixDoubleFree(&PrRight_GmySpMdouble);
  mySpMatrixDoubleFree(&PrLscale_GmySpMdouble);
  mySpMatrixDoubleFree(&PrRscale_GmySpMdouble);

  mySpMatrixFree(&AmySpM);
  mySpMatrixFree(&PrMiddle_AmySpM);
  mySpMatrixFree(&PrPermRow_AmySpM);
  mySpMatrixFree(&PrPermCol_AmySpM);

  mySpMatrixDoubleFree(&PrLeft_AmySpMdouble);
  mySpMatrixDoubleFree(&PrRight_AmySpMdouble);
  mySpMatrixDoubleFree(&PrLscale_AmySpMdouble);
  mySpMatrixDoubleFree(&PrRscale_AmySpMdouble);
}

///////////////////////////////////////////////////////////////////////////

void mna_solve_cpu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info,
                         char *ir_name, gpuETBR *myGPUetbr)
{
  printf("             mna_solve_cpu_gmres()\n");
  Real_Timer interp2_run_time;
  Real_Timer ir_run_time;
  Real_Timer lufact_time;
  Real_Timer gmresCPUilu_time;
   
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, avg_ir; // min_ir, 
  int max_ir_idx; // , min_ir_idx
  ivec sorted_max_value_idx, sorted_min_value_idx, 
	sorted_avg_value_idx, sorted_ir_value_idx;
  int nNodes = tc_node.size();
  int display_num = num<tc_node.size()?num:tc_node.size();
  max_value.set_size(nNodes);
  min_value.set_size(nNodes);
  avg_value.set_size(nNodes);
  sorted_max_value_idx.set_size(nNodes);
  sorted_min_value_idx.set_size(nNodes);
  sorted_avg_value_idx.set_size(nNodes);
  sorted_ir_value_idx.set_size(nNodes);
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vector<int> const_v, const_i, var_v, var_i;
  for(int j = 0; j < nVS; j++){
	if (VS[j].time.size() == 1)
	  const_v.push_back(j);
	else
	  var_v.push_back(j);
  }
  for(int j = 0; j < nIS; j++){
	if (IS[j].time.size() == 1)
	  const_i.push_back(j);
	else
	  var_i.push_back(j);
  }
  /* DC simulation */
  for(vector<int>::iterator it = const_v.begin(); it != const_v.end(); ++it){
	u_col(*it) = VS[*it].value(0);
  }
  for(vector<int>::iterator it = const_i.begin(); it != const_i.end(); ++it){
	u_col(nVS+(*it)) = IS[*it].value(0);
  }
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
  vec x(n);
  x.zeros();

  // ----------- LU part start ----------------
  // cs_dls *SymbolicG, *SymbolicA;
  // cs_dln *NumericG, *NumericA;
  // int order = 2;
  // double tol = 1e-10; // XXLiu: was 1e-14
  // lufact_time.start();
  // SymbolicG = cs_dl_sqr(order, G, 0);
  // NumericG = cs_dl_lu(G, SymbolicG, tol);
  // lufact_time.stop();
  // cs_dl_ipvec(NumericG->pinv, w._data(), x._data(), n);
  // cs_dl_lsolve(NumericG->L, x._data());
  // cs_dl_usolve(NumericG->U, x._data());
  // cs_dl_ipvec(SymbolicG->q, x._data(), xres._data(), n);
  // for (int j = 0; j < port.size(); j++)  sim_port_value.set(j, 0, xres(port(j)));
  // ----------- LU part finish ----------------

  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xres(tc_node[j]);
	  min_value(j) = xres(tc_node[j]);
	  avg_value(j) = xres(tc_node[j]);
	}
	ir_run_time.stop();
  }

  /* Transient simulation */
  cs_dl *right = cs_dl_spalloc(C->m, C->n, C->nzmax, 1, 0);
  for (UF_long i = 0; i < C->n+1; i++){
	right->p[i] = C->p[i];
  }
  for (UF_long i = 0; i < C->nzmax; i++){
	right->i[i] = C->i[i];
	right->x[i] = 1/tstep*C->x[i];
  }
  cs_dl *left = cs_dl_add(G, right, 1, 1);
  // ----------- LU part start ----------------
  // lufact_time.start();
  // SymbolicA = cs_dl_sqr(order, left, 0);
  // NumericA = cs_dl_lu(left, SymbolicA, tol);
  // lufact_time.stop();
  // ----------- LU part finish ----------------


  cout<<"*****************************************************************************"<<endl;
  cout<<" ILU++ "<<endl;
  cout<<"*****************************************************************************"<<endl;
  iluplusplus::preprocessing_sequence L;
  L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM();
  cout<<"Preprocessing selected:"<<endl;
  L.print();
  cout<<endl;
  iluplusplus::iluplusplus_precond_parameter param;
  param.init(L,11,"test"); // setup some other default values. Has no effect on preprocessing (except choice of PQ-Algorithm)
  //double threshold=3.0, factor=1.0; // threshold=3.0
  param.set_threshold(threshold);
  param.set_MEM_FACTOR(factor);
  param.set_MAX_LEVELS(1);
  
  printf("    threshold=%f,  factor=%f\n", threshold, factor);
  
  //Matrix Arow;
  ucr_cs_di Gcs_di, Acs_di;
  Gcs_di.convertFromCS_DL(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  Acs_di.convertFromCS_DL(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  Matrix Gcol(Gcs_di.x, Gcs_di.i, Gcs_di.p, Gcs_di.m, Gcs_di.n, iluplusplus::COLUMN);
  Matrix Acol(Acs_di.x, Acs_di.i, Acs_di.p, Acs_di.m, Acs_di.n, iluplusplus::COLUMN);
  iluplusplus::multilevelILUCDPPreconditioner<Real,Matrix,Vector> PrG, PrA;
  PrG.make_preprocessed_multilevelILUCDP(Gcol,param);
  PrA.make_preprocessed_multilevelILUCDP(Acol,param);
  cout<<"Information on the preconditioner G:"<<endl;
  PrG.print_info();
  cout<<"fill-in: "<<((Real) PrG.total_nnz())/(Real)Gcol.non_zeroes()<<endl;
  cout<<"Information on the preconditioner A:"<<endl;
  PrA.print_info();
  cout<<"fill-in: "<<((Real) PrA.total_nnz())/(Real)Acol.non_zeroes()<<endl;
  
  int min_iter =0;
  int reach_max_iter = 100;
  Real reach_rel_tol = 3.0;
  Real reach_abs_tol = 3.0;
  
  Real rel_tol =  reach_rel_tol;
  Real abs_tol = reach_abs_tol;
  int max_iter = reach_max_iter;
  int restart = 100;
  
  cout<<"*****************************************************************************"<<endl;
  cout<<"GMRES CPU"<<endl;
  cout<<"*****************************************************************************"<<endl;
  Vector b(n), xgmres(n); // xbicgstab,xcgs,
  // gmresCPUilu_time.start();
  // xgmres.resize(Acol.rows(),0.0);
  // b.value( w._data(), 0, n );
  // iluplusplus::gmres<Real,Matrix,Vector>
  //   (PrG,iluplusplus::SPLIT,Gcol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
  // gmresCPUilu_time.stop();
  // cout<<"Iterations "<<max_iter<<endl;
  // //cout<<"error: "<<(xgmres-x_exact).norm_max()<<endl<<flush;
  // cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
  // cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;// cout<<endl;


  ucr_cs_dl leftUCR, rightUCR, G_UCR, B_UCR;
  leftUCR.shallowCpy(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  rightUCR.shallowCpy(right->nzmax, right->m, right->n, right->p, right->i, right->x, right->nz);
  G_UCR.shallowCpy(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  B_UCR.shallowCpy(B->nzmax, B->m, B->n, B->p, B->i, B->x, B->nz);
  MySpMatrix GmySpM, //PrLeft_GmySpM, PrRight_GmySpM,
    PrMiddle_GmySpM,
    PrPermRow_GmySpM, PrPermCol_GmySpM;
  LDcsc2csrMySpMatrix( &GmySpM, &G_UCR );

  // MySpMatrixDouble GmySpMdouble, AmySpMdouble;
  // LDcsc2csrMySpMatrixDouble( &GmySpMdouble, &G_UCR );
  // LDcsc2csrMySpMatrixDouble( &AmySpMdouble, &leftUCR );
  //writeCSRmySpMatrixDouble(&GmySpMdouble, "csrFileGdouble.dat");

  // Test the ILU
  /////////////////////////////////////////////////////
  Matrix PrecondG_left, PrecondG_right;
  Vector PrecondG_middle, PrecondG_lscale, PrecondG_rscale;
  iluplusplus::index_list PrecondG_perm_rows, PrecondG_perm_columns;
  PrecondG_left = PrG.extract_left_matrix(0);
  PrecondG_right = PrG.extract_right_matrix(0);
  PrecondG_middle = PrG.extract_middle_matrix(0);
  PrecondG_perm_rows = PrG.extract_permutation_rows(0);
  PrecondG_perm_columns = PrG.extract_inverse_permutation_columns(0);
  PrecondG_lscale = PrG.extract_left_scaling(0);
  PrecondG_rscale = PrG.extract_right_scaling(0);

  ILUPPvec2csrMySpMatrix( &PrMiddle_GmySpM, PrecondG_middle );
  index_list2csrMySpMatrix( &PrPermRow_GmySpM, PrecondG_perm_rows, n);
  index_list2csrMySpMatrix( &PrPermCol_GmySpM, PrecondG_perm_columns, n);  

  MySpMatrixDouble PrLeft_GmySpMdouble, PrRight_GmySpMdouble,
    PrLscale_GmySpMdouble, PrRscale_GmySpMdouble;
  ILUPPmat2csrMySpMatrixDouble( &PrLeft_GmySpMdouble, PrecondG_left );
  ILUPPmat2csrMySpMatrixDouble( &PrRight_GmySpMdouble, PrecondG_right ); // UPPER_TRIANGULAR ROW ID
  // writeCSRmySpMatrix( &PrPermRow_GmySpM, "PrG_perm_row_mySpM.dat" );
  // writeCSRmySpMatrix( &PrPermCol_GmySpM, "PrG_perm_col_mySpM.dat" );
  ILUPPvec2csrMySpMatrixDouble( &PrLscale_GmySpMdouble, PrecondG_lscale);
  ILUPPvec2csrMySpMatrixDouble( &PrRscale_GmySpMdouble, PrecondG_rscale);
  // writeCSRmySpMatrix( &PrLscale_GmySpM, "PrG_lscale_mySpM.dat" );
  // writeCSRmySpMatrix( &PrRscale_GmySpM, "PrG_rscale_mySpM.dat" );


  // MySpMatrix PrLeft_GmySpM, PrRight_GmySpM,
  // ILUPPmat2csrMySpMatrix( &PrLeft_GmySpM, PrecondG_left );
  // ILUPPmat2csrMySpMatrix( &PrRight_GmySpM, PrecondG_right ); // UPPER_TRIANGULAR ROW ID
  // ILUPPveccsc2csrMySpMatrix( &PrLscale_GmySpM, PrecondG_lscale);
  // ILUPPveccsc2csrMySpMatrix( &PrRscale_GmySpM, PrecondG_rscale);
  //writeCSRmySpMatrix( &PrLeft_GmySpM, "PrG_left_mySpM.dat" );
  //writeCSRmySpMatrix( &PrRight_GmySpM, "PrG_right_mySpM.dat" );

  //////////////////////////////////////////////////
  Matrix PrecondA_left, PrecondA_right;
  Vector PrecondA_middle, PrecondA_lscale, PrecondA_rscale;
  iluplusplus::index_list PrecondA_perm_rows, PrecondA_perm_columns;
  PrecondA_left = PrA.extract_left_matrix(0);
  PrecondA_right = PrA.extract_right_matrix(0);
  PrecondA_middle = PrA.extract_middle_matrix(0);
  PrecondA_perm_rows = PrA.extract_permutation_rows(0);
  PrecondA_perm_columns = PrA.extract_inverse_permutation_columns(0);
  PrecondA_lscale = PrA.extract_left_scaling(0);
  PrecondA_rscale = PrA.extract_right_scaling(0);
  
  MySpMatrix AmySpM, PrMiddle_AmySpM, PrPermRow_AmySpM, PrPermCol_AmySpM;
  ILUPPvec2csrMySpMatrix( &PrMiddle_AmySpM, PrecondA_middle );
  index_list2csrMySpMatrix( &PrPermRow_AmySpM, PrecondA_perm_rows, n);
  index_list2csrMySpMatrix( &PrPermCol_AmySpM, PrecondA_perm_columns, n);

  LDcsc2csrMySpMatrix( &AmySpM, &leftUCR );

  MySpMatrixDouble PrLeft_AmySpMdouble, PrRight_AmySpMdouble,
    PrLscale_AmySpMdouble, PrRscale_AmySpMdouble;
  ILUPPmat2csrMySpMatrixDouble( &PrLeft_AmySpMdouble, PrecondA_left );
  ILUPPmat2csrMySpMatrixDouble( &PrRight_AmySpMdouble, PrecondA_right ); // UPPER_TRIANGULAR ROW ID
  ILUPPvec2csrMySpMatrixDouble( &PrLscale_AmySpMdouble, PrecondA_lscale);
  ILUPPvec2csrMySpMatrixDouble( &PrRscale_AmySpMdouble, PrecondA_rscale);

  // ILUPPmat2csrMySpMatrix( &PrLeft_AmySpM, PrecondA_left );
  // ILUPPmat2csrMySpMatrix( &PrRight_AmySpM, PrecondA_right ); // UPPER_TRIANGULAR ROW ID

  // int *ptr=(int*)malloc((n+1)*sizeof(int));
  // int nnz = PrecondG_left.get_pointer(n);
  // int *idx=(int*)malloc(nnz*sizeof(int));
  // float *val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_left.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_left.get_index(i);
  //   val[i] = PrecondG_left.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_left_mySpM.dat");
  //free(idx); free(val);
  
  // nnz = PrecondG_right.get_pointer(n);
  // idx=(int*)malloc(nnz*sizeof(int));
  // val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_right.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_right.get_index(i);
  //   val[i] = PrecondG_right.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_right_mySpM.dat");
  
  // for(int i=0; i<=n; i++)  ptr[i] = i;
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = i;
  //   val[i] = PrecondG_middle.get(i);
  // }
  // writeCSR(n, n, n, ptr, idx, val, "PrG_middle_mySpM.dat");
  // free(idx); free(val);
  // Test finished

  gmresInterfacePG GmyInterfacePG, AmyInterfacePG;
  GmyInterfacePG.setPrecondPG(&GmySpM,
                              &PrLeft_GmySpMdouble, &PrRight_GmySpMdouble, &PrMiddle_GmySpM,
                              &PrPermRow_GmySpM, &PrPermCol_GmySpM,
                              &PrLscale_GmySpMdouble, &PrRscale_GmySpMdouble);
  AmyInterfacePG.setPrecondPG(&AmySpM,
                              &PrLeft_AmySpMdouble, &PrRight_AmySpMdouble, &PrMiddle_AmySpM,
                              &PrPermRow_AmySpM, &PrPermCol_AmySpM,
                              &PrLscale_AmySpMdouble, &PrRscale_AmySpMdouble);
  for(int i=0; i<n; i++) {
    GmyInterfacePG.xgmres_h[i] = 0.0;
    GmyInterfacePG.rhs_h[i] = *(w._data()+i); // 1.0
  }
  printf("DC simulation:  ");
  gmresCPUilu_time.start();
  GmyInterfacePG.GMRES_host_PG();
  gmresCPUilu_time.stop();
  cout<<"Iterations: "<< GmyInterfacePG.max_it
      <<"  Residual: "<< GmyInterfacePG.tol
      <<"  Time: " << gmresCPUilu_time.get_time() << endl;
  gmresCPUilu_time.reset();
  for(int j = 0; j < port.size(); j++)
    sim_port_value.set(j, 0, GmyInterfacePG.xgmres_h[port(j)]);

  for(int j=0; j<n; j++)  AmyInterfacePG.xgmres_h[j] = GmyInterfacePG.xgmres_h[j];
  // for (int j = 0; j < port.size(); j++)  sim_port_value.set(j, 0, GmyInterfacePG.xgmres_h[port(j)]);

  int iterTotal=0;
  /* GMRES solver part finishes. */
  
  vec xn(n), xnr(n), xn1(n), xn1t(n);
  // ----------- LU part start ----------------
  //xn = xres;
  // ----------- LU part finish ----------------
  //for(int j=0; j<n; j++)  xn._data()[j] = GmyInterfacePG.xgmres_h[j];
  //for(int j=0; j<n; j++)  xn._data()[j] = xgmres[j]; // for ILU++ gmres
  for(int j=0; j<n; j++)  xn._data()[j] = GmyInterfacePG.xgmres_h[j]; // for UCRilu gmres
  xn1.zeros();
  xn1t.zeros();
  printf("   ts.size() = %d.\n",ts.size());
  for (int i = 1; i < ts.size(); i++){//
        /*
        for(int j = 0; j < nVS; j++){
          interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
          u_col(j) = temp;
        }
        for(int j = 0; j < nIS; j++){
          interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
          u_col(nVS+j) = temp;
        }
        */
        interp2_run_time.start();
        for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
          interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
          u_col(*it) = temp;
        }
        for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
          interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
          u_col(nVS+(*it)) = temp;
        }
        interp2_run_time.stop();
  
        w.zeros();
        cs_dl_gaxpy(B, u_col._data(), w._data());
        xnr.zeros();
        // cs_dl_gaxpy(C, xn._data(), xnr._data());
        // w += 1/tstep*xnr;
        cs_dl_gaxpy(right, xn._data(), xnr._data());
        w += xnr;
        

        rel_tol =  reach_rel_tol;
        abs_tol = reach_abs_tol;
        max_iter = reach_max_iter;
        restart = 100;
        //-----------------------------
        // b.value( w._data(), 0, n );
        // iluplusplus::gmres<Real,Matrix,Vector>(PrA,iluplusplus::SPLIT,Acol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
        // for(int j=0; j<n; j++)  xn1._data()[j] = xgmres[j];
        // cout<<"Iterations "<<max_iter
        //     <<" rel resid "<<exp(-rel_tol*log(10.0))
        //     <<" abs resid "<<exp(-abs_tol*log(10.0))<<endl;
        // iterTotal += max_iter;
        //-----------------------------
        for(int j=0; j<n; j++)  AmyInterfacePG.rhs_h[j] = *(w._data()+j);
        //for(int j=0; j<n; j++)  AmyInterfacePG.xgmres_h[j] = 0.0;
        gmresCPUilu_time.start(); 
        AmyInterfacePG.GMRES_host_PG();
        gmresCPUilu_time.stop();
        iterTotal += AmyInterfacePG.max_it;
        for(int j=0; j<n; j++)  xn1._data()[j] = AmyInterfacePG.xgmres_h[j];
        //-----------------------------
        
  
        // ----------- LU part start ----------------
        // cs_dl_ipvec(NumericA->pinv, w._data(), xn1t._data(), n);
        // cs_dl_lsolve(NumericA->L, xn1t._data());
        // cs_dl_usolve(NumericA->U, xn1t._data());
        // cs_dl_ipvec(SymbolicA->q, xn1t._data(), xn1._data(), n);  
        // ----------- LU part finish ----------------
        for (int j = 0; j < port.size(); j++)
          sim_port_value.set(j, i, xn1(port(j)));

        if (ir_info){
          ir_run_time.start();
          for (int j = 0; j < nNodes; j++){
        	if (max_value(j) < xn1(tc_node[j])){
        	  max_value(j) = xn1(tc_node[j]);
        	}
        	if (xn1(tc_node[j]) < min_value(j)){
        	  min_value(j) = xn1(tc_node[j]);
        	}
        	avg_value(j) += xn1(tc_node[j]);
          }
          ir_run_time.stop();
        }
        xn = xn1;
  }
  //------------------------------------------

  int nport = port.size();
  int *invPort=(int*)malloc(nport*sizeof(int));
  for (int j = 0; j < port.size(); j++){
    //sim_port_value.set(j, 0, xgmres[port(j)]);
    // printf("port index: %d\n",port(j));
    invPort[j] = port(j);
  }


  /********************* GPU preparation ******************************/
  Real_Timer GPUtime;
  printf("*** GPU accelerated transient simulation ***\n");
  myGPUetbr->n = B->m;
  myGPUetbr->m = B->n;
  myGPUetbr->numPts = ts.size();
  myGPUetbr->ldUt = (((myGPUetbr->numPts-1) +31)/32)*32;
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;
  myGPUetbr->tstep = tstep;
  myGPUetbr->tstop = tstop;

  if(myGPUetbr->PWLcurExist && myGPUetbr->PULSEcurExist) {
    printf("                    Do not support PWL and PULSE current sources at the same time.\n");
    while( !getchar() ) ;
  }
  if(myGPUetbr->PWLcurExist) {
    if(myGPUetbr->PWLcurExist == nIS)
      printf("       All PWL current sources.\n");
    else {
      printf("       Error: There are non-PWL current sources mingled with PWL.\n");
      while(!getchar()) ;
    }
  }
  if(myGPUetbr->PULSEcurExist) {
    if(myGPUetbr->PULSEcurExist == nIS)
      printf("       All PULSE current sources.\n");
    else {
      printf("       Error: There are non-PULSE current sources mingled with PULSE.\n");
      while(!getchar()) ;
    }
  }
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;

  if(myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m) {
    printf("                    myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m\n");
    while( !getchar() ) ;
  }

  if(myGPUetbr->PWLvolExist || myGPUetbr->PULSEvolExist) {
    printf("                    Do not support PWL and PULSE voltage sources.\n");
    while( !getchar() ) ;
  }

  
  /* The following section need CPU generated source info. */

  if(BLK_SIZE_UTGEN < MAX_PWL_PTS) {
    printf("       Error: BLK_SIZE_UTGEN should be no less than MAX_PWL_PTS\n");
    while(!getchar()) ;
  }

  /* The following section prepares the source info in order to use GPU evaluation. */
  // store DC voltage source info.
  myGPUetbr->dcVt_host=(double*)malloc(nVS*sizeof(double));
  for(int i=0; i<nVS; i++)
    myGPUetbr->dcVt_host[i] = VS[i].value(0);

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    myGPUetbr->dcVt_single_host=(float*)malloc(nVS*sizeof(float));
    myMemcpyD2S(myGPUetbr->dcVt_single_host,  myGPUetbr->dcVt_host, nVS);
  }

  if(myGPUetbr->PWLcurExist) {
    myGPUetbr->PWLnumPts_host=(int*)malloc(nIS*sizeof(int));
    myGPUetbr->PWLtime_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    myGPUetbr->PWLval_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    for(int i=0; i<nIS; i++) {
      int herePWLnumPts=IS[i].time.size();
      //printf(" size: %d\n",herePWLnumPts);
      if(herePWLnumPts > MAX_PWL_PTS) {
	printf("       Error: More PWL points than allowed. %d > %d at source-%d\n",herePWLnumPts, MAX_PWL_PTS, i);
	while(!getchar()) ;
      }
      myGPUetbr->PWLnumPts_host[i] = herePWLnumPts;
      for(int j=0; j<herePWLnumPts; j++) {
	myGPUetbr->PWLtime_host[i*MAX_PWL_PTS+j] = IS[i].time(j);
	myGPUetbr->PWLval_host[i*MAX_PWL_PTS+j] = IS[i].value(j);
      }
    }

    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      myGPUetbr->PWLtime_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myGPUetbr->PWLval_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myMemcpyD2S(myGPUetbr->PWLtime_single_host, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS);
      myMemcpyD2S(myGPUetbr->PWLval_single_host, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS);
    }
  }

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    if(nport > 0)
      myGPUetbr->x_single_host=(float*)malloc(nport*(myGPUetbr->numPts)*sizeof(float));
  }

  // if(nport > 0)
  //   wrapperGMRESforPG(&leftUCR, &rightUCR, &G_UCR, &B_UCR, //w._data(),
  //                     invPort, nport, myGPUetbr);
  // for(int i=0; i<nport; i++)
  //   for(int j=0; j<ts.size(); j++)
  //     sim_port_value.set(i, j, (double)myGPUetbr->x_single_host[i+j*nport]);

  free(invPort);
  
  free(myGPUetbr->dcVt_host); // if(myGPUetbr->dcVt_host != NULL)
  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION  
    free(myGPUetbr->dcVt_single_host); // if(myGPUetbr->dcVt_single_host != NULL)
    if(nport > 0)
      free(myGPUetbr->x_single_host); //if(myGPUetbr->x_single_host != NULL)
  }

  //------------------------------------------

  cs_dl_spfree(left);
  cs_dl_spfree(right);
  // cs_dl_sfree(SymbolicA);
  // cs_dl_nfree(NumericA);
  // cs_dl_sfree(SymbolicG);
  // cs_dl_nfree(NumericG);
  delete [] cur;

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	/*
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	*/
	ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	ir_run_time.stop();
  }

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation2  \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "IR analysis     \t: " << ir_run_time.get_time() << std::endl;
  //std::cout << "LU factorization\t: " << lufact_time.get_time() << std::endl;
  std::cout << "ILU++ GMRES CPU \t: " << gmresCPUilu_time.get_time()
            << "    Avg iter per point: " << (int)ceil(1.0*iterTotal/ts.size())
            << "    Time per point: " << gmresCPUilu_time.get_time() / ts.size() << std::endl;
  mySpMatrixFree(&GmySpM); //PrLeft_GmySpM, PrRight_GmySpM,
  mySpMatrixFree(&PrMiddle_GmySpM);
  mySpMatrixFree(&PrPermRow_GmySpM);
  mySpMatrixFree(&PrPermCol_GmySpM);

  mySpMatrixDoubleFree(&PrLeft_GmySpMdouble);
  mySpMatrixDoubleFree(&PrRight_GmySpMdouble);
  mySpMatrixDoubleFree(&PrLscale_GmySpMdouble);
  mySpMatrixDoubleFree(&PrRscale_GmySpMdouble);

  mySpMatrixFree(&AmySpM);
  mySpMatrixFree(&PrMiddle_AmySpM);
  mySpMatrixFree(&PrPermRow_AmySpM);
  mySpMatrixFree(&PrPermCol_AmySpM);

  mySpMatrixDoubleFree(&PrLeft_AmySpMdouble);
  mySpMatrixDoubleFree(&PrRight_AmySpMdouble);
  mySpMatrixDoubleFree(&PrLscale_AmySpMdouble);
  mySpMatrixDoubleFree(&PrRscale_AmySpMdouble);
}

///////////////////////////////////////////////////////////////////////////

void mna_solve_gpu(cs_dl *G, cs_dl *C, cs_dl *B, 
                   Source *VS, int nVS, Source *IS, int nIS, 
                   double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                   vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info,
                   char *ir_name,
                   gpuETBR *myGPUetbr)
{
  printf("             mna_solve_gpu()\n");
  Real_Timer interp2_run_time;
  Real_Timer ir_run_time;
  Real_Timer lufact_time;
  Real_Timer gmresCPUilu_time;
   
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, avg_ir; // min_ir, 
  int max_ir_idx; // , min_ir_idx
  ivec sorted_max_value_idx, sorted_min_value_idx, 
	sorted_avg_value_idx, sorted_ir_value_idx;
  int nNodes = tc_node.size();
  int display_num = num<tc_node.size()?num:tc_node.size();
  max_value.set_size(nNodes);
  min_value.set_size(nNodes);
  avg_value.set_size(nNodes);
  sorted_max_value_idx.set_size(nNodes);
  sorted_min_value_idx.set_size(nNodes);
  sorted_avg_value_idx.set_size(nNodes);
  sorted_ir_value_idx.set_size(nNodes);
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vector<int> const_v, const_i, var_v, var_i;
  for(int j = 0; j < nVS; j++){
	if (VS[j].time.size() == 1)
	  const_v.push_back(j);
	else
	  var_v.push_back(j);
  }
  for(int j = 0; j < nIS; j++){
	if (IS[j].time.size() == 1)
	  const_i.push_back(j);
	else
	  var_i.push_back(j);
  }
  /* DC simulation */
  for(vector<int>::iterator it = const_v.begin(); it != const_v.end(); ++it){
	u_col(*it) = VS[*it].value(0);
  }
  for(vector<int>::iterator it = const_i.begin(); it != const_i.end(); ++it){
	u_col(nVS+(*it)) = IS[*it].value(0);
  }
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
  vec x(n);
  x.zeros();
  cs_dls *SymbolicG, *SymbolicA;
  cs_dln *NumericG, *NumericA;
  int order = 2;
  double tol = 1e-10; // XXLiu: was 1e-14
  lufact_time.start();
  SymbolicG = cs_dl_sqr(order, G, 0);
  NumericG = cs_dl_lu(G, SymbolicG, tol);
  lufact_time.stop();
  cs_dl_ipvec(NumericG->pinv, w._data(), x._data(), n);
  cs_dl_lsolve(NumericG->L, x._data());
  cs_dl_usolve(NumericG->U, x._data());
  cs_dl_ipvec(SymbolicG->q, x._data(), xres._data(), n);

  int nport = port.size();
  int *invPort=(int*)malloc(nport*sizeof(int));
  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xres(port(j)));
        // printf("port index: %d\n",port(j));
        invPort[j] = port(j);
  }

  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xres(tc_node[j]);
	  min_value(j) = xres(tc_node[j]);
	  avg_value(j) = xres(tc_node[j]);
	}
	ir_run_time.stop();
  }

  /* Transient simulation */
  cs_dl *right = cs_dl_spalloc(C->m, C->n, C->nzmax, 1, 0);
  for (UF_long i = 0; i < C->n+1; i++){
	right->p[i] = C->p[i];
  }
  for (UF_long i = 0; i < C->nzmax; i++){
	right->i[i] = C->i[i];
	right->x[i] = 1/tstep*C->x[i];
  }
  cs_dl *left = cs_dl_add(G, right, 1, 1);
  lufact_time.start();
  SymbolicA = cs_dl_sqr(order, left, 0);
  NumericA = cs_dl_lu(left, SymbolicA, tol);
  lufact_time.stop();

  /* GMRES solver part starts. */
  ucr_cs_dl leftUCR, rightUCR, G_UCR, B_UCR, LG, UG, LA, UA;
  int *pinvG, *qinvG, *pinvA, *qinvA;
  pinvG=(int*)malloc((G->m)*sizeof(int));
  qinvG=(int*)malloc((G->m)*sizeof(int));
  pinvA=(int*)malloc((G->m)*sizeof(int));
  qinvA=(int*)malloc((G->m)*sizeof(int));
  myMemcpyL2I(pinvG, NumericG->pinv, G->m);
  myMemcpyL2I(qinvG, SymbolicG->q, G->m);
  myMemcpyL2I(pinvA, NumericA->pinv, G->m);
  myMemcpyL2I(qinvA, SymbolicA->q, G->m);


  leftUCR.shallowCpy(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  rightUCR.shallowCpy(right->nzmax, right->m, right->n, right->p, right->i, right->x, right->nz);
  G_UCR.shallowCpy(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  B_UCR.shallowCpy(B->nzmax, B->m, B->n, B->p, B->i, B->x, B->nz);
  LG.shallowCpy(NumericG->L->nzmax, NumericG->L->m, NumericG->L->n,
                NumericG->L->p, NumericG->L->i, NumericG->L->x, NumericG->L->nz);
  UG.shallowCpy(NumericG->U->nzmax, NumericG->U->m, NumericG->U->n,
                NumericG->U->p, NumericG->U->i, NumericG->U->x, NumericG->U->nz);
  LA.shallowCpy(NumericA->L->nzmax, NumericA->L->m, NumericA->L->n,
                NumericA->L->p, NumericA->L->i, NumericA->L->x, NumericA->L->nz);
  UA.shallowCpy(NumericA->U->nzmax, NumericA->U->m, NumericA->U->n,
                NumericA->U->p, NumericA->U->i, NumericA->U->x, NumericA->U->nz);

  /********************* GPU preparation ******************************/
  Real_Timer GPUtime;
  printf("*** GPU accelerated transient simulation ***\n");
  myGPUetbr->n = B->m;
  myGPUetbr->m = B->n;
  myGPUetbr->numPts = ts.size();
  myGPUetbr->ldUt = (((myGPUetbr->numPts-1) +31)/32)*32;
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;
  myGPUetbr->tstep = tstep;
  myGPUetbr->tstop = tstop;

  if(myGPUetbr->PWLcurExist && myGPUetbr->PULSEcurExist) {
    printf("                    Do not support PWL and PULSE current sources at the same time.\n");
    while( !getchar() ) ;
  }
  if(myGPUetbr->PWLcurExist) {
    if(myGPUetbr->PWLcurExist == nIS)
      printf("       All PWL current sources.\n");
    else {
      printf("       Error: There are non-PWL current sources mingled with PWL.\n");
      while(!getchar()) ;
    }
  }
  if(myGPUetbr->PULSEcurExist) {
    if(myGPUetbr->PULSEcurExist == nIS)
      printf("       All PULSE current sources.\n");
    else {
      printf("       Error: There are non-PULSE current sources mingled with PULSE.\n");
      while(!getchar()) ;
    }
  }
  myGPUetbr->nIS = nIS;
  myGPUetbr->nVS = nVS;

  if(myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m) {
    printf("                    myGPUetbr->nIS + myGPUetbr->nVS != myGPUetbr->m\n");
    while( !getchar() ) ;
  }

  if(myGPUetbr->PWLvolExist || myGPUetbr->PULSEvolExist) {
    printf("                    Do not support PWL and PULSE voltage sources.\n");
    while( !getchar() ) ;
  }

  
  /* The following section need CPU generated source info. */

  if(BLK_SIZE_UTGEN < MAX_PWL_PTS) {
    printf("       Error: BLK_SIZE_UTGEN should be no less than MAX_PWL_PTS\n");
    while(!getchar()) ;
  }

  /* The following section prepares the source info in order to use GPU evaluation. */
  // store DC voltage source info.
  myGPUetbr->dcVt_host=(double*)malloc(nVS*sizeof(double));
  for(int i=0; i<nVS; i++)
    myGPUetbr->dcVt_host[i] = VS[i].value(0);

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    myGPUetbr->dcVt_single_host=(float*)malloc(nVS*sizeof(float));
    myMemcpyD2S(myGPUetbr->dcVt_single_host,  myGPUetbr->dcVt_host, nVS);
  }

  if(myGPUetbr->PWLcurExist) {
    myGPUetbr->PWLnumPts_host=(int*)malloc(nIS*sizeof(int));
    myGPUetbr->PWLtime_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    myGPUetbr->PWLval_host=(double*)malloc(nIS*MAX_PWL_PTS*sizeof(double));
    for(int i=0; i<nIS; i++) {
      int herePWLnumPts=IS[i].time.size();
      //printf(" size: %d\n",herePWLnumPts);
      if(herePWLnumPts > MAX_PWL_PTS) {
	printf("       Error: More PWL points than allowed. %d > %d at source-%d\n",herePWLnumPts, MAX_PWL_PTS, i);
	while(!getchar()) ;
      }
      myGPUetbr->PWLnumPts_host[i] = herePWLnumPts;
      for(int j=0; j<herePWLnumPts; j++) {
	myGPUetbr->PWLtime_host[i*MAX_PWL_PTS+j] = IS[i].time(j);
	myGPUetbr->PWLval_host[i*MAX_PWL_PTS+j] = IS[i].value(j);
      }
    }

    if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
      myGPUetbr->PWLtime_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myGPUetbr->PWLval_single_host=(float*)malloc(nIS*MAX_PWL_PTS*sizeof(float));
      myMemcpyD2S(myGPUetbr->PWLtime_single_host, myGPUetbr->PWLtime_host, nIS*MAX_PWL_PTS);
      myMemcpyD2S(myGPUetbr->PWLval_single_host, myGPUetbr->PWLval_host, nIS*MAX_PWL_PTS);
    }
  }

  if(myGPUetbr->use_cuda_single) { // SINGLE PRECISION
    if(nport > 0)
      myGPUetbr->x_single_host=(float*)malloc(nport*(myGPUetbr->numPts)*sizeof(float));
  }
  
  if(nport > 0)
    wrapperGPUforPG(&leftUCR, &rightUCR, &G_UCR, &B_UCR, //w._data(),
                    invPort, nport,
                    &LG, &UG, pinvG, qinvG,
                    &LA, &UA, pinvA, qinvA, myGPUetbr);

  for(int i=0; i<nport; i++)
    for(int j=0; j<ts.size(); j++)
      sim_port_value.set(i, j, (double)myGPUetbr->x_single_host[i+j*nport]);
  

  cs_dl_spfree(left);
  free(invPort);
  //XXLiu 2012-11-26 // vec xn(n), xnr(n), xn1(n), xn1t(n);
  //XXLiu 2012-11-26 // xn = xres;
  //XXLiu 2012-11-26 // xn1.zeros();
  //XXLiu 2012-11-26 // xn1t.zeros();
  //XXLiu 2012-11-26 // printf("   ts.size() = %d.\n",ts.size());
  //XXLiu 2012-11-26 // for (int i = 1; i < ts.size(); i++){
  //XXLiu 2012-11-26 //       /*
  //XXLiu 2012-11-26 //       for(int j = 0; j < nVS; j++){
  //XXLiu 2012-11-26 //         interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
  //XXLiu 2012-11-26 //         u_col(j) = temp;
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       for(int j = 0; j < nIS; j++){
  //XXLiu 2012-11-26 //         interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
  //XXLiu 2012-11-26 //         u_col(nVS+j) = temp;
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       */
  //XXLiu 2012-11-26 //       interp2_run_time.start();
  //XXLiu 2012-11-26 //       for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
  //XXLiu 2012-11-26 //         interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
  //XXLiu 2012-11-26 //         u_col(*it) = temp;
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
  //XXLiu 2012-11-26 //         interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
  //XXLiu 2012-11-26 //         u_col(nVS+(*it)) = temp;
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       interp2_run_time.stop();
  //XXLiu 2012-11-26 //       w.zeros();
  //XXLiu 2012-11-26 //       cs_dl_gaxpy(B, u_col._data(), w._data());
  //XXLiu 2012-11-26 //       xnr.zeros();
  //XXLiu 2012-11-26 //       // cs_dl_gaxpy(C, xn._data(), xnr._data());
  //XXLiu 2012-11-26 //       // w += 1/tstep*xnr;
  //XXLiu 2012-11-26 //       cs_dl_gaxpy(right, xn._data(), xnr._data());
  //XXLiu 2012-11-26 //       w += xnr;
  //XXLiu 2012-11-26 //       cs_dl_ipvec(NumericA->pinv, w._data(), xn1t._data(), n);
  //XXLiu 2012-11-26 //       cs_dl_lsolve(NumericA->L, xn1t._data());
  //XXLiu 2012-11-26 //       cs_dl_usolve(NumericA->U, xn1t._data());
  //XXLiu 2012-11-26 //       cs_dl_ipvec(SymbolicA->q, xn1t._data(), xn1._data(), n);   
  //XXLiu 2012-11-26 //       for (int j = 0; j < port.size(); j++){
  //XXLiu 2012-11-26 //         sim_port_value.set(j, i, xn1(port(j)));
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       if (ir_info){
  //XXLiu 2012-11-26 //         ir_run_time.start();
  //XXLiu 2012-11-26 //         for (int j = 0; j < nNodes; j++){
  //XXLiu 2012-11-26 //       	if (max_value(j) < xn1(tc_node[j])){
  //XXLiu 2012-11-26 //       	  max_value(j) = xn1(tc_node[j]);
  //XXLiu 2012-11-26 //       	}
  //XXLiu 2012-11-26 //       	if (xn1(tc_node[j]) < min_value(j)){
  //XXLiu 2012-11-26 //       	  min_value(j) = xn1(tc_node[j]);
  //XXLiu 2012-11-26 //       	}
  //XXLiu 2012-11-26 //       	avg_value(j) += xn1(tc_node[j]);
  //XXLiu 2012-11-26 //         }
  //XXLiu 2012-11-26 //         ir_run_time.stop();
  //XXLiu 2012-11-26 //       }
  //XXLiu 2012-11-26 //       xn = xn1;
  //XXLiu 2012-11-26 // }
  cs_dl_spfree(right);
  cs_dl_sfree(SymbolicA);
  cs_dl_nfree(NumericA);
  cs_dl_sfree(SymbolicG);
  cs_dl_nfree(NumericG);
  delete [] cur;

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	/*
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	*/
	ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	ir_run_time.stop();
  }

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation2  \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "IR analysis     \t: " << ir_run_time.get_time() << std::endl;
  std::cout << "LU factorization\t: " << lufact_time.get_time() << std::endl;
}


////////////////////////////////////////////////////////////////////////////
void mna_solve_cpu_ilu_gmres(cs_dl *G, cs_dl *C, cs_dl *B, 
                         Source *VS, int nVS, Source *IS, int nIS, 
                         double tstep, double tstop, const ivec &port, mat &sim_port_value, 
                         vector<int> &tc_node, vector<string> &tc_name, int num, int ir_info,
                         char *ir_name)//, gpuETBR *myGPUetbr
{
  printf("             mna_solve_cpu_ilu_gmres()\n");
  Real_Timer interp2_run_time;
  Real_Timer ir_run_time;
  Real_Timer lufact_time;
  Real_Timer gmresCPUilu_time;
   
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, avg_ir; // min_ir, 
  int max_ir_idx; // , min_ir_idx
  ivec sorted_max_value_idx, sorted_min_value_idx, 
	sorted_avg_value_idx, sorted_ir_value_idx;
  int nNodes = tc_node.size();
  int display_num = num<tc_node.size()?num:tc_node.size();
  max_value.set_size(nNodes);
  min_value.set_size(nNodes);
  avg_value.set_size(nNodes);
  sorted_max_value_idx.set_size(nNodes);
  sorted_min_value_idx.set_size(nNodes);
  sorted_avg_value_idx.set_size(nNodes);
  sorted_ir_value_idx.set_size(nNodes);
  UF_long n = G->n;
  vec u_col(nVS+nIS);
  u_col.zeros();
  vec w(n);
  w.zeros();
  vec ts;
  form_vec(ts, 0, tstep, tstop);
  sim_port_value.set_size(port.size(), ts.size());
  double temp;
  int* cur = new int[nVS+nIS];
  for(int i = 0; i < nVS+nIS; i++){
	cur[i] = 0;
  }
  vector<int> const_v, const_i, var_v, var_i;
  for(int j = 0; j < nVS; j++){
	if (VS[j].time.size() == 1)
	  const_v.push_back(j);
	else
	  var_v.push_back(j);
  }
  for(int j = 0; j < nIS; j++){
	if (IS[j].time.size() == 1)
	  const_i.push_back(j);
	else
	  var_i.push_back(j);
  }
  /* DC simulation */
  for(vector<int>::iterator it = const_v.begin(); it != const_v.end(); ++it){
	u_col(*it) = VS[*it].value(0);
  }
  for(vector<int>::iterator it = const_i.begin(); it != const_i.end(); ++it){
	u_col(nVS+(*it)) = IS[*it].value(0);
  }
  for (int i = 0; i < 1; i++){
	for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
	  interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
	  u_col(*it) = temp;
	}
	for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	  interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
	  u_col(nVS+(*it)) = temp;
	}
	cs_dl_gaxpy(B, u_col._data(), w._data());
  }
  vec xres(n);
  xres.zeros();
  vec x(n);
  x.zeros();
  // cs_dls *SymbolicG, *SymbolicA;
  // cs_dln *NumericG, *NumericA;
  // int order = 2;
  // double tol = 1e-10; // XXLiu: was 1e-14
  // lufact_time.start();
  // SymbolicG = cs_dl_sqr(order, G, 0);
  // NumericG = cs_dl_lu(G, SymbolicG, tol);
  // lufact_time.stop();
  // cs_dl_ipvec(NumericG->pinv, w._data(), x._data(), n);
  // cs_dl_lsolve(NumericG->L, x._data());
  // cs_dl_usolve(NumericG->U, x._data());
  // cs_dl_ipvec(SymbolicG->q, x._data(), xres._data(), n);

  // int nport = port.size();
  // int *invPort=(int*)malloc(nport*sizeof(int));
  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xres(port(j)));
        // printf("port index: %d\n",port(j));
        //invPort[j] = port(j);
  }

  if (ir_info){
	ir_run_time.start();
	for (int j = 0; j < nNodes; j++){
	  max_value(j) = xres(tc_node[j]);
	  min_value(j) = xres(tc_node[j]);
	  avg_value(j) = xres(tc_node[j]);
	}
	ir_run_time.stop();
  }

  /* Transient simulation */
  cs_dl *right = cs_dl_spalloc(C->m, C->n, C->nzmax, 1, 0);
  for (UF_long i = 0; i < C->n+1; i++){
	right->p[i] = C->p[i];
  }
  for (UF_long i = 0; i < C->nzmax; i++){
	right->i[i] = C->i[i];
	right->x[i] = 1/tstep*C->x[i];
  }
  cs_dl *left = cs_dl_add(G, right, 1, 1);
  // lufact_time.start();
  // SymbolicA = cs_dl_sqr(order, left, 0);
  // NumericA = cs_dl_lu(left, SymbolicA, tol);
  // lufact_time.stop();


  cout<<"*****************************************************************************"<<endl;
  cout<<" ILU++ "<<endl;
  cout<<"*****************************************************************************"<<endl;
  typedef iluplusplus::Real Real;
  typedef iluplusplus::matrix_sparse<Real> Matrix;
  typedef iluplusplus::vector_dense<Real> Vector;
  
  iluplusplus::preprocessing_sequence L;
  L.set_MAX_WEIGHTED_MATCHING_ORDERING_DD_MOV_COR_IM(); // from original example.
  //2013Feb05 L.set_MAX_WEIGHTED_MATCHING_ORDERING_PQ();
  cout<<"Preprocessing selected:"<<endl;
  L.print();
  cout<<endl;
  iluplusplus::iluplusplus_precond_parameter param;
  param.init(L,10,"test"); // setup some other default values. Has no effect on preprocessing (except choice of PQ-Algorithm)
  param.set_threshold(threshold);
  param.set_MEM_FACTOR(factor);
  param.set_MAX_LEVELS(1);
  param.set_MAX_FILLIN_IS_INF(true);
  
  //Matrix Arow;
  ucr_cs_di Gcs_di, Acs_di;
  Gcs_di.convertFromCS_DL(G->nzmax, G->m, G->n, G->p, G->i, G->x, G->nz);
  Acs_di.convertFromCS_DL(left->nzmax, left->m, left->n, left->p, left->i, left->x, left->nz);
  Matrix Gcol(Gcs_di.x, Gcs_di.i, Gcs_di.p, Gcs_di.m, Gcs_di.n, iluplusplus::COLUMN);
  Matrix Acol(Acs_di.x, Acs_di.i, Acs_di.p, Acs_di.m, Acs_di.n, iluplusplus::COLUMN);
  iluplusplus::multilevelILUCDPPreconditioner<Real,Matrix,Vector> PrG, PrA;
  PrG.make_preprocessed_multilevelILUCDP(Gcol,param);
  PrA.make_preprocessed_multilevelILUCDP(Acol,param);
  cout<<"Information on the preconditioner:"<<endl;
  PrG.print_info();
  cout<<"fill-in: "<<((Real) PrG.total_nnz())/(Real)Gcol.non_zeroes()<<endl;
  
  Vector b(n), xgmres(n); // xbicgstab,xcgs,

  b.value( w._data(), 0, n );
  //Acol.matrix_vector_multiplication(iluplusplus::ID,x_exact,b);
  
  int min_iter =0;
  int reach_max_iter = 10000;
  Real reach_rel_tol = 4.0;
  Real reach_abs_tol = 4.0;
  
  Real rel_tol =  reach_rel_tol;
  Real abs_tol = reach_abs_tol;
  int max_iter = reach_max_iter;
  int restart = 100;
  
  cout<<"*****************************************************************************"<<endl;
  cout<<"GMRES"<<endl;
  cout<<"*****************************************************************************"<<endl;
  gmresCPUilu_time.start();
  iluplusplus::gmres<Real,Matrix,Vector>(PrG,iluplusplus::SPLIT,Gcol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
  gmresCPUilu_time.stop();
  cout<<"Iterations "<<max_iter<<endl;
  //cout<<"error: "<<(xgmres-x_exact).norm_max()<<endl<<flush;
  cout<<"relative decrease in norm of residual: "<<exp(-rel_tol*log(10.0))<<endl;
  cout<<"absolute residual: "<<exp(-abs_tol*log(10.0))<<endl;// cout<<endl;


  // // Test the ILU
  // Matrix PrecondG_left, PrecondG_right;
  // Vector PrecondG_middle;
  // PrecondG_left = PrG.extract_left_matrix(0);
  // PrecondG_right = PrG.extract_right_matrix(0);
  // PrecondG_middle = PrG.extract_middle_matrix(0);
  // 
  // int *ptr=(int*)malloc((n+1)*sizeof(int));
  // int nnz = PrecondG_left.get_pointer(n);
  // int *idx=(int*)malloc(nnz*sizeof(int));
  // float *val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_left.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_left.get_index(i);
  //   val[i] = PrecondG_left.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_left_mySpM.dat");
  // free(idx); free(val);
  // 
  // nnz = PrecondG_right.get_pointer(n);
  // idx=(int*)malloc(nnz*sizeof(int));
  // val=(float*)malloc(nnz*sizeof(float));
  // for(int i=0; i<=n; i++)  ptr[i] = PrecondG_right.get_pointer(i);
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = PrecondG_right.get_index(i);
  //   val[i] = PrecondG_right.get_data(i);
  // }
  // writeCSR(n, n, nnz, ptr, idx, val, "PrG_right_mySpM.dat");
  // 
  // for(int i=0; i<=n; i++)  ptr[i] = i;
  // for(int i=0; i<nnz; i++) {
  //   idx[i] = i;
  //   val[i] = PrecondG_middle.get(i);
  // }
  // writeCSR(n, n, n, ptr, idx, val, "PrG_middle_mySpM.dat");
  // free(idx); free(val);
  // // Test finished

  //exit(0);


  for (int j = 0; j < port.size(); j++){
	sim_port_value.set(j, 0, xgmres[port(j)]);
        // printf("port index: %d\n",port(j));
        //invPort[j] = port(j);
  }

  int iterTotal=0;
  /* GMRES solver part finishes. */
  
  vec xn(n), xnr(n), xn1(n), xn1t(n);
  for(int j=0; j<n; j++)
    xn._data()[j] = xgmres[j]; // xn = xres;
  xn1.zeros();
  xn1t.zeros();
  printf("   ts.size() = %d.\n",ts.size());
  for (int i = 1; i < ts.size(); i++){
        /*
        for(int j = 0; j < nVS; j++){
          interp1(VS[j].time, VS[j].value, ts(i), temp, cur[j]);
          u_col(j) = temp;
        }
        for(int j = 0; j < nIS; j++){
          interp1(IS[j].time, IS[j].value, ts(i), temp, cur[nVS+j]);
          u_col(nVS+j) = temp;
        }
        */
        interp2_run_time.start();
        for(vector<int>::iterator it = var_v.begin(); it != var_v.end(); ++it){
          interp1(VS[*it].time, VS[*it].value, ts(i), temp, cur[*it]);
          u_col(*it) = temp;
        }
        for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
          interp1(IS[*it].time, IS[*it].value, ts(i), temp, cur[nVS+(*it)]);
          u_col(nVS+(*it)) = temp;
        }
        interp2_run_time.stop();

        w.zeros();
        cs_dl_gaxpy(B, u_col._data(), w._data());
        xnr.zeros();
        // cs_dl_gaxpy(C, xn._data(), xnr._data());
        // w += 1/tstep*xnr;
        cs_dl_gaxpy(right, xn._data(), xnr._data());
        w += xnr;
        
        rel_tol =  reach_rel_tol;
        abs_tol = reach_abs_tol;
        max_iter = reach_max_iter;
        restart = 100;
        b.value( w._data(), 0, n );
        gmresCPUilu_time.start();
        iluplusplus::gmres<Real,Matrix,Vector>(PrA,iluplusplus::SPLIT,Acol,b,xgmres,restart,min_iter,max_iter,rel_tol,abs_tol,true);
        gmresCPUilu_time.stop();
        //cout<<"Iterations "<<max_iter<<endl;
        
        iterTotal += max_iter;

        for(int j=0; j<n; j++)
          xn1._data()[j] = xgmres[j];
        // cs_dl_ipvec(NumericA->pinv, w._data(), xn1t._data(), n);
        // cs_dl_lsolve(NumericA->L, xn1t._data());
        // cs_dl_usolve(NumericA->U, xn1t._data());
        // cs_dl_ipvec(SymbolicA->q, xn1t._data(), xn1._data(), n);   

        for (int j = 0; j < port.size(); j++){
          sim_port_value.set(j, i, xn1(port(j)));
        }
        if (ir_info){
          ir_run_time.start();
          for (int j = 0; j < nNodes; j++){
        	if (max_value(j) < xn1(tc_node[j])){
        	  max_value(j) = xn1(tc_node[j]);
        	}
        	if (xn1(tc_node[j]) < min_value(j)){
        	  min_value(j) = xn1(tc_node[j]);
        	}
        	avg_value(j) += xn1(tc_node[j]);
          }
          ir_run_time.stop();
        }
        xn = xn1;
  }
  cs_dl_spfree(left);
  cs_dl_spfree(right);
  // cs_dl_sfree(SymbolicA);
  // cs_dl_nfree(NumericA);
  // cs_dl_sfree(SymbolicG);
  // cs_dl_nfree(NumericG);
  delete [] cur;

  if (ir_info){
	ir_run_time.start();
	avg_value /= ts.size();
	sorted_max_value_idx = sort_index(max_value);
	sorted_avg_value_idx = sort_index(avg_value);
	/*
	vec sgn_value = sgn(max_value) - sgn(min_value);
	ir_value.set_size(max_value.size());
	for (int i = 0; i < sgn_value.size(); i++){
	  if(sgn_value(i) == 0){
		ir_value(i) = max_value(i) - min_value(i);
	  }
	  else{
		ir_value(i) = abs(max_value(i)) > abs(min_value(i))? abs(max_value(i)):abs(min_value(i));
	  }
	}
	*/
	ir_value = max_value - min_value;
	max_ir = max(ir_value);
	max_ir_idx = max_index(ir_value);
	avg_ir = sum(ir_value)/ir_value.size(); 
	sorted_ir_value_idx = sort_index(ir_value);
	std::cout.precision(6);
	cout << "****** Node Voltage Info ******  " << endl;
	cout << "#Tap Currents: " << tc_node.size() << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_max_value_idx(nNodes-1-i)] << " : " 
		   << max_value(sorted_max_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;
	cout << "Avg " << display_num << " Node Voltage: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		   << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
	}
	cout << "****** IR Drop Info ******  " << endl;
	cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
	cout << "Avg IR:     " << avg_ir << endl;
	cout << "******" << endl;
	cout << "Max " << display_num << " IR: " << endl;
	for (int i = 0; i < display_num; i++){
	  cout << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	cout << "******" << endl;

	ofstream out_ir;
	out_ir.open(ir_name);
	if (!out_ir){
	  cout << "couldn't open " << ir_name << endl;
	  exit(-1);
	}
	for (int i = 0; i < tc_node.size(); i++){
	  out_ir << tc_name[sorted_ir_value_idx(nNodes-1-i)] << " : " 
		   << ir_value(sorted_ir_value_idx(nNodes-1-i)) << endl;
	}
	out_ir.close();
	ir_run_time.stop();
  }

  std::cout.setf(std::ios::fixed,std::ios::floatfield); 
  std::cout.precision(2);
  std::cout << "interpolation2  \t: " << interp2_run_time.get_time() << std::endl;
  std::cout << "IR analysis     \t: " << ir_run_time.get_time() << std::endl;
  std::cout << "LU factorization\t: " << lufact_time.get_time() << std::endl;
  std::cout << "ILU++ GMRES CPU \t: " << gmresCPUilu_time.get_time() << "    Avg Iter: " << (int)iterTotal/ts.size() << std::endl;
}
