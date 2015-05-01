/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: mna.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:59 $
 *    Authors: Ning Mi 
 *
 *    Functions: stamp function 
 *
 */


#include "mna.h"

MNA::MNA(circuit* ckt)
{
        int i;

	printf("Start MNA Simulation...\n");

	netlist = ckt;

	size_G = ckt->nodePool->numNode()+ckt->L->numBranch()+ckt->V->numBranch();
	size_C = size_G;
	row_B = size_G;
	col_B = ckt->V->numBranch()+ckt->I->numBranch();
	row_u = ckt->V->numBranch()+ckt->I->numBranch();
	
	if((G = new matrix(size_G, size_G)) == NULL)
	{
		printf("Out of memory!\n"); exit(1);
	}

	if((C = new matrix(size_C, size_C)) == NULL)
	{
		printf("Out of memory!\n"); exit(1);
	}

	if((B = new matrix(row_B, col_B)) == NULL)
	  {
	    printf("Out of memory!\n"); exit(1);
	  }

	if((uIndex = (int*)malloc(row_u*sizeof(int))) == NULL)
	  {
	    printf("Out of memory!\n"); exit(1);
	  }
	// stamp
	// printf("Begin stamp...\n");
	//	stamp();

	//print
	/*	printf("Print matrix...\n");
	printf("G...\n");
	G->printmatrix();
	printf("C...\n");
	C->printmatrix();
	printf("B...\n");
	B->printmatrix();
	printf("u index ...\n");
	for (i=0; i<row_u; i++){
	  printf("uIndex[%d] = %d \n", i, uIndex[i]);
	  }
	  printf("Finish stamp.\n");*/
}

MNA::~MNA()
{
  // delete G; delete C; delete B; 
  free(uIndex);
}

void MNA::stamp()
{
	int i;
	int n1,n2;
	int index_i, index_j;
	int node_number;

	double value;

	NodeList* node = netlist->nodePool; 
	LinBranch::Branch* tmp_lb;
	LinBranch::Branch* tmp_ib;
	LinBranch* tmp_LBranch;

	SrcBranch::Branch* tmp_sb;
	SrcBranch* tmp_SBranch;	
	node_number = node->numNode();

	// push G element
	for ( i=0;i<netlist->R->numBranch();i++ )
	{
		tmp_lb = netlist->R->getBranch(i);
		n1 = node->getNode(tmp_lb->node1)->row_no;
		n2 = node->getNode(tmp_lb->node2)->row_no;
		value = tmp_lb->value;
		if (n1 != GNDNODE) G->pushEntry(n1, n1, value);
		if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
		if (n1 != GNDNODE && n2 != GNDNODE)
		{
			G->pushEntry(n1,n2,-value);
			G->pushEntry(n2,n1,-value);
		}
	}

	// push C element
	for (i=0;i<netlist->C->numBranch();i++)
	{
		tmp_lb = netlist->C->getBranch(i);
		n1 = node->getNode(tmp_lb->node1)->row_no;
		n2 = node->getNode(tmp_lb->node2)->row_no;
		value = tmp_lb->value;
		if(n1 != GNDNODE) C->pushEntry(n1, n1, value);
 		if(n2 != GNDNODE) C->pushEntry(n2, n2, value);
 		if (n1 != GNDNODE && n2 != GNDNODE)
		{
			C->pushEntry(n1,n2,-value);
			C->pushEntry(n2,n1,-value);
		}
	}

	// push L element
	for (i=0;i<netlist->L->numBranch();i++)
	  {
	    tmp_ib = netlist->L->getBranch(i);
	    n1 = node->getNode(tmp_ib->node1)->row_no;
	    n2 = node->getNode(tmp_ib->node2)->row_no;
	    index_i = node_number+i;
	    if( n1!=GNDNODE )
	      { G->pushEntry(index_i,n1,-1); G->pushEntry(n1,index_i,1); }
	    if( n2!=GNDNODE )
	      { G->pushEntry(index_i,n2,1); G->pushEntry(n2,index_i,-1); }
	    C->pushEntry(index_i, index_i, tmp_ib->value);
	  }            

	// push V element
	for (i=0;i<netlist->V->numBranch();i++)
	  {
	    tmp_sb = netlist->V->getBranch(i);
	    n1 = node->getNode(tmp_sb->node1)->row_no;
	    n2 = node->getNode(tmp_sb->node2)->row_no;
	    index_i = node_number+netlist->L->numBranch()+i;
	    index_j = i;
	    if(n1 != GNDNODE) {
	      G->pushEntry(n1, index_i, 1);
	      G->pushEntry(index_i, n1, -1);
	    }
	    if(n2 != GNDNODE) {
	      G->pushEntry(n2, index_i, -1);
	      G->pushEntry(index_i, n2, 1);
	    }
	    B->pushEntry(index_i, index_j, -1);
	    uIndex[index_j] = tmp_sb->wave;
	  }

	// push I element
	for (i=0;i<netlist->I->numBranch();i++)
	  {
	    tmp_sb = netlist->I->getBranch(i);
	    n1 = node->getNode(tmp_sb->node1)->row_no;
	    n2 = node->getNode(tmp_sb->node2)->row_no;
	    index_j = netlist->V->numBranch()+i;
	    if(n1 != GNDNODE) {
	      B->pushEntry(n1,index_j,1);
	    }
	    if(n2 != GNDNODE) {
	      B->pushEntry(n2,index_j,1);
	    }
	    uIndex[index_j] = tmp_sb->wave;
	  }
	G->sort(); //G->compact();
	C->sort(); //C->compact();
	B->sort();
}

void MNA::stampG()
{
	int i;
	int n1,n2;
	int index_i, index_j;
	int node_number;

	double value;

	NodeList* node = netlist->nodePool; 
	LinBranch::Branch* tmp_lb;
	LinBranch::Branch* tmp_ib;
	LinBranch* tmp_LBranch;

	SrcBranch::Branch* tmp_sb;
	SrcBranch* tmp_SBranch;	
	node_number = node->numNode();

	// push V element
	for (i=0;i<netlist->V->numBranch();i++)
	  {
	    tmp_sb = netlist->V->getBranch(i);
	    n1 = node->getNode(tmp_sb->node1)->row_no;
	    n2 = node->getNode(tmp_sb->node2)->row_no;
	    index_i = node_number+netlist->L->numBranch()+i;
	    index_j = i;
	    if(n1 != GNDNODE) {
	      G->pushEntry(n1, index_i, 1);
	      G->pushEntry(index_i, n1, -1);
	    }
	    if(n2 != GNDNODE) {
	      G->pushEntry(n2, index_i, -1);
	      G->pushEntry(index_i, n2, 1);
	    }
		free(tmp_sb);
	  }
	netlist->deleteV();
	// push L element
	for (i=0;i<netlist->L->numBranch();i++)
	  {
	    tmp_ib = netlist->L->getBranch(i);
	    n1 = node->getNode(tmp_ib->node1)->row_no;
	    n2 = node->getNode(tmp_ib->node2)->row_no;
	    index_i = node_number+i;
	    if( n1!=GNDNODE )
	      { G->pushEntry(index_i,n1,-1); G->pushEntry(n1,index_i,1); }
	    if( n2!=GNDNODE )
	      { G->pushEntry(index_i,n2,1); G->pushEntry(n2,index_i,-1); }
		free(tmp_ib);
	  }            
	netlist->deleteL();
	// push G element
	for ( i=0;i<netlist->R->numBranch();i++ )
	{
		tmp_lb = netlist->R->getBranch(i);
		n1 = node->getNode(tmp_lb->node1)->row_no;
		n2 = node->getNode(tmp_lb->node2)->row_no;
		value = tmp_lb->value;
		if (n1 != GNDNODE) G->pushEntry(n1, n1, value);
		if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
		if (n1 != GNDNODE && n2 != GNDNODE)
		{
			G->pushEntry(n1,n2,-value);
			G->pushEntry(n2,n1,-value);
		}
		free(tmp_lb);
	}
	netlist->deleteR();
	G->sort(); //G->compact();
}

void MNA::stampC()
{
	int i;
	int n1,n2;
	int index_i, index_j;
	int node_number;

	double value;

	NodeList* node = netlist->nodePool; 
	LinBranch::Branch* tmp_lb;
	LinBranch::Branch* tmp_ib;
	LinBranch* tmp_LBranch;

	SrcBranch::Branch* tmp_sb;
	SrcBranch* tmp_SBranch;	
	node_number = node->numNode();

	// push C element
	for (i=0;i<netlist->C->numBranch();i++)
	{
		tmp_lb = netlist->C->getBranch(i);
		n1 = node->getNode(tmp_lb->node1)->row_no;
		n2 = node->getNode(tmp_lb->node2)->row_no;
		value = tmp_lb->value;
		if(n1 != GNDNODE) C->pushEntry(n1, n1, value);
 		if(n2 != GNDNODE) C->pushEntry(n2, n2, value);
 		if (n1 != GNDNODE && n2 != GNDNODE)
		{
			C->pushEntry(n1,n2,-value);
			C->pushEntry(n2,n1,-value);
		}
		free(tmp_lb);
	}
	netlist->deleteC();
	// push L element
	for (i=0;i<netlist->L->numBranch();i++)
	  {
	    tmp_ib = netlist->L->getBranch(i);
	    index_i = node_number+i;
	    C->pushEntry(index_i, index_i, tmp_ib->value);
	  }            

	C->sort(); //C->compact();
}

void MNA::stampB()
{
	int i;
	int n1,n2;
	int index_i, index_j;
	int node_number;

	double value;

	NodeList* node = netlist->nodePool; 
	LinBranch::Branch* tmp_lb;
	LinBranch::Branch* tmp_ib;
	LinBranch* tmp_LBranch;

	SrcBranch::Branch* tmp_sb;
	SrcBranch* tmp_SBranch;	
	node_number = node->numNode();
	// push I element
	for (i=0;i<netlist->I->numBranch();i++)
	  {
	    tmp_sb = netlist->I->getBranch(i);
	    n1 = node->getNode(tmp_sb->node1)->row_no;
	    n2 = node->getNode(tmp_sb->node2)->row_no;
	    index_j = netlist->V->numBranch()+i;
	    if(n1 != GNDNODE) {
	      B->pushEntry(n1,index_j,1);
	    }
	    if(n2 != GNDNODE) {
	      B->pushEntry(n2,index_j,1);
	    }
	    uIndex[index_j] = tmp_sb->wave;
		free(tmp_sb);
	  }
	netlist->deleteI();
	// push V element
	for (i=0;i<netlist->V->numBranch();i++)
	  {
	    tmp_sb = netlist->V->getBranch(i);
	    index_i = node_number+netlist->L->numBranch()+i;
	    index_j = i;
	    B->pushEntry(index_i, index_j, -1);
	    uIndex[index_j] = tmp_sb->wave;
	  }
	B->sort();
}


/*void MNA::transform(sparse_mat* sG, sparse_mat* sC, sparse_mat* sB)
{
  G->trans(sG);
  C->trans(sC);
  B->trans(sB);
  }*/

cs* MNA::G2cs()
{
  cs* T=G->mat2cs();
  return T;
}

cs* MNA::C2cs()
{
  cs* T=C->mat2cs();
  return T;
}

cs* MNA::B2cs()
{
  cs* T=B->mat2cs();
  return T;
}

cs_dl* MNA::G2csdl()
{
  cs_dl* T=G->mat2csdl();
  return T;
}

cs_dl* MNA::C2csdl()
{
  cs_dl* T=C->mat2csdl();
  return T;
}

cs_dl* MNA::B2csdl()
{
  cs_dl* T=B->mat2csdl();
  return T;
}
