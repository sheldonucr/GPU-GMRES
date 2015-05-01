/********************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: circuit.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:54 $
 *    Authors: Ning Mi 
 *
 *    Functions: circuit and waveform class
 *
 */


#include "circuit.h"
#include <stdlib.h>

wave::wave(){

}

wave::~wave(){
}

int wave::newDC(double value)
{
  Element* temp_element = (Element*)malloc(sizeof(Element));
  temp_element->type = DC;
  temp_element->addr = valueData.size();
  eleData.push_back(*temp_element);
  free(temp_element);

  valueData.push_back(value);

  return eleData.size()-1;
}

int wave::newPWL(double time, double value)
{
  Element* temp_element = (Element*)malloc(sizeof(Element));
  int addr = valueData.size();

  temp_element->type = PWL;
  temp_element->addr = addr;
  eleData.push_back(*temp_element);
  free(temp_element);

  //valueData.push_back(addr+2);
  //valueData.push_back(addr+5);

  if(time != 0){
    valueData.push_back(0.0);
    valueData.push_back(value);
    //valueData[addr+1] = addr+4;
  }
  valueData.push_back(time);
  valueData.push_back(value);
  //valueData[addr+1] += 2;

  return eleData.size()-1;
}

bool wave::pushPWL(double time, double value)
{
  int addr = valueData.size()-2;
  if( valueData[addr] >= time) return false;
  valueData.push_back(time);
  valueData.push_back(value);
  return true;
}

/*bool wave::getsrc()
{
  int i;
  int size;
  for (i=0;i<eleData.size();i++){
    if(eleData[i]==PWL){
      if(i == eleData.size()-1)
	size = valueData.size()-eleData[i].addr;
      else
	size = eleData[i+1].addr - eleData[i].addr;
      if((src[i] = (double*)malloc(size*sizeof(double)))==NULL) 
	{ printf("Out of memory!"); return false; }
    }
    else if(eleData[i]==DC){
      size = 4;
      if((src[i] = (double*)malloc(size*sizeof(double)))==NULL) 
      { printf("Out of memory!"); return false; }
      
    }
  }
  }*/

void wave::printWaveElement(int i)
{
  int addr;
  int j,k;
  if (i>=eleData.size()) {printf("%dth element is not available",i); exit(1);}
  else if(i<eleData.size()-1){
    addr = eleData[i].addr;
    if(eleData[i].type == DC)
      printf("%dth  element: DC, value %f\n\n", i, valueData[addr]);
    else if(eleData[i].type == PWL){
      printf("%dth element: PWL\n",i);
      for(j=addr;j<eleData[i+1].addr;j+=2){
	k = j+1;
	printf("time: %e, value %f\n", valueData[j], valueData[k]);
      }
      printf("\n");
    }
  }
  else{
    addr = eleData[i].addr;
    if(eleData[i].type == DC)
      printf("%dth element: DC, value %f\n\n", valueData[addr]);
    else if(eleData[i].type == PWL){
      printf("%dth element: PWL\n",i);
      for(j=addr;j<valueData.size();j+=2){
	k = j+1;
	printf("time: %e, value %f\n", valueData[j], valueData[k]);
      }
      printf("\n");
    }
  }
}

void wave::printWave()
{
  int i;
  for(i=0;i<eleData.size();i++){
    printWaveElement(i);
  }
}

void wave::printvalueData()
{
  for (int i=0;i<valueData.size();i++)
    {
      printf("value: %e\n ",valueData[i]);
    }
  printf("\n");
}

circuit::circuit(){
  if((nodePool = new NodeList) == NULL){
    printf("Not enough memory! \n"); exit(1); 
  }
  if((R = new LinBranch) == NULL){
    printf("Not enough memory! \n"); exit(1);
  }
  if((C = new LinBranch) == NULL){
    printf("Not enough memory! \n"); exit(1);
  }
  if((L = new LinBranch) == NULL){
    printf("Not enough memory! \n"); exit(1);
  } 
  if((V = new SrcBranch) == NULL){
    printf("Not enough memory! \n"); exit(1);
  }
  if((I = new SrcBranch) == NULL){
    printf("Not enough memory! \n"); exit(1);
  }
}

circuit::~circuit(){
  delete nodePool; 
  //delete R; delete C; delete L;
  //delete V; delete I;
}

char circuit::addRes(const char* name, const char* node1, const char* node2, double Rvalue)
{
	int n1, n2;

	if(Rvalue <= 0) return 3;
	
	n1 = nodePool->findorPushNode(node1);
	n2 = nodePool->findorPushNode(node2);

	if(n1==n2) return 1;

	R->pushBranch(n1, n2, 1.0/Rvalue);
	return 0;
}

char circuit::addCap(const char* name, const char* node1, const char* node2, double Cvalue)
{
	int n1, n2;

	if(Cvalue <= 0) return 3;
	
	n1 = nodePool->findorPushNode(node1);
	n2 = nodePool->findorPushNode(node2);
	//printf("row_no %d\n",nodePool->getNode(n1)->row_no);

	if(n1==n2) return 1;

	C->pushBranch(n1, n2, Cvalue);
	return 0;
}

char circuit::addselfInc(const char* name, const char* node1, const char* node2, double Lvalue)
{
  int n1, n2;

  if( Lvalue<=0) return 3;

  n1 = nodePool->findorPushNode(node1);
  n2 = nodePool->findorPushNode(node2);

  if(n1 == n2) return 1;

  L->pushBranch(n1, n2, Lvalue);
  return 0;
}


char circuit::addIsrc(const char* name, const char* node1, const char* node2, int Iwave)
{
	int n1, n2;

	n1 = nodePool->findorPushNode(node1);
	n2 = nodePool->findorPushNode(node2);

	if(n1==n2) return 1;

	I->pushBranch(n1, n2, Iwave);
	return 0;

}

char circuit::addVsrc(const char* name, const char* node1, const char* node2, int Vwave)
{
	int n1, n2;

	n1 = nodePool->findorPushNode(node1);
	n2 = nodePool->findorPushNode(node2);

	if(n1==n2) return 1;

	V->pushBranch(n1, n2, Vwave);
	return 0;
}

void circuit::addPort(const char* name)
{
	nodePool->pushPort(name);
}

/*void circuit::addIC(const char* name, int value_pt)
{
	nodePool->pushICnode(name, value_pt);
	}*/


void circuit::printRes()
{
  R->printBranch();
}

void circuit::printCap()
{
  C->printBranch();
}
