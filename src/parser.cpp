
/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: parser.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Ning Mi, Xue-Xin Liu
 * 
 *    Functions: parser--read circuit information from spice file 
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "parser.h"
#include "matrix.h"
//#include "gpuData.h"

double StrToNum(char* strnum)
{
  double value;
  char* endptr;

  value = strtod(strnum, &endptr);

  switch(endptr[0])
    {
    case 'T': case 't':
      return value*pow(10.0,12);
    case 'G': case 'g':
      return value*pow(10.0, 9);
    case 'K': case 'k':
      return value*pow(10.0, 3);
    case 'M': case 'm':
      switch(endptr[1])
	{
	case 'E': case 'e':
	  return (value*pow(10.0, 6));
	default:
	  return (value*pow(10.0, -3));
	}
    case 'U': case 'u':
      return value*pow(10.0,(int)-6);
    case 'n': case 'N':
      return value*pow(10.0,(int)-9);

    case 'p': case 'P':
      return value*pow(10.0,(int)-12);
    case 'f': case 'F':
      return value*pow(10.0,(int)-15);
    case 'e': case 'E':
      return value;
    default:
      return value;
    }
}

void parser(const char* filename, double& tstep, double& tstop, int& nVS, int& nIS, int& nL, NodeList* nodePool)
{
  char* strline;
  char* node1, *node2;
  char *tstepstr, *tstopstr;
  char *portstr;
  char *incfilename, *tmp_incfilename;

  int num_R,num_C,num_L,num_V,num_I,nport;
  int n1, n2;
  int pos;

  int i,j;

  FILE* fid_tmp = NULL;

  FILE* fid = fopen(filename, "r");

  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));

  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  portstr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  tstepstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  tstopstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  // get the director of file

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }
  /*for(j=0;j<=pos;j++){
    incfilename[j] = filename[j];
    }*/



  //get number of R,C,L,I,V, construct nodelist
  num_R = 0;
  num_C = 0;
  num_L = 0;
  num_V = 0;
  num_I = 0;
 
 

  while(!feof(fid) || (fid_tmp!=NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //      printf("%s\n",strline);
    }
    else{
      //if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE, fid_tmp);
      //      printf("%s\n",strline);
    }

    switch(strline[0])
      {
      case'R': case'r':
	num_R++;
	if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	  n1 = nodePool->findorPushNode(node1);
	  n2 = nodePool->findorPushNode(node2);
	}
	//printf("In R branch with %s and %s, total node number is %d.\n", node1, node2, nodePool->numNode());
	break;
	
      case'C': case'c':
	num_C++;
	if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	  n1 = nodePool->findorPushNode(node1);
	  n2 = nodePool->findorPushNode(node2);
	}
	//printf("In C branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'L': case'l':
	num_L++;
	if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	  n1 = nodePool->findorPushNode(node1);
	  n2 = nodePool->findorPushNode(node2);
	}
	//printf("In L branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'V': case'v':
	num_V++;
	if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	  n1 = nodePool->findorPushNode(node1);
	  n2 = nodePool->findorPushNode(node2);
	}
	//printf("In V branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'I': case'i':
	num_I++;
	if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	  n1 = nodePool->findorPushNode(node1);
	  n2 = nodePool->findorPushNode(node2);
	}
	//printf("In I branch, total node number is %d.\n", nodePool->numNode());
	break;
	
      case'.':
	if(strline[1]=='t'){//get simulation start and stop time
	  if(sscanf(strline, "%*s %s %s", tstepstr, tstopstr)==2){
	    tstep = StrToNum(tstepstr);
	    tstop = StrToNum(tstopstr);
	  }
	}
	else if (strline[1]=='p' && strline[2]=='r'){ //get port name
	  i = 0;
	  while(strline[i]!='\0'){
	    if(strline[i]=='('){
	      i++;
	      j = 0;
	      while(strline[i]!=')'){
		portstr[j++] = strline[i++];
	      }
	      portstr[j] = '\0';
	      nodePool->pushPort(portstr);
	      //cir->addPort( portstr );
	    }
	    i++;
	  }
	}
	else if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    /*i = 0;
	    for(j=0;j<=pos;j++){
	      incfilename[j] = filename[i];
	      i++;
	      }*/
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';

	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	break;

      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  nIS = num_I;
  nVS = num_V;
  nL = num_L;


  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(portstr);
  free(tstepstr);
  free(tstopstr);

}



void parser_sub(const char* filename, double& tstep, double& tstop, int& num_subnode, int& nVS, int& nIS, int& nL, NodeList* nodePool)
{
  char* strline;
  char* node1, *node2;
  char *tstepstr, *tstopstr;
  char *portstr;
  char *incfilename, *tmp_incfilename;
  char *subcktportstr, *subcktportname;

  int num_R,num_C,num_L,num_V,num_I,nport;
  int n1, n2;
  int pos;
  int spacemark;

  int subportnum =0;
  int subcktportnum = 0;	
  int i,j,p;

  bool subckt=false, mainckt=false, subctn=false;
  NodeList *subnode;

  FILE* fid_tmp = NULL;

  FILE* fid = fopen(filename, "r");

  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));

  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  portstr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  subcktportstr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  subcktportname = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  tstepstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  tstopstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  // get the director of file

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';  }
  /*for(j=0;j<=pos;j++){
    incfilename[j] = filename[j];
    }*/



  //get number of R,C,L,I,V, construct nodelist
  num_R = 0;
  num_C = 0;
  num_L = 0;
  num_V = 0;
  num_I = 0;
 
 

  while(!feof(fid) || (fid_tmp!=NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //      printf("%s\n",strline);
    }
    else{
      //if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE, fid_tmp);
      //      printf("%s\n",strline);
    }

    switch(strline[0])
      {
      case'R': case'r':
	num_R++;
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	    n1 = nodePool->findorPushNode(node1);
	    n2 = nodePool->findorPushNode(node2);
	  }
	}
	else{
	  if(subctn == true) subctn = false;

	  if(sscanf(strline, "%*s %s %s", node1, node2) == 2){
	    n1 = subnode->findorPushNode(node1);
	    n2 = subnode->findorPushNode(node2);
	  }
	}
	//printf("In R branch with %s and %s, total node number is %d.\n", node1, node2, nodePool->numNode());
	break;
	
      case'C': case'c':
	num_C++;
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	    n1 = nodePool->findorPushNode(node1);
	    n2 = nodePool->findorPushNode(node2);
	  }
	}
	else{
	  if(subctn == true) subctn = false;
	  if(sscanf(strline, "%*s %s %s", node1, node2) == 2){
	    n1 = subnode->findorPushNode(node1);
	    n2 = subnode->findorPushNode(node2);
	  }
	}
	//printf("In C branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'L': case'l':
	num_L++;
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	    n1 = nodePool->findorPushNode(node1);
	    n2 = nodePool->findorPushNode(node2);
	  }
	}
	else{
	  if(subctn == true) subctn = false;
	  if(sscanf(strline, "%*s %s %s", node1, node2) == 2){
	    n1 = subnode->findorPushNode(node1);
	    n2 = subnode->findorPushNode(node2);
	  }
	}
	//printf("In L branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'V': case'v':
	num_V++;
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	    n1 = nodePool->findorPushNode(node1);
	    n2 = nodePool->findorPushNode(node2);
	  }
	}
	else{
	  if(subctn == true) subctn = false;
	  if(sscanf(strline, "%*s %s %s", node1, node2) == 2){
	    n1 = subnode->findorPushNode(node1);
	    n2 = subnode->findorPushNode(node2);
	  }
	}
	//printf("In V branch, total node number is %d.\n", nodePool->numNode());
	break;

      case'I': case'i':
	num_I++;
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %*s", node1, node2) == 2){
	    n1 = nodePool->findorPushNode(node1);
		nodePool->pushTCNode(node1);
	    n2 = nodePool->findorPushNode(node2);
		nodePool->pushTCNode(node2);
	  }
	}
	else{
	  if(subctn == true) subctn = false;
	  if(sscanf(strline, "%*s %s %s", node1, node2) == 2){
	    n1 = subnode->findorPushNode(node1);
		nodePool->pushTCNode(node1);
	    n2 = subnode->findorPushNode(node2);
		nodePool->pushTCNode(node2);
	  }
	}
	//printf("In I branch, total node number is %d.\n", nodePool->numNode());
	break;
	
      case'X': case'x':
	mainckt = true;
	subcktportnum = 0;
	if(sscanf(strline, "%*s %s", subcktportstr) == 1){
	  i = 0;
	  while(strline[i]!= ' ')
	    i++;
	  while(strline[i]!= '\0'){
	    j = 0;
	    while(strline[i] != ' ' && strline[i]!='\n') //copy a port name
	      subcktportname[j++] = strline[i++]; 
	    subcktportname[j] = '\0';
	    if(j != 0){
	      n1 = nodePool->findorPushNode(subcktportname);
              subcktportnum++;
	    }
	    i++;
	  }
	  subcktportname[0] = '\0';
	}
	break;

      case'+':
	if(mainckt == true){ //read port name in X.. subckt in mainckt
	  i = 2;
	  while(strline[i] != '\0'){
	    j = 0;
	    if(subcktportname[0] != '\0'){
	      n1 = nodePool->findorPushNode(subcktportname);
	      subcktportnum++;
            }
	    while(strline[i] != ' ' && strline[i]!='\n'){
	      subcktportname[j++] = strline[i++];
	    }
	    subcktportname[j] = '\0';
	    i++;
	    //	    if(subcktportname[0]=='P')
	      //	      printf("maincktportnum: %d\n",subcktportnum);
	  }
	}
	else if(subctn == true){ // read port name in subckt
	  i = 2;
	  p = 0;
	  while(strline[i]!='\0'){
	    while(strline[i]!=' ' && strline[i]!='\n'){
	      subcktportname[p++] = strline[i++];
	    }
	    if(p != 0){
	      subcktportname[p] = '\0';
	      subnode->findorPushNode(subcktportname);
	      subportnum++;
	      p = 0;
	    }
	    i++;
	  }	  
	}
	break;

      case'.':
	if(strncmp(strline, ".tran", 5)==0 || strncmp(strline, ".TRAN", 5)==0){//get simulation start and stop time
	  if(sscanf(strline, "%*s %s %s", tstepstr, tstopstr)==2){
	    tstep = StrToNum(tstepstr);
	    tstop = StrToNum(tstopstr);
	  }
	}
	else if (strncmp(strline, ".print", 6)==0 ){ //get port name
	  i = 0;
	  //if (strline[7] != 'v' && strline[7] != 'V' && strline[7] != 'i' && strline[7] != 'I')
	  if (strline[12] != 'v' && strline[12] != 'V' && strline[12] != 'i' && strline[12] != 'I')
	    printf("Invalid command: %s\n",strline);
	  else{
	    while(strline[i]!='\0'){
	      if(strline[i]=='('){
		i++;
		j = 0;
		while(strline[i]!=')'){
		  portstr[j++] = strline[i++];
		}
		portstr[j] = '\0';
		nodePool->pushPort(portstr);
		//cir->addPort( portstr );
	      }
	      i++;
	    }
	  }
	}
	else if (strncmp(strline, ".include", 8)==0 || strncmp(strline, ".INCLUDE", 8)==0){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    /*i = 0;
	    for(j=0;j<=pos;j++){
	      incfilename[j] = filename[i];
	      i++;
	      }*/
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';

	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	else if(strncmp(strline, ".SUBCKT", 7)==0 || strncmp(strline, ".subckt", 7)==0)
	  {
	    subckt = true;
	    subctn = true;
	    mainckt = false;
	    subportnum = 0;
	    if((subnode = new NodeList) == NULL){
	      printf("Out of memory!\n");exit(1);
	    }
	    i = 0;
	    spacemark = 0;
	    while(spacemark < 2 && strline[i]!='\n'){
	      if(strline[i] == ' '){
		spacemark++;
	      }
	      i++;
	    }
	    p = 0;
	    while(strline[i]!='\0'){
	      while(strline[i]!=' ' && strline[i]!='\n'){
		subcktportname[p++] = strline[i++];
	      }
	      if(p != 0){
		subcktportname[p] = '\0';
		subnode->findorPushNode(subcktportname);
		subportnum++;
		p = 0;
	      }
	      i++;
	    }
	  }
	else if((subckt == true) && (strncmp(strline, ".ends", 5) == 0 || strncmp(strline, ".ENDS", 5) == 0))
	  {
	    subckt = false;
	    num_subnode = num_subnode + subnode->numNode()-subportnum;
	    //	    printf("port num: %d\n", subportnum);
	    delete subnode;
	    subportnum = 0;
	  }
	break;

      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  nIS = num_I;
  nVS = num_V;
  nL = num_L;


  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(portstr);
  free(tstepstr);
  free(tstopstr);
  free(subcktportstr);
  free(subcktportname);

}


void stamp_sub(const char* filename, int nL, int nIS, int nVS, int& nNodes,
	       double& tstep, double& tstop, Source *VS, Source *IS,
	       matrix* G, matrix* C, matrix* B, NodeList* nodePool,
	       gpuETBR *myGPUetbr)
{
  char* strline;
  char* node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *incfilename, *tmp_incfilename;
  char *istr, *istr_tmp;
  char *subcktportname, *subcktportstr;

  double ptime, value, Rvalue;

  int num_point;
  int n1, n2, n1_tmp, n2_tmp;
  int num_L_tmp, num_I_tmp, num_V_tmp;
  int index_i, index_j;
  int i,j,k,p;
  int tmp_portnum,  subportnum, subnodenum;
  int mark_value;
  int pos;
  int num_mainckt, num_subckt;
  int inc_index;
  int spacemark;
  int substart, subend;
  int last_npoint_I, last_npoint_V;

  bool isI, isV;
  bool dc, pwl;
  bool subckt, mainckt, subctn=false;
  bool subgnd, subportgnd;

  NodeList* subNode;
  vector<subelement> subData;
  //subelement subele;

  //NodeList::Node* node_tmp;


  FILE* fid_tmp=NULL;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));


  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr_tmp = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  subcktportname = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  subcktportstr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  // get the director of file

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;	  
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }


  /* stamp circuit to G,C,B and waveform */   
  num_mainckt = nodePool->numNode(); // node number in main ckt
  num_subckt = 0;    //node number in subckt
  num_L_tmp = 0;
  num_I_tmp = -1;
  num_V_tmp = -1;
  while(!feof(fid) || (fid_tmp!=NULL && !feof(fid_tmp))){
    //  while(!feof(fid)||(fid_tmp != NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //printf("%s\n",strline);
    }
    else{
      if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      //      printf("%s\n",strline);
    }


    switch(strline[0])
      {
	//  ************ Resistor ****************
      case'R': case'r':
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	    Rvalue = StrToNum(valuestr);
	    value = 1.0/Rvalue;
	    n1_tmp = nodePool->findorPushNode(node1);
	    n2_tmp = nodePool->findorPushNode(node2);
	    n1 = nodePool->getNode(n1_tmp)->row_no;
	    n2 = nodePool->getNode(n2_tmp)->row_no;
	    if (n1 != GNDNODE) G->pushEntry(n1, n1, value);
	    if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
	    if (n1 != GNDNODE && n2 != GNDNODE)
	      {
		G->pushEntry(n1,n2,-value);
		G->pushEntry(n2,n1,-value);
	      }
	  }
	  else
	    printf("Fail in obtaining resistence value.\n");
	}
	else{
	  if(subctn == true)
	    subctn = false;
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	    Rvalue = StrToNum(valuestr);
	    value = 1/Rvalue;
	    subNode->findorPushNode(node1);
	    subNode->findorPushNode(node2);
	    if((subportgnd == false) && (!strcmp(node1, "0") || !strcmp(node1, "gnd") || !strcmp(node2, "0") || !strcmp(node2, "gnd")) )
	      subgnd = true;
	    subelement subele;
	    subele.type = 'R';
	    subele.node1 = (char*)malloc(strlen(node1)*sizeof(char));
	    subele.node2 = (char*)malloc(strlen(node2)*sizeof(char));
	    strcpy(subele.node1, node1);
	    strcpy(subele.node2, node2);
	    subele.value = value;
	    subData.push_back(subele);
	  }
	}
	break;

	//  ************ Capacitor ***************
      case'C': case'c':
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	    value = StrToNum(valuestr);
	    n1_tmp = nodePool->findorPushNode(node1);
	    n2_tmp = nodePool->findorPushNode(node2);
	    n1 = nodePool->getNode(n1_tmp)->row_no;
	    n2 = nodePool->getNode(n2_tmp)->row_no;
	    if (n1 != GNDNODE) C->pushEntry(n1, n1, value);
	    if (n2 != GNDNODE) C->pushEntry(n2, n2, value);
	    if (n1 != GNDNODE && n2 != GNDNODE)
	      {
		C->pushEntry(n1,n2,-value);
		C->pushEntry(n2,n1,-value);
	      }
	  }
	  else
	    printf("Fail in obtaining capacitor value.\n");
	}
	else{
	  if(subctn == true)
	    subctn = false;
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	    value = StrToNum(valuestr);
	    subNode->findorPushNode(node1);
	    subNode->findorPushNode(node2);
	    if(subportgnd == false && (!strcmp(node1, "0") || !strcmp(node1, "gnd") || !strcmp(node2, "0") || !strcmp(node2, "gnd")) )
	      subgnd = true;
	    subelement subele;
	    subele.type = 'C';
	    subele.node1 = (char*)malloc(strlen(node1)*sizeof(char));
	    subele.node2 = (char*)malloc(strlen(node2)*sizeof(char));
	    strcpy(subele.node1, node1);
	    strcpy(subele.node2, node2);
	    subele.value = value;
	    subData.push_back(subele);
	  }
	}
	break;

	//  ************ Inductance ****************
      case'L': case'l':
	if(subckt == false){
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	    value = StrToNum(valuestr);
	    n1_tmp = nodePool->findorPushNode(node1);
	    n2_tmp = nodePool->findorPushNode(node2);
	    n1 = nodePool->getNode(n1_tmp)->row_no;
	    n2 = nodePool->getNode(n2_tmp)->row_no;
	    index_i = nNodes + num_L_tmp;
	    
	    if( n1!=GNDNODE )
	      { G->pushEntry(index_i,n1,-1); G->pushEntry(n1,index_i,1); }
	    if( n2!=GNDNODE )
	      { G->pushEntry(index_i,n2,1); G->pushEntry(n2,index_i,-1); }  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
	    node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
	    
	    C->pushEntry(index_i, index_i, value);
	    
	    num_L_tmp++;
	  }
	  else
	    printf("Fail in obtaining inductance value.\n");
	}
	break;

	// ************ Voltage *******************
      case'V': case'v':
	if(subckt == false){
	  num_V_tmp++;
	  isV = true; isI = false;
	  if(sscanf(strline, "%*s %s %s %s",  node1, node2, istr) == 3){
	    n1_tmp = nodePool->findorPushNode(node1);
	    n2_tmp = nodePool->findorPushNode(node2);
	    n1 = nodePool->getNode(n1_tmp)->row_no;
	    n2 = nodePool->getNode(n2_tmp)->row_no;
	    
	    index_i = nNodes + nL + num_V_tmp;
	    index_j = num_V_tmp;
	    
	    if(n1 != GNDNODE) {
	      G->pushEntry(n1, index_i, 1);
	      G->pushEntry(index_i, n1, -1);
	    }
	    if(n2 != GNDNODE) {
	      G->pushEntry(n2, index_i, -1);
	      G->pushEntry(index_i, n2, 1);
	    }
	    B->pushEntry(index_i, index_j, -1);
	    
	    if(istr[0] == 'P'||istr[0] == 'p'){ //PWL
	      if(num_V_tmp != 0 && pwl == true && dc == false){
		VS[num_V_tmp-1].time.set_size(num_point,true);
		VS[num_V_tmp-1].value.set_size(num_point,true);
	      }
	      pwl = true;
	      dc = false;
	      if(num_V_tmp == nVS-1){
		last_npoint_V = 0;
	      }
	      num_point = 0;
	      if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
		break;
	      else{ 
		i = 0;
		while(strline[i] != '(')
		  i++;
		i++;
		j = 0;
		k = 0;
		
		mark_value = 0;
		while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		  if(strline[i] != ' ' && mark_value == 0){ //input timestr
		    timestr[j] = strline[i];
		    i++;
		    j++;
		  }
		  else{
		    mark_value = 1;
		    i++;
		    if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		      valuestr[k] = strline[i];
		      k++;
		    } 
		    else{
		      valuestr[k] = '\0';
		      timestr[j] = '\0';
		      k = 0;
		      j = 0;
		      value = StrToNum(valuestr);
		      ptime = StrToNum(timestr);
		      if (num_point==0){
			VS[num_V_tmp].time.set_size(PWL_SIZE, false);
			VS[num_V_tmp].value.set_size(PWL_SIZE, false);
			if(ptime!=0){
			  VS[num_V_tmp].time(num_point) = 0;
			  VS[num_V_tmp].value(num_point) = value;
			  if(num_V_tmp == nVS-1){
			    last_npoint_V++;
			  }

			  num_point++;	    
			}
			VS[num_V_tmp].time(num_point) = ptime;
			VS[num_V_tmp].value(num_point) = value;
			if(num_V_tmp == nVS-1){
			  last_npoint_V++;
			}
			num_point++;
		      }
		      else{
			VS[num_V_tmp].time(num_point) = ptime;
			VS[num_V_tmp].value(num_point) = value;
			if(num_V_tmp == nVS-1){
			  last_npoint_V++;
			}
			num_point++;
		      }
		      mark_value = 0;
		      i++;
		    }
		  }
		}
	      }
	    }
	    else{ //DC
	      dc = true;
	      pwl = false;

	      value = StrToNum(istr);
	      
	      VS[num_V_tmp].time.set_size(2, false);
	      VS[num_V_tmp].value.set_size(2, false);
	      
	      VS[num_V_tmp].time(0) = 0;
	      VS[num_V_tmp].time(1) = tstop;
	      VS[num_V_tmp].value(0) = value;
	      VS[num_V_tmp].value(1) = value;
	      
	    }
	  }
	}
	break;

	// ******************** Current source ******************
      case'I': case'i':
	if(subckt == false){
	  num_I_tmp++;
	  isV = false; isI = true;
	  if(sscanf(strline, "%*s %s %s %s", node1, node2, istr) == 3){
	    n1_tmp = nodePool->findorPushNode(node1);
	    n2_tmp = nodePool->findorPushNode(node2);
	    n1 = nodePool->getNode(n1_tmp)->row_no;
	    n2 = nodePool->getNode(n2_tmp)->row_no;
	    
	    index_j = nVS + num_I_tmp; 
	    
	    if(n1 != GNDNODE) {
	      B->pushEntry(n1,index_j,-1);
	    }
	    if(n2 != GNDNODE) {
	      B->pushEntry(n2,index_j,1);
	    }
	    
	    if(istr[0] == 'P'||istr[0] == 'p'){ //PWL
	      if(num_I_tmp != 0 && dc == false && pwl == true){
		IS[num_I_tmp-1].time.set_size(num_point,true);
		IS[num_I_tmp-1].value.set_size(num_point,true);
	      }
	      dc = false;
	      pwl = true;
	      if(num_I_tmp == nIS - 1)
		last_npoint_I = 0;
	      num_point = 0;
	      if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
		break;
	      else{ 
		i = 0;
		while(strline[i] != '(')
		  i++;
		i++;
		j = 0;
		k = 0;
		
		mark_value = 0;
		while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		  if(strline[i] != ' ' && mark_value == 0){ //input timestr
		    timestr[j] = strline[i];
		    i++;
		    j++;
		  }
		  else{
		    mark_value = 1;
		    i++;
		    if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		      valuestr[k] = strline[i];
		      k++;
		    } 
		    else{
		      valuestr[k] = '\0';
		      timestr[j] = '\0';
		      k = 0;
		      j = 0;
		      value = StrToNum(valuestr);
		      ptime = StrToNum(timestr);
		      if (num_point==0){
			IS[num_I_tmp].time.set_size(PWL_SIZE, false);
			IS[num_I_tmp].value.set_size(PWL_SIZE, false);
			if(ptime!=0){
			  IS[num_I_tmp].time(num_point) = 0;
			  IS[num_I_tmp].value(num_point) = value;
			  if(num_I_tmp == nIS - 1)
			    last_npoint_I++;
			  num_point++;	    
			}
			IS[num_I_tmp].time(num_point) = ptime;
			IS[num_I_tmp].value(num_point) = value;
			if(num_I_tmp == nIS - 1)
			  last_npoint_I++;
			num_point++;
		      }
		      else{
			IS[num_I_tmp].time(num_point) = ptime;
			IS[num_I_tmp].value(num_point) = value;
			if(num_I_tmp == nIS - 1)
			  last_npoint_I++;
			num_point++;
		      }
		      mark_value = 0;
		      i++;
		    }
		  }
		}
		
	      }
	    }
	    else{ //DC
	      dc = true;
	      pwl = false;
	      value = StrToNum(istr);
	      IS[num_I_tmp].time.set_size(2, false);
	      IS[num_I_tmp].value.set_size(2, false);
	      
	      IS[num_I_tmp].time(0) = 0;
	      IS[num_I_tmp].time(1) = tstop;
	      IS[num_I_tmp].value(0) = value;
	      IS[num_I_tmp].value(1) = value;
	      
	    }
	  }
	}
	break;

	// ************ subckt used in mainckt
      case'X':
	mainckt = true;
	tmp_portnum = 0;
	if(sscanf(strline, "%*s %s", subcktportstr) == 1){
	  i = 0;
	  while(strline[i]!= ' ')
	    i++;
	  while(strline[i]!= '\0'){
	    p = 0;
	    while(strline[i] != ' ' && strline[i]!= '\n') //copy a port name
	      subcktportname[p++] = strline[i++]; 
	    subcktportname[p] = '\0';
	    if(p != 0){
	      if(tmp_portnum < subportnum){
		if(subcktportname[0] == 'P'){ // end of subckt instance
		  mainckt = false;
		  tmp_portnum++;
		  break;
		}
		n1_tmp = nodePool->findorPushNode(subcktportname);
		n1 = nodePool->getNode(n1_tmp)->row_no;
		subNode->setRowNum(tmp_portnum, n1);
		tmp_portnum++;
	      }
	      else if(tmp_portnum == subportnum){
		inc_index = num_mainckt + num_subckt;
		if(subportgnd == true){//gnd in the subckt port
		  substart = tmp_portnum+1;
		  subend = subNode->numNode()+1;
		}
		else if(subgnd == true){//gnd inside subckt
		  substart = tmp_portnum;
		  subend = subNode->numNode()+1;
		}
		else{//no gnd in subckt or in subckt port
		  substart = tmp_portnum;
		  subend = subNode->numNode();
		}
		for(j = substart; j<subend; j++){
		  if(subNode->getNode(j)->row_no != GNDNODE)
		    subNode->incRowNum(j,inc_index);
		}
		for(j = 0; j<subData.size(); j++){
		  n1_tmp = subNode->findorPushNode(subData[j].node1);
		  n2_tmp = subNode->findorPushNode(subData[j].node2);
		  n1 = subNode->getNode(n1_tmp)->row_no;
		  n2 = subNode->getNode(n2_tmp)->row_no;
		  if(subData[j].type == 'R'){
		    if(n1 != GNDNODE) G->pushEntry(n1, n1, value);
		    if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
		    if (n1 != GNDNODE && n2 != GNDNODE)
		      {
			G->pushEntry(n1,n2,-value);
			G->pushEntry(n2,n1,-value); 
		      }
		  }
		  else if(subData[j].type == 'C'){
		    if (n1 != GNDNODE) C->pushEntry(n1, n1, value);
		    if (n2 != GNDNODE) C->pushEntry(n2, n2, value);
		    if (n1 != GNDNODE && n2 != GNDNODE)
		      {
			C->pushEntry(n1,n2,value);
			C->pushEntry(n2,n1,value);
		      }
		  }
		}
		num_subckt = num_subckt + subNode->numNode() - subportnum;
		delete subNode;
		for (j = 0; j<subData.size(); j++){
		  free(subData[j].node1);
		  free(subData[j].node2);
		}
		subData.clear();
		if(subcktportname[0]=='P')
		  mainckt = false;
		if(subgnd == true)
		  subgnd = false;
		if(subportgnd == true)
		  subportgnd = false;
		tmp_portnum++;
		//break;
	      }
	      else{
		if(subcktportname[0]=='P')
		  mainckt = false;

	      }
	    }
	    i++;
	  }
	}
	break;

	//  ************ PWL input ***************
      case'+':
	if(subctn == true) {//subcircuit port name
	  i = 2;
	  p = 0;
	  while(strline[i]!='\0'){
	    while(strline[i]!=' ' && strline[i]!='\n'){
	      subcktportname[p++] = strline[i++];
	    }
	    subcktportname[p] = '\0';
	    if(p != 0){
	      subNode->findorPushNode(subcktportname);
	      subportnum++;
	      p = 0;
	      if(!strcmp(subcktportname,"0")||!strcmp(subcktportname,"gnd"))
		subportgnd = true;
	    }
	    i++;
	  }	  
	}
	else if(mainckt == true){ //main circuit port name using subckt
	  i = 2;
	  while(strline[i] != '\0'){
	    p = 0;
	    while(strline[i] != ' ' && strline[i]!= '\n'){
	      subcktportname[p++] = strline[i++];
	    }
	    subcktportname[p] = '\0';
	    if(p != 0){
	      if(tmp_portnum < subportnum){
		if(subcktportname[0] == 'P'){
		  mainckt = false;
		  tmp_portnum++;
		  /*printf("Error: subckt ports don't match with mainckt ports");
		    exit(1);*/
		  break;
		}

		n1_tmp = nodePool->findorPushNode(subcktportname);
		n1 = nodePool->getNode(n1_tmp)->row_no;
		subNode->setRowNum(tmp_portnum, n1);
		tmp_portnum++;
	      }
	      else if(tmp_portnum == subportnum){
		inc_index = num_mainckt + num_subckt - subportnum;
		if(subportgnd == true){//gnd in the subckt port
		  substart = tmp_portnum+1;
		  subend = subNode->numNode()+1;
		}
		else if(subgnd == true){//gnd inside subckt
		  substart = tmp_portnum;
		  subend = subNode->numNode()+1;
		}
		else{//no gnd in subckt or in subckt port
		  substart = tmp_portnum;
		  subend = subNode->numNode();
		}
		for(j = substart; j<subend; j++){
		  if(subNode->getNode(j)->row_no != GNDNODE)
		    subNode->incRowNum(j,inc_index);
		}
		for(j = 0; j<subData.size(); j++){
		  n1_tmp = subNode->findorPushNode(subData[j].node1);
		  n2_tmp = subNode->findorPushNode(subData[j].node2);
		  n1 = subNode->getNode(n1_tmp)->row_no;
		  n2 = subNode->getNode(n2_tmp)->row_no;
		  if(subData[j].type == 'R'){
		    if(n1 != GNDNODE) G->pushEntry(n1, n1, value);
		    if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
		    if (n1 != GNDNODE && n2 != GNDNODE)
		      {
			G->pushEntry(n1,n2,-value);
			G->pushEntry(n2,n1,-value); 
		      }
		  }
		  else if(subData[j].type == 'C'){
		    if (n1 != GNDNODE) C->pushEntry(n1, n1, value);
		    if (n2 != GNDNODE) C->pushEntry(n2, n2, value);
		    if (n1 != GNDNODE && n2 != GNDNODE)
		      {
			C->pushEntry(n1,n2,value);
			C->pushEntry(n2,n1,value);
		      }
		  }
		}
		num_subckt = num_subckt + subNode->numNode() - subportnum;
		delete subNode;
		for (j = 0; j<subData.size(); j++){
		  free(subData[j].node1);
		  free(subData[j].node2);
		}
		subData.clear();
		if(subcktportname[0]=='P')
		  mainckt = false;
		if(subgnd == true)
		  subgnd = false;
		if(subportgnd == true)
		  subportgnd = false;
		tmp_portnum++;
		//		break;
	      }
	      else{
		if(subcktportname[0]=='P')
		  mainckt = false;
	      }
	    }
	    i++;
	  }
	}
	else{
	  if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){ // pwl for V or I
	  value = StrToNum(valuestr);
	  ptime = StrToNum(timestr);
	  if(isV == true && isI == false){ //PWL for V
	    if (num_point==0){
	      VS[num_V_tmp].time.set_size(PWL_SIZE, false);
	      VS[num_V_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		VS[num_V_tmp].time(num_point) = 0;
		VS[num_V_tmp].value(num_point) = value;
		if(num_V_tmp == nVS - 1)
		  last_npoint_V++;
		num_point++;	    
	      }
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      if(num_V_tmp == nVS - 1)
		last_npoint_V++;
	      num_point++;
	    }
	    else{
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      if(num_V_tmp == nVS - 1)
		last_npoint_V++;
	      num_point++;
	    }
	  }
	  else if(isI == true && isV == false){ //PWL for I
	    if (num_point==0){
	      IS[num_I_tmp].time.set_size(PWL_SIZE, false);
	      IS[num_I_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		IS[num_I_tmp].time(num_point) = 0;
		IS[num_I_tmp].value(num_point) = value;
		if(num_I_tmp == nIS - 1)
		  last_npoint_I++;
		num_point++;	    
	      }
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      if(num_I_tmp == nIS - 1)
		last_npoint_I++;
	      num_point++;
	    }
	    else{
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      if(num_I_tmp == nIS - 1)
		last_npoint_I++;
	      num_point++;
	    }
	  }
	  else{
	    printf("PWL input error.\n"); exit(1);
	  }
	  }
	}
	break;

        //  ********** .include ********************
      case'.':
	if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';


	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	else if(strncmp(strline, ".SUBCKT", 7)==0 || strncmp(strline, ".subckt", 7)==0)
	  {
	    subckt = true;
	    subportnum = 0;
	    //mainckt = false;
	    subctn = true; //mark +
	    if((subNode = new NodeList) == NULL){
	      printf("Out of memory!\n");exit(1);
	    }
	    i = 0;
	    spacemark = 0;
	    while(spacemark < 2 && strline[i]!='\0'){
	      if(strline[i] == ' '){
		spacemark++;
	      }
	      i++;
	    }
	    p = 0;
	    while(strline[i]!='\0'){
	      while(strline[i]!=' ' && strline[i]!='\n'){
		subcktportname[p++] = strline[i++];
	      }
	      subcktportname[p] = '\0';
	      if(p != 0){
		subNode->findorPushNode(subcktportname);
		subportnum++;
		p = 0;
		if(!strcmp(subcktportname,"0")||!strcmp(subcktportname,"gnd"))
		  subportgnd = true;
	      }
	      i++;
	    }
	  }

	else if((subckt == true) && (strncmp(strline, ".ends", 5) == 0 || strncmp(strline, ".ENDS", 5) == 0))
	  {
	    subckt = false;
	  }
	else if(strncmp(strline, ".end", 4) == 0 || strncmp(strline, ".END", 4) == 0)
	  {
	    if(last_npoint_I != 0){
	      IS[num_I_tmp].time.set_size(last_npoint_I,true);
	      IS[num_I_tmp].value.set_size(last_npoint_I,true);
	    }
	    if(last_npoint_V != 0){
	      VS[num_V_tmp].time.set_size(last_npoint_V,true);
	      VS[num_V_tmp].value.set_size(last_npoint_V,true);
	    }
	  }
	break;

      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  G->sort();
  C->sort();
  B->sort();

  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(istr);
  free(istr_tmp);
  free(timestr);
  free(stimestr);
  free(valuestr);
  free(subcktportstr);
  free(subcktportname);

}




void stamp(const char* filename, int nL, int nVS, int& nNodes, double& tstep, double& tstop, Source *VS, Source *IS, matrix* G, matrix* C, matrix* B, NodeList* nodePool)
{
  char* strline;
  char* node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *incfilename, *tmp_incfilename;
  char *istr, *istr_tmp;

  double ptime, value, Rvalue;

  int num_point;
  int n1, n2, n1_tmp, n2_tmp;
  int num_L_tmp, num_I_tmp, num_V_tmp;
  int index_i, index_j;
  int i,j,k;
  int mark_value;
  int pos;

  bool isI, isV;

  //NodeList::Node* node_tmp;


  FILE* fid_tmp=NULL;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));


  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr_tmp = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  // get the director of file

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }


  /* stamp circuit to G,C,B and waveform */   

  num_L_tmp = 0;
  num_I_tmp = -1;
  num_V_tmp = -1;
  while(!feof(fid) || (fid_tmp!=NULL && !feof(fid_tmp))){
    //  while(!feof(fid)||(fid_tmp != NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //printf("%s\n",strline);
    }
    else{
      if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      //      printf("%s\n",strline);
    }


    switch(strline[0])
      {
	//  ************ Resistor ****************
      case'R': case'r':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  Rvalue = StrToNum(valuestr);
	  value = 1.0/Rvalue;
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  if (n1 != GNDNODE) G->pushEntry(n1, n1, value);
	  if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
	  if (n1 != GNDNODE && n2 != GNDNODE)
	    {
	      G->pushEntry(n1,n2,-value);
	      G->pushEntry(n2,n1,-value);
	    }
	}
	else
	  printf("Fail in obtaining resistence value.\n");

	break;

	//  ************ Capacitor ***************
      case'C': case'c':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  value = StrToNum(valuestr);
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  if (n1 != GNDNODE) C->pushEntry(n1, n1, value);
	  if (n2 != GNDNODE) C->pushEntry(n2, n2, value);
	  if (n1 != GNDNODE && n2 != GNDNODE)
	    {
	      C->pushEntry(n1,n2,-value);
	      C->pushEntry(n2,n1,-value);
	    }
	}
	else
	  printf("Fail in obtaining capacitor value.\n");

	break;

	//  ************ Inductance ****************
      case'L': case'l':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  value = StrToNum(valuestr);
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  index_i = nNodes + num_L_tmp;

	  if( n1!=GNDNODE )
	    { G->pushEntry(index_i,n1,-1); G->pushEntry(n1,index_i,1); }
	  if( n2!=GNDNODE )
	    { G->pushEntry(index_i,n2,1); G->pushEntry(n2,index_i,-1); }  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

	  C->pushEntry(index_i, index_i, value);
	  
	  num_L_tmp++;
	}
	else
	  printf("Fail in obtaining inductance value.\n");

	break;

	// ************ Voltage *******************
      case'V': case'v':
	num_V_tmp++;
	isV = true; isI = false;
	if(sscanf(strline, "%*s %s %s %s",  node1, node2, istr) == 3){
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  
	  index_i = nNodes + nL + num_V_tmp;
	  index_j = num_V_tmp;

	  if(n1 != GNDNODE) {
	    G->pushEntry(n1, index_i, 1);
	    G->pushEntry(index_i, n1, -1);
	  }
	  if(n2 != GNDNODE) {
	    G->pushEntry(n2, index_i, -1);
	    G->pushEntry(index_i, n2, 1);
	  }
	  B->pushEntry(index_i, index_j, -1);

	  if(istr[0] == 'P'||istr[0] == 'p'){ //PWL
	    if(num_V_tmp != 0){
	      VS[num_V_tmp-1].time.set_size(num_point,true);
	      VS[num_V_tmp-1].value.set_size(num_point,true);
	    }
	    num_point = 0;
	    if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
	      break;
	    else{ 
	      i = 0;
	      while(strline[i] != '(')
		i++;
	      i++;
	      j = 0;
	      k = 0;

	      mark_value = 0;
	      while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		if(strline[i] != ' ' && mark_value == 0){ //input timestr
		  timestr[j] = strline[i];
		  i++;
		  j++;
		}
		else{
		  mark_value = 1;
		  i++;
		  if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		    valuestr[k] = strline[i];
		    k++;
		  } 
		  else{
		    valuestr[k] = '\0';
		    timestr[j] = '\0';
		    k = 0;
		    j = 0;
		    value = StrToNum(valuestr);
		    ptime = StrToNum(timestr);
		    if (num_point==0){
		      VS[num_V_tmp].time.set_size(PWL_SIZE, false);
		      VS[num_V_tmp].value.set_size(PWL_SIZE, false);
		      if(ptime!=0){
			VS[num_V_tmp].time(num_point) = 0;
			VS[num_V_tmp].value(num_point) = value;
			num_point++;	    
		      }
		      VS[num_V_tmp].time(num_point) = ptime;
		      VS[num_V_tmp].value(num_point) = value;
		      num_point++;
		    }
		    else{
		      VS[num_V_tmp].time(num_point) = ptime;
		      VS[num_V_tmp].value(num_point) = value;
		      num_point++;
		    }
		    mark_value = 0;
		    i++;
		  }
		}
	      }
	    }
	  }
	else{ //DC

	    value = StrToNum(istr);
	    
	    VS[num_V_tmp].time.set_size(2, false);
	    VS[num_V_tmp].value.set_size(2, false);
	    
	    VS[num_V_tmp].time(0) = 0;
	    VS[num_V_tmp].time(1) = tstop;
	    VS[num_V_tmp].value(0) = value;
	    VS[num_V_tmp].value(1) = value;

	}
	}
	break;

	// ******************** Current source ******************
      case'I': case'i':
	num_I_tmp++;
	isV = false; isI = true;
	if(sscanf(strline, "%*s %s %s %s", node1, node2, istr) == 3){
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  
	  index_j = nVS + num_I_tmp; 

	  if(n1 != GNDNODE) {
	    B->pushEntry(n1,index_j,-1);
	  }
	  if(n2 != GNDNODE) {
	    B->pushEntry(n2,index_j,1);
	  }

	  if(istr[0] == 'P'||istr[0] == 'p'){ //PWL
	    if(num_I_tmp != 0){
	      IS[num_I_tmp-1].time.set_size(num_point,true);
	      IS[num_I_tmp-1].value.set_size(num_point,true);
	    }
	    num_point = 0;
	    if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
	      break;
	    else{ 
	      i = 0;
	      while(strline[i] != '(')
		i++;
	      i++;
	      j = 0;
	      k = 0;

	      mark_value = 0;
	      while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		if(strline[i] != ' ' && mark_value == 0){ //input timestr
		  timestr[j] = strline[i];
		  i++;
		  j++;
		}
		else{
		  mark_value = 1;
		  i++;
		  if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		    valuestr[k] = strline[i];
		    k++;
		  } 
		  else{
		    valuestr[k] = '\0';
		    timestr[j] = '\0';
		    k = 0;
		    j = 0;
		    value = StrToNum(valuestr);
		    ptime = StrToNum(timestr);
		    if (num_point==0){
		      IS[num_I_tmp].time.set_size(PWL_SIZE, false);
		      IS[num_I_tmp].value.set_size(PWL_SIZE, false);
		      if(ptime!=0){
			IS[num_I_tmp].time(num_point) = 0;
			IS[num_I_tmp].value(num_point) = value;
			num_point++;	    
		      }
		      IS[num_I_tmp].time(num_point) = ptime;
		      IS[num_I_tmp].value(num_point) = value;
		      num_point++;
		    }
		    else{
		      IS[num_I_tmp].time(num_point) = ptime;
		      IS[num_I_tmp].value(num_point) = value;
		      num_point++;
		    }
		    mark_value = 0;
		    i++;
		  }
		}
	      }
	     
	    }
	  }
	  else{ //DC
	    value = StrToNum(istr);
	    IS[num_I_tmp].time.set_size(2, false);
	    IS[num_I_tmp].value.set_size(2, false);
	    
	    IS[num_I_tmp].time(0) = 0;
	    IS[num_I_tmp].time(1) = tstop;
	    IS[num_I_tmp].value(0) = value;
	    IS[num_I_tmp].value(1) = value;
	    
	  }
	}
	break;

	//  ************ PWL input ***************
      case'+':
	if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){
	  value = StrToNum(valuestr);
	  ptime = StrToNum(timestr);
	  if(isV == true && isI == false){ //PWL for V
	    if (num_point==0){
	      VS[num_V_tmp].time.set_size(PWL_SIZE, false);
	      VS[num_V_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		VS[num_V_tmp].time(num_point) = 0;
		VS[num_V_tmp].value(num_point) = value;
		num_point++;	    
	      }
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      num_point++;
	    }
	    else{
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      num_point++;
	    }
	  }
	  else if(isI == true && isV == false){ //PWL for I
	    if (num_point==0){
	      IS[num_I_tmp].time.set_size(PWL_SIZE, false);
	      IS[num_I_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		IS[num_I_tmp].time(num_point) = 0;
		IS[num_I_tmp].value(num_point) = value;
		num_point++;	    
	      }
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      num_point++;
	    }
	    else{
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      num_point++;
	    }
	  }
	  else{
	    printf("PWL input error.\n"); exit(1);
	  }
	}
	break;

        //  ********** .include ********************
      case'.':
	if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';


	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	break;

      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  G->sort();
  C->sort();
  B->sort();

  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(istr);
  free(istr_tmp);
  free(timestr);
  free(stimestr);
  free(valuestr);

}

void stampG(const char* filename, int nL, int nVS, int nNodes, matrix* G, NodeList* nodePool)
{
  char* strline;
  char* node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *incfilename, *tmp_incfilename;

  double ptime, value, Rvalue;

  int num_point;
  int n1, n2, n1_tmp, n2_tmp;
  int num_L_tmp, num_I_tmp, num_V_tmp;
  int index_i, index_j;
  int i,j;
  int pos;

  bool isI, isV;

  //NodeList::Node* node_tmp;

  FILE* fid_tmp=NULL;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));

  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  /* get the director of file */

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }



  /* stamp circuit to G,C,B and waveform */   

  num_L_tmp = 0;
  num_I_tmp = -1;
  num_V_tmp = -1;

  while(!feof(fid)||(fid_tmp != NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //printf("%s\n",strline);
    }
    else{
      if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      //printf("%s\n",strline);
    }

    switch(strline[0])
      {
	//  ************ Resistor ****************
      case'R': case'r':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  Rvalue = StrToNum(valuestr);
	  value = 1.0/Rvalue;
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  if (n1 != GNDNODE) G->pushEntry(n1, n1, value);
	  if (n2 != GNDNODE) G->pushEntry(n2, n2, value);
	  if (n1 != GNDNODE && n2 != GNDNODE)
	    {
	      G->pushEntry(n1,n2,-value);
	      G->pushEntry(n2,n1,-value);
	    }
	}
	else{
	  printf("%s \n", strline);
	  printf("Fail in obtaining resistence value.\n");
	}
	break;

	//  ************ Inductance ****************
      case'L': case'l':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  value = StrToNum(valuestr);
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  index_i = nNodes + num_L_tmp;

	  if( n1!=GNDNODE )
	    { G->pushEntry(index_i,n1,-1); G->pushEntry(n1,index_i,1); }
	  if( n2!=GNDNODE )
	    { G->pushEntry(index_i,n2,1); G->pushEntry(n2,index_i,-1); }
	  num_L_tmp++;
	}
	else
	  printf("Fail in obtaining inductance value.\n");

	break;

	// ************ Voltage *******************
      case'V': case'v':
	num_V_tmp++;
	isV = true; isI = false;
	if(sscanf(strline, "%*s %s %s",  node1, node2) == 2){
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  
	  index_i = nNodes + nL + num_V_tmp;
	  index_j = num_V_tmp;

	  if(n1 != GNDNODE) {
	    G->pushEntry(n1, index_i, 1);
	    G->pushEntry(index_i, n1, -1);
	  }
	  if(n2 != GNDNODE) {
	    G->pushEntry(n2, index_i, -1);
	    G->pushEntry(index_i, n2, 1);
	  }
	}
	break;

        //  ********** .include ********************
      case'.':
	if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';


	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	break;

      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  G->sort();

  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(timestr);
  free(stimestr);
  free(valuestr);

}

void stampC(const char* filename, int nL, int nVS, int nNodes, matrix* C, NodeList* nodePool)
{
  char* strline;
  char* node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *incfilename, *tmp_incfilename;

  double ptime, value, Rvalue;

  int num_point;
  int n1, n2, n1_tmp, n2_tmp;
  int num_L_tmp, num_I_tmp, num_V_tmp;
  int index_i, index_j;
  int i,j;
  int pos;

  bool isI, isV;

  //NodeList::Node* node_tmp;


  FILE* fid_tmp=NULL;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));

  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  /* get the director of file */

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }

  /* stamp circuit to G,C,B and waveform */   

  num_L_tmp = 0;
  num_I_tmp = -1;
  num_V_tmp = -1;

  while(!feof(fid)||(fid_tmp!=NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //printf("%s\n",strline);
    }
    else{
      if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      //printf("%s\n",strline);
    }


    switch(strline[0])
      {

	//  ************ Capacitor ***************
      case'C': case'c':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  value = StrToNum(valuestr);
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  if (n1 != GNDNODE) C->pushEntry(n1, n1, value);
	  if (n2 != GNDNODE) C->pushEntry(n2, n2, value);
	  if (n1 != GNDNODE && n2 != GNDNODE)
	    {
	      C->pushEntry(n1,n2,-value);
	      C->pushEntry(n2,n1,-value);
	    }
	}
	else
	  printf("Fail in obtaining capacitor value.\n");

	break;

	//  ************ Inductance ****************
      case'L': case'l':
	if(sscanf(strline, "%*s %s %s %s", node1, node2, valuestr) == 3){
	  value = StrToNum(valuestr);
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  index_i = nNodes + num_L_tmp;

	  C->pushEntry(index_i, index_i, value);
	  
	  num_L_tmp++;
	}
	else
	  printf("Fail in obtaining inductance value.\n");

	break;

        //  ********** .include ********************
      case'.':
	if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';


	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	break;


      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  C->sort();

  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(timestr);
  free(stimestr);
  free(valuestr);

}

void stampB(const char* filename, int nL, int nIS, int nVS, int nNodes, double tstop,
	    Source *VS, Source *IS, matrix* B, NodeList* nodePool,
	    gpuETBR *myGPUetbr)
{
  char* strline;
  char* node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *incfilename, *tmp_incfilename;
  char *istr, *istr2, *istr3, *istr4, *istr5, *istr6, *istr7, *istr8;

  char isrcName[128];
  
  double ptime, value, Rvalue;

  int num_point;
  int n1, n2, n1_tmp, n2_tmp;
  int num_L_tmp, num_I_tmp, num_V_tmp;
  int index_i, index_j;
  int i,j,k,mark_value, pos;
  int nblock;
  int last_npoint_I, last_npoint_V;

  bool isI, isV;
  bool dc, pwl;

  //NodeList::Node* node_tmp;

  FILE* fid_tmp=NULL;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));

  incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  tmp_incfilename = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr3 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr4 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr5 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr6 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr7 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  istr8 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));

  /* get the director of file */

  i = 0;
  pos = 0;
  while(filename[i]!='\0'){
    if(filename[i] == '/')
      pos = i;
    i++;
  }

  if(pos != 0){
    pos++;
    strncpy(incfilename,filename,pos);
    incfilename[pos] = '\0';
  }

  /* stamp circuit to G,C,B and waveform */   

  num_L_tmp = 0;
  num_I_tmp = -1;
  num_V_tmp = -1;
  last_npoint_I = 0;
  last_npoint_V = 0;


  while(!feof(fid) || (fid_tmp != NULL && !feof(fid_tmp))){

    if(feof(fid)==0){
      //if (fgets(strline, READ_BLOCK_SIZE, fid) == NULL) break;
      fgets(strline, READ_BLOCK_SIZE,fid);
      //printf("%s\n",strline);
    }
    else{
      if (fgets(strline, READ_BLOCK_SIZE, fid_tmp) == NULL) break;
      //printf("%s\n",strline);
    }

    switch(strline[0])
      {


	// ************ Voltage *******************
      case'V': case'v':
	num_V_tmp++;
	isV = true; isI = false;
	if(sscanf(strline, "%*s %s %s %s",  node1, node2, istr) == 3){
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  
	  index_i = nNodes + nL + num_V_tmp;
	  index_j = num_V_tmp;

	  B->pushEntry(index_i, index_j, -1);

	  if((istr[0] == 'P' && istr[1] == 'W') ||
	     (istr[0] == 'p' && istr[1] == 'w')){ //PWL
	      myGPUetbr->PWLvolExist += 1; // XXLiu
	    if(num_V_tmp != 0 && pwl == true && dc == false){
	      VS[num_V_tmp-1].time.set_size(num_point,true);
	      VS[num_V_tmp-1].value.set_size(num_point,true);
	    }
	    pwl = true;
	    dc = false;
	    if(num_V_tmp == nVS-1){
	      last_npoint_V = 0;
	    }
	    num_point = 0;
	    if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
	      break;
	    else{ 
	      i = 0;
	      while(strline[i] != '(')
		i++;
	      i++;
	      j = 0;
	      k = 0;

	      mark_value = 0;
	      while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		if(strline[i] != ' ' && mark_value == 0){ //input timestr
		  timestr[j] = strline[i];
		  i++;
		  j++;
		}
		else{
		  mark_value = 1;
		  i++;
		  if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		    valuestr[k] = strline[i];
		    k++;
		  } 
		  else{
		    valuestr[k] = '\0';
		    timestr[j] = '\0';
		    k = 0;
		    j = 0;
		    value = StrToNum(valuestr);
		    ptime = StrToNum(timestr);
		    if (num_point==0){
		      VS[num_V_tmp].time.set_size(PWL_SIZE, false);
		      VS[num_V_tmp].value.set_size(PWL_SIZE, false);
		      if(ptime!=0){
			VS[num_V_tmp].time(num_point) = 0;
			VS[num_V_tmp].value(num_point) = value;
			if(num_V_tmp == nVS-1){
			  last_npoint_V++;
			}
			num_point++;	    
		      }
		      VS[num_V_tmp].time(num_point) = ptime;
		      VS[num_V_tmp].value(num_point) = value;
		      num_point++;
		      if(num_V_tmp == nVS-1){
			last_npoint_V++;
		      }
		    }
		    else{
		      if(num_point%PWL_SIZE==0){
			nblock = num_point/PWL_SIZE + 1;
			VS[num_V_tmp].time.set_size(nblock*PWL_SIZE,true);
			VS[num_V_tmp].value.set_size(nblock*PWL_SIZE,true);
		      }
		      VS[num_V_tmp].time(num_point) = ptime;
		      VS[num_V_tmp].value(num_point) = value;
		      if(num_V_tmp == nVS-1){
			last_npoint_V++;
		      }
		      num_point++;
		    }
		    mark_value = 0;
		    i++;
		  }
		}
	      }
	    }
	  }
	  else if (sscanf(strline, "%*s %s %s %s %s",  node1, node2, istr, istr2) == 4){
	    if((istr2[0] == 'P' && istr2[1] == 'U') ||
		  (istr2[0] == 'p' && istr2[1] == 'u')){ //PULSE
	      myGPUetbr->PULSEvolExist += 1; // XXLiu
	    if(sscanf(strline, "%*s %s %s %s %s %s %s %s %s %s %s", 
		      node1, node2, istr, istr2, istr3, istr4, istr5, istr6, istr7, istr8) == 10){
	      double v1, v2, td, tr, tf, pw, period;
	      int nperiod = 1;
	      if (istr2[strlen(istr2)-1] == ','){
		istr2[strlen(istr2)-1] = 0;
	      }
	      v1 = StrToNum(istr2 + 6);
	      if (istr3[strlen(istr3)-1] == ','){
		istr3[strlen(istr3)-1] = 0;
	      }
	      v2 = StrToNum(istr3);
	      if (istr4[strlen(istr4)-1] == ','){
		istr4[strlen(istr4)-1] = 0;
	      }
	      td = StrToNum(istr4);
	      if (istr5[strlen(istr5)-1] == ','){
		istr5[strlen(istr5)-1] = 0;
	      }
	      tr = StrToNum(istr5);
	      if (istr6[strlen(istr6)-1] == ','){
		istr6[strlen(istr6)-1] = 0;
	      }
	      tf = StrToNum(istr6);
	      if (istr7[strlen(istr7)-1] == ','){
		istr7[strlen(istr7)-1] = 0;
	      }
	      pw = StrToNum(istr7);
	      if (istr8[strlen(istr8)-1] == ')'){
		istr8[strlen(istr8)-1] = 0;
	      }
	      period = StrToNum(istr8);
	      nperiod = floor_i(tstop/period) + 1;
	      VS[num_V_tmp].time.set_size(6*nperiod, false);
	      VS[num_V_tmp].value.set_size(6*nperiod, false);
	      for (i = 0; i < nperiod; ++i){
		VS[num_V_tmp].time(i*6) = i*period;
		VS[num_V_tmp].value(i*6) = v1;
		VS[num_V_tmp].time(i*6 + 1) = i*period + td;
		VS[num_V_tmp].value(i*6 + 1) = v1;
		VS[num_V_tmp].time(i*6 + 2) = i*period + td + tr;
		VS[num_V_tmp].value(i*6 + 2) = v2;
		VS[num_V_tmp].time(i*6 + 3) = i*period  + td + tr + pw;
		VS[num_V_tmp].value(i*6 + 3) = v2;
		VS[num_V_tmp].time(i*6 + 4) = i*period + td + tr + pw + tf;
		VS[num_V_tmp].value(i*6 + 4) = v1;
		VS[num_V_tmp].time(i*6 + 5) = (i+1)*period;
		VS[num_V_tmp].value(i*6 + 5) = v1;
	      }
	      if (nperiod*period > tstop){
		VS[num_V_tmp].time(6*nperiod - 1) = tstop;
	      }
	    }
	    }
	  }
	  else{ //DC
	  dc = true;
	  pwl = false;

	  value = StrToNum(istr);
	  
	  VS[num_V_tmp].time.set_size(2, false);
	  VS[num_V_tmp].value.set_size(2, false);
	  
	  VS[num_V_tmp].time(0) = 0;
	  VS[num_V_tmp].time(1) = tstop;
	  VS[num_V_tmp].value(0) = value;
	  VS[num_V_tmp].value(1) = value;

	}
	}
	break;

	// ******************** Current source ******************
      case'I': case'i':
	num_I_tmp++;
	isV = false; isI = true;
	if(sscanf(strline, "%s %s %s %s", isrcName, node1, node2, istr) == 4){ // XXLiu
	  n1_tmp = nodePool->findorPushNode(node1);
	  n2_tmp = nodePool->findorPushNode(node2);
	  n1 = nodePool->getNode(n1_tmp)->row_no;
	  n2 = nodePool->getNode(n2_tmp)->row_no;
	  
	  index_j = nVS + num_I_tmp; 

	  if(n1 != GNDNODE) {
	    B->pushEntry(n1,index_j,-1);
	  }
	  if(n2 != GNDNODE) {
	    B->pushEntry(n2,index_j,1);
	  }

	  if((istr[0] == 'P' && istr[1] == 'W') ||
	     (istr[0] == 'p' && istr[1] == 'w')){ //PWL
	      myGPUetbr->PWLcurExist += 1; // XXLiu
	    if(num_I_tmp != 0 && dc == false && pwl == true){
	      IS[num_I_tmp-1].time.set_size(num_point,true);
	      IS[num_I_tmp-1].value.set_size(num_point,true);
	    }
	    dc = false;
	    pwl = true;
	    if(num_I_tmp == nIS - 1)
	      last_npoint_I = 0;
	    num_point = 0;
	    if(strcmp(istr,"PWL(") == 0 || strcmp(istr,"pwl(") == 0 || strcmp(istr,"PWL") == 0 || strcmp(istr,"pwl") == 0)
	      break;
	    else{ 
	      i = 0;
	      while(strline[i] != '(')
		i++;
	      i++;
	      j = 0;
	      k = 0;

	      mark_value = 0;
	      while((strline[i] != '\0' && strline[i]!=')')||mark_value == 1){
		if(strline[i] != ' ' && mark_value == 0){ //input timestr
		  timestr[j] = strline[i];
		  i++;
		  j++;
		}
		else{
		  mark_value = 1;
		  i++;
		  if(strline[i] != ' ' && strline[i] != '\0' && strline[i]!=')' && mark_value == 1){ //input valuestr
		    valuestr[k] = strline[i];
		    k++;
		  } 
		  else{
		    valuestr[k] = '\0';
		    timestr[j] = '\0';
		    k = 0;
		    j = 0;
		    value = StrToNum(valuestr);
		    ptime = StrToNum(timestr);
		    if (num_point==0){
		      IS[num_I_tmp].time.set_size(PWL_SIZE, false);
		      IS[num_I_tmp].value.set_size(PWL_SIZE, false);
		      if(ptime!=0){
			IS[num_I_tmp].time(num_point) = 0;
			IS[num_I_tmp].value(num_point) = value;
			if(num_I_tmp == nIS - 1)
			  last_npoint_I++;
			num_point++;	    
		      }
		      IS[num_I_tmp].time(num_point) = ptime;
		      IS[num_I_tmp].value(num_point) = value;
		      if(num_I_tmp == nIS - 1)
			last_npoint_I++;
		      num_point++;
		    }
		    else{
		      if(num_point%PWL_SIZE==0){
			nblock = num_point/PWL_SIZE + 1;
			IS[num_I_tmp].time.set_size(nblock*PWL_SIZE,true);
			IS[num_I_tmp].value.set_size(nblock*PWL_SIZE,true);
		      }
		      IS[num_I_tmp].time(num_point) = ptime;
		      IS[num_I_tmp].value(num_point) = value;
		      if(num_I_tmp == nIS - 1)
			last_npoint_I++;
		      num_point++;
		    }
		    mark_value = 0;
		    i++;
		  }
		}
	      }
	     
	    }
	    //while(!getchar()) ; // XXLiu
	  }
	  else if (sscanf(strline, "%*s %s %s %s %s",  node1, node2, istr, istr2) == 4){
	    if((istr2[0] == 'P' && istr2[1] == 'U') ||
		  (istr2[0] == 'p' && istr2[1] == 'u')){ //PULSE
	      /************************************************/
	      if(myGPUetbr->PULSEcurExist==0) { // XXLiu
		myGPUetbr->PULSEtime_host=(double*)malloc(5*sizeof(double));
		myGPUetbr->PULSEval_host=(double*)malloc(2*sizeof(double));
	      }
	      else { // XXLiu
		myGPUetbr->PULSEtime_host=(double*)
		  realloc(myGPUetbr->PULSEtime_host,
			  5*(myGPUetbr->PULSEcurExist+1)*sizeof(double));
		myGPUetbr->PULSEval_host=(double*)
		  realloc(myGPUetbr->PULSEval_host,
			  2*(myGPUetbr->PULSEcurExist+1)*sizeof(double));
	      }
	      myGPUetbr->PULSEcurExist += 1; // XXLiu
	      /************************************************/
	    if(sscanf(strline, "%*s %s %s %s %s %s %s %s %s %s %s", 
		      node1, node2, istr, istr2, istr3, istr4, istr5, istr6, istr7, istr8) == 10){
	      double v1, v2, td, tr, tf, pw, period;
	      int nperiod = 1;
	      if (istr2[strlen(istr2)-1] == ','){
		istr2[strlen(istr2)-1] = 0;
	      }
	      v1 = StrToNum(istr2 + 6);
	      if (istr3[strlen(istr3)-1] == ','){
		istr3[strlen(istr3)-1] = 0;
	      }
	      v2 = StrToNum(istr3);
	      if (istr4[strlen(istr4)-1] == ','){
		istr4[strlen(istr4)-1] = 0;
	      }
	      td = StrToNum(istr4);
	      if (istr5[strlen(istr5)-1] == ','){
		istr5[strlen(istr5)-1] = 0;
	      }
	      tr = StrToNum(istr5);
	      if (istr6[strlen(istr6)-1] == ','){
		istr6[strlen(istr6)-1] = 0;
	      }
	      tf = StrToNum(istr6);
	      if (istr7[strlen(istr7)-1] == ','){
		istr7[strlen(istr7)-1] = 0;
	      }
	      pw = StrToNum(istr7);
	      if (istr8[strlen(istr8)-1] == ')'){
		istr8[strlen(istr8)-1] = 0;
	      }
	      period = StrToNum(istr8);

	      /************************************************/
	      myGPUetbr->PULSEval_host[(myGPUetbr->PULSEcurExist-1)*2+0] = v1;
	      myGPUetbr->PULSEval_host[(myGPUetbr->PULSEcurExist-1)*2+1] = v2;

	      myGPUetbr->PULSEtime_host[(myGPUetbr->PULSEcurExist-1)*5+0] = td;
	      myGPUetbr->PULSEtime_host[(myGPUetbr->PULSEcurExist-1)*5+1] = tr;
	      myGPUetbr->PULSEtime_host[(myGPUetbr->PULSEcurExist-1)*5+2] = tf;
	      myGPUetbr->PULSEtime_host[(myGPUetbr->PULSEcurExist-1)*5+3] = pw;
	      myGPUetbr->PULSEtime_host[(myGPUetbr->PULSEcurExist-1)*5+4] = period;
	      /************************************************/

	      nperiod = floor_i(tstop/period) + 1;
	      IS[num_I_tmp].time.set_size(6*nperiod, false);
	      IS[num_I_tmp].value.set_size(6*nperiod, false);
	      for (i = 0; i < nperiod; ++i){
		IS[num_I_tmp].time(i*6) = i*period;
		IS[num_I_tmp].value(i*6) = v1;
		IS[num_I_tmp].time(i*6 + 1) = i*period + td;
		IS[num_I_tmp].value(i*6 + 1) = v1;
		IS[num_I_tmp].time(i*6 + 2) = i*period + td + tr;
		IS[num_I_tmp].value(i*6 + 2) = v2;
		IS[num_I_tmp].time(i*6 + 3) = i*period  + td + tr + pw;
		IS[num_I_tmp].value(i*6 + 3) = v2;
		IS[num_I_tmp].time(i*6 + 4) = i*period + td + tr + pw + tf;
		IS[num_I_tmp].value(i*6 + 4) = v1;
		IS[num_I_tmp].time(i*6 + 5) = (i+1)*period;
		IS[num_I_tmp].value(i*6 + 5) = v1;
	      }
	      if (nperiod*period > tstop){
		IS[num_I_tmp].time(6*nperiod - 1) = tstop;
	      }
	    }
	    }
	  }
	  else{ //DC
	    printf("   not PWL nor pulse: %s\n",isrcName); // XXLiu
	    while( !getchar() ) ; // XXLiu
	    dc = true;
	    pwl = false;
	    value = StrToNum(istr);
	    IS[num_I_tmp].time.set_size(2, false);
	    IS[num_I_tmp].value.set_size(2, false);
	    
	    IS[num_I_tmp].time(0) = 0;
	    IS[num_I_tmp].time(1) = tstop;
	    IS[num_I_tmp].value(0) = value;
	    IS[num_I_tmp].value(1) = value;
	    
	  }
	}

	break;

	//  ************ PWL input ***************
      case'+':
	if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){
	  value = StrToNum(valuestr);
	  ptime = StrToNum(timestr);
	  if(isV == true && isI == false){ //PWL for V
	    if (num_point==0){
	      VS[num_V_tmp].time.set_size(PWL_SIZE, false);
	      VS[num_V_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		VS[num_V_tmp].time(num_point) = 0;
		VS[num_V_tmp].value(num_point) = value;
		if(num_V_tmp == nVS -1)
		  last_npoint_V++;
		num_point++;	    
	      }
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      if(num_V_tmp == nVS-1)
		last_npoint_V++;
	      num_point++;
	    }
	    else{
	      if(num_point%PWL_SIZE==0){
		nblock = num_point/PWL_SIZE + 1;
		VS[num_V_tmp].time.set_size(nblock*PWL_SIZE,true);
		VS[num_V_tmp].value.set_size(nblock*PWL_SIZE,true);
	      }
	      VS[num_V_tmp].time(num_point) = ptime;
	      VS[num_V_tmp].value(num_point) = value;
	      if(num_V_tmp == nVS-1)
		last_npoint_V++;
	      num_point++;
	    }
	  }
	  else if(isI == true && isV == false){ //PWL for I
	    if (num_point==0){
	      IS[num_I_tmp].time.set_size(PWL_SIZE, false);
	      IS[num_I_tmp].value.set_size(PWL_SIZE, false);
	      if(ptime!=0){
		IS[num_I_tmp].time(num_point) = 0;
		IS[num_I_tmp].value(num_point) = value;
		if(num_I_tmp == nIS - 1)
		  last_npoint_I++;
		num_point++;	    
	      }
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      if(num_I_tmp == nIS - 1)
		last_npoint_I++;
	      num_point++;
	    }
	    else{
	      if(num_point%PWL_SIZE==0){
		nblock = num_point/PWL_SIZE + 1;
		IS[num_I_tmp].time.set_size(nblock*PWL_SIZE,true);
		IS[num_I_tmp].value.set_size(nblock*PWL_SIZE,true);
	      }
	      IS[num_I_tmp].time(num_point) = ptime;
	      IS[num_I_tmp].value(num_point) = value;
	      if(num_I_tmp == nIS - 1)
		last_npoint_I++;
	      num_point++;
	    }
	  }
	  else{
	    printf("PWL input error.\n"); exit(1);
	  }
	}
	else {
	  if(isI == true){ //PWL for I
	      IS[num_I_tmp].time.set_size(num_point,true);
	      IS[num_I_tmp].value.set_size(num_point,true);
	  }
	}

	break;

        //  ********** .include ********************
      case'.':
	if (strline[1]=='i' && strline[2]=='n'){ //.include
	  if(sscanf(strline, "%*s %s", tmp_incfilename) == 1){
	    i = 0;
	    j = pos;
	    while(tmp_incfilename[i] != '\0'){
	      if(tmp_incfilename[i]!='\"'){
		incfilename[j] = tmp_incfilename[i];
		j++;
	      }
	      i++;
	    }
	    incfilename[j] = '\0';

	    fid_tmp = fid;

	    fid = fopen(incfilename,"r");
	    if(fid == NULL){
	      printf("Open file Error!\n");
	      exit(-1);
	    }	    
	  }
	}
	else if(strncmp(strline, ".end", 4) == 0 || strncmp(strline, ".END", 4) == 0){
	  if(last_npoint_I != 0){
	    IS[num_I_tmp].time.set_size(last_npoint_I,true);
	    IS[num_I_tmp].value.set_size(last_npoint_I,true);
	  }
	  if(last_npoint_V != 0){
	    VS[num_V_tmp].time.set_size(last_npoint_V,true);
	    VS[num_V_tmp].value.set_size(last_npoint_V,true);
	  }
	}
	  
	break;


      default:
	break;
      }
  }
  fclose(fid);

  if(fid_tmp != NULL){
    fclose(fid_tmp);
  }

  B->sort();

  free(strline);   
  free(tmp_incfilename);
  free(incfilename);
  free(node1);
  free(node2);
  free(istr);
  free(istr2);
  free(istr3);
  free(istr4);
  free(istr5);
  free(istr6);
  free(istr7);
  free(istr8);
  free(timestr);
  free(stimestr);
  free(valuestr);

}

void parser_old(const char* filename, circuit* cir, wave* waveform)
{
  char* strline;
  char *branch, *node1, *node2;
  char *timestr, *stimestr, *valuestr;
  char *tstepstr, *tstopstr;
  char *portstr;

  double tstep, tstop;
  double ptime, value;

  int i,j;

  FILE* fid = fopen(filename, "r");
  if(fid == NULL){
    printf("Open file Error!\n");
    exit(-1);
  }

  strline = (char*)malloc(READ_BLOCK_SIZE*sizeof(char));
  
  branch = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node1 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  node2 = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));
  portstr = (char*)malloc(NAME_BLOCK_SIZE*sizeof(char));

  timestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  stimestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  valuestr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  tstepstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
  tstopstr = (char*)malloc(VALUE_BLOCK_SIZE*sizeof(char));
 

  if(strline == NULL){
    printf("Allocate string memory failed.\n");
    return;
  }

  while(!feof(fid)){

    if(fgets(strline, READ_BLOCK_SIZE, fid)==NULL)
      break;

    switch(strline[0])
      {
	//  ************ Resistor ****************
      case'R': case'r':

	if(sscanf(strline, "%s %s %s %s", branch, node1, node2, valuestr) == 4){
	  value = StrToNum(valuestr);
	  switch(cir->addRes(branch, node1, node2, value))
	    {
	    case 1:
	      printf("Error: R%s connects to the same node %s in file %s\n", branch, node1, filename);
	      break;
	    case 3:
	      printf("Error: R%s has zero value in file %s\n", branch, filename);
	      break;
	    }
	}
	else
	  printf("Fail in obtaining resistance value \n");

	break;

	//  ************ Capacitor ***************
      case'C': case'c':
	if(sscanf(strline, "%s %s %s %s", branch, node1, node2, valuestr) == 4){
	  value = StrToNum(valuestr);
	  switch(cir->addCap(branch, node1, node2, value))
	    {
	    case 1:
	      printf("Error: C%s connects to the same node %s in file %s\n", branch, node1, filename);
	      break;
	    case 3:
	      printf("Error: C%s has zero value in file %s\n", branch, filename);
	      break;
	    }
	}
	else
	  printf("Fail in obtaining capacitor value \n");

	break;
	
	//  ************ Inductor ****************
      case'L': case'l':
	if(sscanf(strline, "%s %s %s %s", branch, node1, node2, valuestr)==4)
	{
	  value = StrToNum(valuestr);
	  switch(cir->addselfInc(branch, node1, node2, value))
	    {
	    case 1:
	      printf("Error: L%s connects to the same node %s in file %s\n", branch, node1, filename);
	      break;
	    case 3: 
	      printf("Error: L%s has zero value in file %s\n", branch, filename);
	      break;
	    }
	}	
	break;
	//  ************ Voltage  ****************
      case'V': case'v':
	if(sscanf(strline, "%s %s %s %s %s", branch, node1, node2, timestr, valuestr)==5){ //PWL
	  if(sscanf(timestr, "%*4c %s",stimestr)==1) 
	    ptime = StrToNum(stimestr);
	  else
	    break;
	  value = StrToNum(valuestr);
	  switch(cir->addVsrc(branch, node1, node2, waveform->newPWL(ptime, value)))
	    {
	    case 1:
	      printf("Error: C%s connects to the same node %s in file %s\n",branch, node1, filename);
	      break;
	    }
	  /*while(1){
	    if(fgets(strline, READ_BLOCK_SIZE, fid)==NULL) break;
	    if(strline[0] != '+') break;
	    if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){
	      value = StrToNum(valuestr);
	      ptime = StrToNum(timestr);
	      if(waveform -> pushPWL(ptime, value)==false)
		{ printf("Input PWL error"); exit(1); }
	    }
	    }*/	  
	}
	else{ //DC
	if(sscanf(strline, "%s %s %s %s", branch, node1, node2, valuestr)==4){
	  value = StrToNum(valuestr);
	  switch(cir->addVsrc(branch, node1, node2, waveform->newDC(value)))
	    {
	    case 1:
	      printf("Error: C%s connects to the same node %s in file %s\n",branch, node1, filename);
	      break;
	    }
	}
	}
	break;
	//  ************ Current  ****************
      case'I': case'i': 
	if(sscanf(strline, "%s %s %s %s %s", branch, node1, node2, timestr, valuestr)==5){ //PWL
	  if(sscanf(timestr, "%*4c %s",stimestr)==1) ptime = StrToNum(stimestr);
	  value = StrToNum(valuestr);
	  switch(cir->addIsrc(branch, node1, node2, waveform->newPWL(ptime, value)))
	    {
	    case 1:
	      printf("Error: C%s connects to the same node %s in file %s\n",branch, node1, filename);
	      break;
	      	    }
	  /*while(1){
	    if(fgets(strline, READ_BLOCK_SIZE, fid)==NULL) break;
	    if(strline[0] != '+') break;
	    if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){
	      value = StrToNum(valuestr);
	      ptime = StrToNum(timestr);
	      if(waveform -> pushPWL(ptime, value)==false)
		{ printf("Input PWL error"); exit(1); }
	    }
	    }*/
	}
	else{ //DC
	  sscanf(strline, "%s %s %s %s", branch, node1, node2, valuestr);
	  value = StrToNum(valuestr);
	  switch(cir->addIsrc(branch, node1, node2, waveform->newDC(value)))
	    {
	    case 1:
	      printf("Error: C%s connects to the same node %s in file %s\n", branch, node1, filename);
	      break;
	    }
	} 
      
	break;
	//  ************ PWL input ***************
      case'+':
	if(sscanf(strline, "%*2c %s %s", timestr, valuestr)==2){
	  value = StrToNum(valuestr);
	  ptime = StrToNum(timestr);
	  if(waveform -> pushPWL(ptime, value)==false)
	    { printf("Input PWL error"); exit(1); }
	}
	break;
	//  ************ Simulation **************
      case'.':
	if(strline[1]=='t'){
	  if(sscanf(strline, "%*s %s %s", tstepstr, tstopstr)==2){
	    tstep = StrToNum(tstepstr);
	    tstop = StrToNum(tstopstr);
	    waveform->set_tstep(tstep);
	    waveform->set_tstop(tstop);
	  }
	}
	else if (strline[1]=='p' && strline[2]=='r'){
	  i = 0;
	  while(strline[i]!='\0'){
	    if(strline[i]=='('){
	      i++;
	      j = 0;
	      while(strline[i]!=')'){
		portstr[j++] = strline[i++];
	      }
	      portstr[j] = '\0';
	      cir->addPort( portstr );
	    }
	    i++;
	  }
	}
	  
	break;
	//  ************ Commons & blankline *****
      case'*': case'$': case ';': case '\n': case '\0': 
	break;
	
      default:
	printf("Warning: Unknown expression in line %s\n", strline);
	break;
      }
  }    
  fclose(fid);

  free(strline);   
  free(branch);
  free(node1);
  free(node2);
  free(portstr);
  free(timestr);
  free(stimestr);
  free(valuestr);
  free(tstepstr);
  free(tstopstr);
  
}


void psource(wave* waveform, circuit* ckt, Source *VS, int nVS, Source *IS, int nIS)
{
  int index,addr;
  int size;
  int i,j;

  SrcBranch::Branch* tmp_b;

  double* tmp_value;
  wave::Element* tmp_element;

  //********************* transform voltage source ***********************
  for(i=0;i<nVS;i++)
    {
      tmp_b = ckt->V->getBranch(i);
      index = tmp_b->wave;
      tmp_element = waveform->geteData(index);
      addr = tmp_element->addr;
      if(tmp_element->type==DC){
	VS[i].time.set_size(2,false);
	VS[i].time(0) = 0;
	VS[i].time(1) = waveform->get_tstop();
	tmp_value = waveform->getvData(addr);
	VS[i].value.set_size(2,false);
	VS[i].value(0) = *tmp_value;
	VS[i].value(1) = *tmp_value;
      }
      else if(tmp_element->type==PWL){
	if ( index == nVS + nIS -1 ){
	  size = waveform->getvNum()-addr;
	  size = size/2;
	}
	else{
	  size = waveform->geteData(index+1)->addr - addr;
	  size = size/2;
	}
	VS[i].time.set_size(size,false);
	VS[i].value.set_size(size,false);
	for ( j=0;j<size;j++ )
	  {
	    tmp_value = waveform->getvData(addr+j*2);
	    VS[i].time(j) = *tmp_value;
	    tmp_value = waveform->getvData(addr+j*2+1);
	    VS[i].value(j) = *tmp_value;
	  }
      }
      
    }

  //*********************** transform current source **********************
  for(i=0;i<nIS;i++)
    {
      tmp_b = ckt->I->getBranch(i);
      index = tmp_b->wave;
      tmp_element = waveform->geteData(index);
      addr = tmp_element->addr;
      if(tmp_element->type==DC){
	size = 2;
	IS[i].time.set_size(size,false);
	IS[i].value.set_size(size,false);
	IS[i].time(0) = 0;
	IS[i].time(1) = waveform->get_tstop();
	tmp_value = waveform->getvData(addr);
	IS[i].value(0) = *tmp_value;
	IS[i].value(1) = *tmp_value;
      }
      else if(tmp_element->type==PWL){
	if ( index == nVS + nIS -1 ){
	  size = waveform->getvNum()-addr;
	  size = size/2;
	}
	else{
	  size = waveform->geteData(index+1)->addr - addr;
	  size = size/2;
	}
	IS[i].time.set_size(size,false);
	IS[i].value.set_size(size,false);
	for ( j=0;j<size;j++ )
	  {
	    tmp_value = waveform->getvData(addr+j*2);
	    IS[i].time(j) = *tmp_value;
	    tmp_value = waveform->getvData(addr+j*2+1);
	    IS[i].value(j) = *tmp_value;
	  }
      }
      
    }  
 
}

void printsource(Source* VS, int nVS, Source* IS, int nIS)
{
  int i,j;
  for( i=0;i<nVS;i++){
    if(VS[i].time.size()!=VS[i].value.size()) { printf("Error!\n");exit(1); }
    for(j=0;j<VS[i].time.size();j++)
      printf("time: %e\t value: %f\n",VS[i].time(j), VS[i].value(j));
  }
  for( i=0;i<nIS;i++){
    if(IS[i].time.size()!=IS[i].value.size()) { printf("Error!\n");exit(1); }
    for(j=0;j<IS[i].time.size();j++)
      printf("time: %e\t value: %f\n",IS[i].time(j), IS[i].value(j));
  }
}

void printmatrix(sparse_mat* smatrix)
{
  int i,j,nnz;
  int colsize;
  sparse_vec temp_vec;

  nnz = smatrix->nnz();
  colsize = smatrix->cols();

  for( j=0; j<smatrix->cols(); j++ ){
    temp_vec = smatrix->get_col(j);
    for( i=0; i<temp_vec.nnz(); i++ ){
      printf("(%d %d) %f \t", temp_vec.get_nz_index(i),j,temp_vec.get_nz_data(i));
    }
    printf("\n");
  }
}
