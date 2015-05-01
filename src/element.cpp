/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: element.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:55 $
 *    Authors: Ning Mi
 *
 *    Functions: node class, branch(resistance, capacitors, 
 *    inductance) class and source(voltage, current)class
 *
 */


#include "element.h"
#include <iostream>

NodeList::NodeList(){
	nodeSize = 0;
	/*if((nodeHash = new HashTable( &nameData )) == NULL ){
	  printf("Out of memory.\n"); exit(1);
	  }*/
	
}

NodeList::~NodeList(){
  nodeData.clear();
  icNodeData.clear();
  portData.clear();
  nodeTable.clear();
  /*
  for (int i = 0; i < nodeData.size(); i++){
	free(nodeData[i]);
  }
  for (int i = 0; i < icNodeData.size(); i++){
	free(icNodeData[i]);
  }
  */
  // if( nodeHash!=NULL ) delete nodeHash;
}

void NodeList::map_clear()
{
  nodeTable.clear();
}

void NodeList::get_name_table(map<int,string>& name_table)
{
  for (map<string, int>::iterator iter = nodeTable.begin(); iter != nodeTable.end(); ++iter){
	if (nodeData[iter->second].row_no != GNDNODE)
	  name_table[nodeData[iter->second].row_no] = iter->first;
  }
}

bool NodeList::findNode(const char* name){
  int addr;
  string sname = name;
  map<string, int>::iterator it;

  it = nodeTable.find(sname);
  
  if(it != nodeTable.end()){ // name in the nodelist
    return true;
  }
  else
    return false;
}

int NodeList::findNode2(const char* name){
  int addr;
  string sname = name;
  map<string, int>::iterator it;

  it = nodeTable.find(sname);
  
  if(it != nodeTable.end()){ // name in the nodelist
    return it->second;
  }
  else
    return -1;
}

int NodeList::findorPushNode(const char* name){
	int addr;
	string sname = name;
	//char * sname = new char [strlen(name)];
	//strcpy(sname, name);
  	//map<char*, int, strCmp>::iterator it;
	//map<const char*, int, strcomp>::iterator it;
	map<string, int>::iterator it;
	
	it = nodeTable.find(sname);

	if (it != nodeTable.end()){
		addr = it->second;
		//cout << name << endl;
		//cout << it->first << endl;
		//delete [] sname;
		return addr;
	}
	else{
		addr = nodeData.size();
		// Node* tempnode=(Node*)malloc(sizeof(Node));
		Node tempnode;
		// tempnode->name = nameData.pushName(name);
		if (!strcmp(name,"0") || !strcmp(name,"gnd"))
			tempnode.row_no = GNDNODE;
		else 
			tempnode.row_no = nodeSize++;

		nodeData.push_back(tempnode);
		nodeTable[sname] = addr;
		//nodeTable.insert(pair<char*, int>(sname, addr));
		//for (map<char*, int, strCmp>::iterator iter = nodeTable.begin(); iter != nodeTable.end(); ++iter){
		//  cout << "nodeTable[" << iter->first << "] = " << iter->second << endl;
		//}
		// cout << "=============" << endl;
		// free(tempnode);
		//delete [] sname;
	}
	return addr;

}		 

void NodeList::setRowNum(int i, int index){
  nodeData[i].row_no = index;
}

void NodeList::incRowNum(int i, int inc_index){
  nodeData[i].row_no = nodeData[i].row_no + inc_index;
}

/*int NodeList::findorPushNode(const char* name){
	int addr;

	addr = nodeHash->find(name);
	if (addr != -1){
		return addr;
	}
	else{
		addr = nodeData.size();
		Node* tempnode=(Node*)malloc(sizeof(Node));
		tempnode->name = nameData.pushName(name);
		if (!strcmp(name,"0") || !strcmp(name,"gnd"))
			tempnode->row_no = GNDNODE;
		else 
			tempnode->row_no = nodeSize++;
		nodeData.push_back(*tempnode);
		nodeHash->insertAtCur(tempnode->name, addr);
		free(tempnode);
	}
	return addr;
	}*/	


void NodeList::pushPort(const char* name)
{
  int i = findNode2(name);
  if(nodeData[i].row_no >= 0){
    portData.push_back(i);
    string sname(name);
    portName.push_back(sname);
  }else if(nodeData[i].row_no == GNDNODE){
	printf("Print GND node. \n");
  }
  else
    printf("Print port node %s does not exist. \n", name);
}

void NodeList::pushTCNode(const char* name)
{
  int i = findNode2(name);
  if(nodeData[i].row_no >= 0){
    tcData.push_back(i);
    string sname(name);
    tcName.push_back(sname);
  }else if(nodeData[i].row_no == GNDNODE){
  }
  else
    printf("tap current node %s does not exist. \n", name);
}

void NodeList::pushICnode(const char* name, int value)
{
  // Node* tempNode = (Node*)malloc(sizeof(Node));
  Node tempNode;
	// tempNode->name = nameData.pushName(name);
	tempNode.row_no = value;
	icNodeData.push_back(tempNode);
	// free(tempNode);
}

void NodeList::findAndPrintName(int row_no)
{
	for( int i=0; i<nodeData.size(); i++ )
		if( nodeData[i].row_no == row_no )
		{ 
		  // nameData.printName(nodeData[i]->name); 
			return; 
		}
} 

void NodeList::printPortName(FILE* fid)
{
	int i;
	
	if(fid == NULL){
		for( i=0;i<portData.size();i++ ){
		  //			printf(" V(");
		  printf("%d \t",nodeData[portData[i]].row_no);
		  // nameData.printName(nodeData[portData[i]]->name);
			//	printf(")<V>   \t");
		}
	}
	else{
		for( i=0;i<portData.size();i++ ){
		  //		fprintf(fid," V(");
		  printf("%d \t",nodeData[portData[i]].row_no);
		  // nameData.printName(nodeData[portData[i]]->name,fid);
			//	fprintf(fid,")<V>   \t");
		}
	}
}

void NodeList::printNodeInf(){
  int i;
  for (i=0;i<nodeData.size();i++){
    Node* temp = getNode(i);
    // printf("%d %d\n", temp->name, temp->row_no);
  }
}

LinBranch::LinBranch(){
}

LinBranch::~LinBranch(){
  //for (int i = 0; i < branchData.size(); i++){
  //	free(branchData[i]);
  //}
  //branchData.clear();
}

void LinBranch::pushBranch(int node1, int node2, double value)
{
  // Branch* tempBranch = (Branch*)malloc(sizeof(Branch));
	Branch tempBranch;
  	tempBranch.node1 = node1;
	tempBranch.node2 = node2;
	tempBranch.value = value;
	// tempBranch->name = nameData.pushName(name);
	branchData.push_back(tempBranch);
	// free(tempBranch);
}

void LinBranch::printBranch()
{
  int i;

  double value;

  for(i=0;i<branchData.size();i++)
    {
      int n1 = branchData[i].node1;
      int n2 = branchData[i].node2;
      //int n3 = branchData[i]->name;
      //nameData.printName(n3);
      printf("%d %d %f\n", n1,n2,branchData[i].value);
      
    }
}

SrcBranch::SrcBranch(){
}

SrcBranch::~SrcBranch(){
  // for (int i = 0; i < branchData.size(); i++){
  //	free(branchData[i]);
  // }
  //branchData.clear();
}

void SrcBranch::pushBranch(int node1, int node2, int wave)
{
  //Branch* tempBranch = (Branch*)malloc(sizeof(Branch));
	Branch tempBranch;	
	tempBranch.node1 = node1;
	tempBranch.node2 = node2;
	tempBranch.wave = wave;
	//tempBranch->name = nameData.pushName(name);
	branchData.push_back(tempBranch);
	// free(tempBranch);
}


void SrcBranch::printBranch()
{
  int i;

  double value;

  for(i=0;i<branchData.size();i++)
    {
      int n1 = branchData[i].node1;
      int n2 = branchData[i].node2;
      //int n3 = branchData[i]->name;
      //nameData.printName(n3);
      printf("%d %d %f\n", n1,n2,branchData[i].wave);
    }
}
