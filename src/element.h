/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: element.h,v $
 *    $Revision: 1.2 $
 *    $Date: 2011/12/06 02:25:41 $
 *    Authors: Ning Mi
 *
 *    Functions: header file for node class, branch
 *    (resistance, capacitors, 
 *    inductance) class and source(voltage, current)class
 *
 */


#ifndef __ELEMENT_H
#define __ELEMENT_H

#include "namepool.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include "hashtable.h"

const int GNDNODE = -1;
const int INVALID = -2;

/*struct strCmp{
  bool operator()( const char* s1, const char* s2) const {
    return strcmp(s1,s2) < 0;
  }
  };*/

/**********************************
* class NodeList
**********************************/
struct strcomp
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1,s2) == 0;
  }
};

class NodeList
{
public:
	struct Node
	{
	  //int name;
		int row_no;
	};
    struct strCmp {
      bool operator()( char* const s1, char* const s2 ) const {
        return strcmp(s1, s2) < 0;
      }
    };


private:
	vector<Node> nodeData;
	vector<Node> icNodeData;
	vector<int>  portData;
	vector<string> portName;
	vector<int> tcData;
	vector<string> tcName;
	// namepool nameData;
	map<string, int> nodeTable;
	// map<char*, int, strCmp> nodeTable;
	// map<const char*, int, strcomp> nodeTable;
	HashTable *nodeHash;
	int nodeSize;

public:
	NodeList(void);
	~NodeList(void);
	
	void map_clear();
	void get_name_table(map<int, string>& name_table);
	int findorPushNode_map(const char* name);
	int findorPushNode(const char* name);
	bool findNode(const char* name);
	int findNode2(const char* name);
	void pushPort(const char* name);
	void pushTCNode(const char* name);
	void pushICnode(const char* name, int value);

	void setRowNum(int i, int index); //set the row_no of ith 

	void incRowNum(int i, int inc_index); // increase by inc_index

	void findAndPrintName(int row_no);
	void printPortName(FILE* fid=NULL);
	void printNodeInf();

	inline Node* getNode(int i)        //access the ith entry
	{  return &(nodeData[i]);  } 
	
	inline Node* getPort(int i)        //access the ith entry
	{  return &(nodeData[portData[i]]);  } 

	inline string getPortName(int i)        //access the ith entry
	{  return portName[i];  } 

	inline Node* getTCNode(int i)        //access the ith entry
	{  return &(nodeData[tcData[i]]);  } 

	inline string getTCName(int i)        //access the ith entry
	{  return tcName[i];  } 

	inline int getTCNum() {return tcData.size();}

	//inline Node* getICnode(int i)        //access the ith entry
	//{  return (nodeData[icNodeData[i]->name]);  } 

	int numNode(void){  return nodeSize; }
	int numElement(void) { return nodeData.size(); }
	int numPort(void){  return portData.size(); }
	int numICnode(void){  return icNodeData.size(); }
	
};


/**********************************************
* class LinBranch
**********************************************/
class LinBranch
{
public:
	struct Branch
	{
		//int name;         //address of name in namepool
		int node1;
		int node2;
		double value; 
	};

private:
	vector<Branch> branchData;
	//namepool nameData;

public:
	LinBranch(void);
	~LinBranch(void);

	void pushBranch(int node1, int node2, double value);

	inline Branch* getBranch(int i)
	{ return &(branchData[i]); }

	int numBranch() 
	{ return branchData.size(); }  

	void printBranch();
};


/**********************************************
* class SrcBranch
**********************************************/
class SrcBranch
{
public:
	struct Branch
	{
		//int name;
		int node1;
		int node2;
		int wave;
	};

private:
	vector<Branch> branchData;

public:
	SrcBranch(void);
	~SrcBranch(void);
	void pushBranch(int node1, int node2, int wave);
	inline Branch* getBranch(int i)
	{ return &(branchData[i]); }

	int numBranch(){ return branchData.size(); }

	void printBranch();
};

#endif
