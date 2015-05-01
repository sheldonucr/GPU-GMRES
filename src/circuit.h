/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: circuit.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:54 $
 *    Authors: Ning Mi 
 * 
 *    Functions: header file for circuit and waveform class
 *
 */


#ifndef __CIRCUIT_H
#define __CIRCUIT_H

#include"namepool.h"
#include "element.h"

class NodeList;
class LinBranch;
class SrcBranch;

#define DC  1
#define PWL 2
/*******************************
*  class Waveform
*******************************/
class wave{
 public:
  struct Element
  {
    int type;
    int addr;
  };

 private:
  vector<Element> eleData;
  vector<double> valueData;
  double tstep,tstop;

 public:
  wave();
  ~wave();
  int newDC(double value);
  int newPWL(double time, double value);
  bool pushPWL(double time, double value);
  //  bool getsrc();

  void set_tstep(double t_step)
  { tstep = t_step; }
  void set_tstop(double t_stop)
  { tstop = t_stop; }

  double get_tstep() { return tstep; }
  double get_tstop() { return tstop; }

  inline double* getvData(int i)
    { return &(valueData[i]); }
  inline Element* geteData(int i)
    { return &(eleData[i]); }

  int getvNum() { return valueData.size(); }
  void printWaveElement(int i);  //print ith wave element in eleData
  void printWave();
  void printvalueData();


};

/*******************************
* class circuit
*******************************/
class circuit{
public:
	NodeList *nodePool;
	LinBranch *R, *C, *L;
	SrcBranch *V, *I;
	namepool nameData;

public:
	circuit();
	~circuit();
	inline void deleteR(){delete R;}
	inline void deleteC(){delete C;}
	inline void deleteL(){delete L;}
	inline void deleteV(){delete V;}
	inline void deleteI(){delete I;}
	char addRes(const char* name, const char* node1, const char* node2, double Rvalue);
	// 0 success, 1 node1==node2, 3 value <= 0
	char addCap(const char* name, const char* node1, const char* node2, double Cvalue);
	// 0 success, 1 node1==node2, 3 value <= 0
	char addselfInc(const char* name, const char* node1, const char* node2, double Lvalue);

	char addIsrc(const char* name, const char* node1, const char* node2, int Iwave);
	// 0 seccess, 1 node1==node2
	char addVsrc(const char* name, const char* node1, const char* node2, int Vwave);
	// 0 seccess, 1 node1==node2
	void addPort(const char* name);  //add I/O port
	//void addIC(const char* name, int value_pt);  //add initial condition

	void printRes();
	void printCap();
};
#endif
