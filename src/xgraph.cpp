/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: xgraph.cpp,v $
 *    $Revision: 1.2 $
 *    $Date: 2011/12/06 02:25:45 $
 *    Authors: Duo Li
 *
 *    Functions: Write XGraph
 *
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <itpp/base/vec.h>
#include <itpp/base/mat.h>

using namespace std;
using namespace itpp;

void write_xgraph(char *outGraphName, vec &ts, mat &sim_port_value, 
				  ivec &port, vector<string> &port_name)
{
  ofstream outGraph;
  outGraph.open(outGraphName);
  if (!outGraph){
	cout << "couldn't open " << outGraphName << endl;
	exit(-1);
  }
	 
  outGraph << "TitleText: " << "Voltage Response" << endl;
  outGraph << "XUnitText: " << "Time Seconds" << endl;
  outGraph << "YUnitText: " << "V" << endl;
  outGraph << "LogX: " << "False" << endl;
  outGraph << "XLowLimit: " << "1.000000e+35" << endl;
  outGraph << "XHighLimit: " << "-1.000000e+35" << endl;
  outGraph << "YLowLimit: " << "1.000000e+35" << endl;
  outGraph << "YHighLimit: " << "-1.000000e+35" << endl;
  outGraph << "LineWidth: " << "1" << endl;
  outGraph << "BoundBox: " << "True" << endl;
  outGraph << endl;

  int nport = port.size();
  for (int i = 0; i < nport; i++){
	outGraph << "\"on  " << port_name[i] << "\"" << endl;
	vec pv = sim_port_value.get_row(i);
	for (int j = 0; j < ts.length(); j++){	  
	  outGraph << ts(j) << " " << pv(j) << endl;
	}
	outGraph << endl;
  }
  outGraph.close();	
  cout << "** " << outGraphName << " dumped" << endl;
}
