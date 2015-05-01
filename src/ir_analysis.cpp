/********************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: ir_analysis.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:58 $
 *    Authors: Duo Li
 *
 *    Functions: IR analysis
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <itpp/base/mat.h>
#include <itpp/base/math/min_max.h>
#include <itpp/base/matfunc.h>
#include <itpp/base/sort.h>

using namespace itpp;
using namespace std;

void ir_analysis(int num, vector<int> &tc_node,
				 vector<string> &tc_name, mat &X, mat &sim_value, char *ir_name)
{
  vec max_value, min_value, avg_value, ir_value;
  double max_ir, min_ir, avg_ir;
  int max_ir_idx, min_ir_idx;
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
  int ntstep = sim_value.cols();
  vec row_value(ntstep);
  mat sim_value_t = sim_value.T();
  for (int i = 0; i < nNodes; i++){
	row_value = sim_value_t * X.get_row(tc_node[i]);
	min_value(i) = min(row_value);
	max_value(i) = max(row_value);
	avg_value(i) = sum(row_value)/ntstep;
  }
  sorted_max_value_idx = sort_index(max_value);
  // sorted_min_value_idx = sort_index(min_value);
  sorted_avg_value_idx = sort_index(avg_value);
  ir_value = max_value - min_value;
  // min_ir = min(ir_value);
  // min_ir_idx = min_index(ir_value);
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
  /*
  cout << "Min " << display_num << " Node Voltage: " << endl;
  for (int i = 0; i < display_num; i++){
	cout << tc_name[sorted_min_value_idx(i)] << " : " 
		 << min_value(sorted_min_value_idx(i)) << endl;
  }
  cout << "******" << endl;
  */
  cout << "Avg " << display_num << " Node Voltage: " << endl;
  for (int i = 0; i < display_num; i++){
	cout << tc_name[sorted_avg_value_idx(nNodes-1-i)] << " : " 
		 << avg_value(sorted_avg_value_idx(nNodes-1-i)) << endl;
  }
  cout << "****** IR Drop Info ******  " << endl;
  cout << "Max IR:     " << tc_name[max_ir_idx] << " : " << max_ir << endl;
  // cout << "Min IR:     " << tc_name[min_ir_idx] << " : " << min_ir << endl;
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
  cout << "** " << ir_name << " dumped" << endl;
}

