/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: interp.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:58 $
 *    Authors: Duo Li
 *
 *    Functions: Interpolation
 *
 */

#include <iostream>
#include <itpp/base/mat.h>
#include <itpp/base/vec.h>
#include <itpp/base/math/elem_math.h>
#include <assert.h>
#include "interp.h"
#include "etbr.h"

using namespace itpp;

long interp1_sum = 0;
long interp2_sum = 0;

void find_nextpos(const vec &x0, double x, int &a, int &b, int cur)
{
  if (cur == -1){
	if (x < x0(0) || x > x0(x0.size()-1)){
	  a = -1;
	  b = -1;
	  return;
	}else{
	  a = 0;
	  b = a+1;
	  return;
	}
  }else if (cur < x0.size()-1){
	if (x < x0(cur+1)){
	  a = cur;
	  b = a+1;
	  return;
	}else{
	  int i;
	  for (i = cur; i < x0.size()-1; i++){
		if (x >= x0(i) && x <= x0(i+1)){
		  a = i;
		  b = a+1;
		  return;
		}
	  }
	  if (i == x0.size()-1){
		a = -1;
		b = -1;
		return;
	  }
	}
  }else{
	a = -1;
	b = -1;
	return;	
  }
}

/*
void find_nextpos(const vec &x0, double x, int &cur)
{
int len = x0.size();  
if (cur == -1){
	if (x < x0(0) || x > x0(len-1)){
	  cur = -1;
	  return;
	}else{
	  cur = 0;
	  return;
	}
  }else if (cur < len-1){
	if (x < x0(cur+1)){
	  return;
	}else{
	  int i;
	  for (i = cur; i < len-1; i++){
		if (x >= x0(i) && x <= x0(i+1)){
		  cur = i;
		  return;
		}
	  }
	  if (i == len-1){
		cur = -1;
		return;
	  }
	}
  }else{
	cur = -1;
	return;	
  }
}
*/

void find_nextpos(const vec &x0, double x, int &cur)
{
  int len = x0.size();
  if (cur == len-1){
	return;
  }	
  if (x > x0(cur+1)){
	for (int i = cur+1; i < len-1; i++){
	  //interp1_sum++;
	  if (x <= x0(i+1)){
		cur = i;
		return;
	  }
	}
	if (x > x0(len-1))
	  cur = len-1;
  }
}

void interp1(const vec &x0, const vec &y0, double x, double &y, int &cur)
{
  int len = x0.size(); 
  // find_nextpos(x0, x, cur);
  if (len == 1){
	y = y0(0);
	return;
  }
  if (cur == len-1){
	y = y0(len-1);
	return;
  }
  if (x == x0(cur+1)){
    y = y0(cur+1);
    return;
  }else if (x > x0(cur+1)){
	for (int i = cur+1; i < len-1; i++){
	  // interp2_sum++;
	  // std::cout << interp2_sum << std::endl;
	  if (x <= x0(i+1)){
		cur = i;
		y = y0(cur) + (x - x0(cur)) * (y0(cur+1) - y0(cur)) / (x0(cur+1) - x0(cur));
		return;
	  }
	}
	if (x > x0(len-1)){
	  cur = len-1;
	  y = y0(len-1);
	  return;	
	}
  }else{
	// interp2_sum++;
	y = y0(cur) + (x - x0(cur)) * (y0(cur+1) - y0(cur)) / (x0(cur+1) - x0(cur));
  }
}

void interp1(const vec &x0, const vec &y0, double x, double &y, int &cur, vec &slope)
{
  int len = x0.size(); 
  // find_nextpos(x0, x, cur);
   
  if (len == 1){ // single value
	y = y0(0);
	return;
  }

  if (cur == len-1){ // last value
	y = y0(len-1);
	return;
  }

  if(x == x0(cur+1)) {
    y = y0(cur+1);
    return;
  }
  else if(x > x0(cur+1)){
    for (int i = cur+1; i < len-1; i++){
      if (x <= x0(i+1)){
	cur = i;
	y = y0(cur) + (x - x0(cur)) * slope(cur);
	return;
      }
    }
    if (x > x0(len-1)){
      cur = len-1;
      y = y0(len-1);
      return;
    }
  }
  else{
  	y = y0[cur] + (x - x0[cur]) * slope[cur];
  }
  //interp2_sum++;
  //std::cout << interp2_sum << std::endl;
  //y = y0[cur] + (x - x0[cur]) * slope[cur];
}

void interp1(int len, double *x0, double *y0, double x, double &y, int &cur, double *slope)
{  
  
  if (x == x0[cur+1]){
    y = y0[cur+1];
    return;
  }else if (x > x0[cur+1]){
	for (int i = cur+1; i < len-1; i++){
	  if (x <= x0[i+1]){
		cur = i;
		y = y0[cur] + (x - x0[cur]) * slope[cur];
		return;
	  }
	}
	if (x > x0[len-1]){
	  cur = len-1;
	  y = y0[len-1];
	  return;
	}
  }else{
  	y = y0[cur] + (x - x0[cur]) * slope[cur];
  }
}


void interp1(int len, Source &s, double x, double &y, int &cur, double *slope)
{  
  double *x0 = s.time._data();
  double *y0 = s.value._data();
  
  if (x == x0[cur+1]){
    y = y0[cur+1];
    return;
  }else if (x > x0[cur+1]){
	for (int i = cur+1; i < len-1; i++){
	  if (x <= x0[i+1]){
		cur = i;
		y = y0[cur] + (x - x0[cur]) * slope[cur];
		return;
	  }
	}
	if (x > x0[len-1]){
	  cur = len-1;
	  y = y0[len-1];
	  return;
	}
  }else{
  	y = y0[cur] + (x - x0[cur]) * slope[cur];
  }
}

/*
void interp1(const vec &x0, const vec &y0, const vec &x, vec &y)
{
  y.set_size(x.size());
  if (x0.size() == 1 && y0.size() == 1){
	y = y0(0);
	return;
  }
  int a = -1, b = -1, cur = -1;
  for (int i = 0; i < x.size(); i++) {
	find_nextpos(x0, x(i), a, b, cur);
	// assert(a != -1 || b != -1);
	if (a == -1 || b == -1){
	  y(i) = 0;
	}else{
	  cur = a;
	  y(i) = y0(a) + (x(i) - x0(a)) * (y0(b) - y0(a)) / (x0(b) - x0(a));		
	}
  }
}
*/

void interp1(const vec &x0, const vec &y0, const vec &x, vec &y)
{
  int len = x.size();
  int len0 = x0.size();
  y.set_size(len);
  if (x0.size() == 1 && y0.size() == 1){
	y = y0(0);
	return;
  }
  int cur = 0;
  for (int i = 0; i < len; i++){
	find_nextpos(x0, x(i), cur);
	if (cur == len0-1){
	  y(i) = y0(len0-1);
	}else{
	  //interp1_sum++;
	  if (x0(cur+1) == x0(cur)){
	    y(i) = y0(cur);
	  }else{
	    y(i) = y0(cur) + (x(i) - x0(cur)) * (y0(cur+1) - y0(cur)) / (x0(cur+1) - x0(cur));
	  }
	}
  }
}

void interp_next_step(double x, Source *IS, vector<int> &var_i, int* cur, vec* slope, int nVS, vec &u_col)
{
  for(vector<int>::iterator it = var_i.begin(); it != var_i.end(); ++it){
	int len = IS[*it].time.size();
	if (cur[nVS+(*it)] == len-1){
	   u_col[nVS+*it] = IS[*it].value[len-1];
	}else{
	  double *x0 = IS[*it].time._data();
	  double *y0 = IS[*it].value._data();
	  int *cur0 = &(cur[nVS+(*it)]);
	  if (x > x0[*cur0+1]){
		for (int i = *cur0+1; i < len-1; i++){
		  if (x <= x0[i+1]){
			*cur0 = i;
			u_col[nVS+*it] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+*it][*cur0];
			break;
		  }
		}
		if (x > x0[len-1]){
		  *cur0 = len-1;
		  u_col[nVS+*it] = y0[len-1];
		}
	  }else{
		u_col[nVS+*it] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+*it][*cur0];
	  }
	}	
  }
}

void interp_next_step2(double x, Source *IS, vector<int> &var_i, int* cur, vec* slope, int nVS, vec &u_col)
{
  for(int k = 0; k < var_i.size(); ++k){
	int j = var_i[k];
	int len = IS[j].time.size();
	if (cur[nVS+j] == len-1){
	   u_col[nVS+j] = IS[j].value[len-1];
	}else{
	  double *x0 = IS[j].time._data();
	  double *y0 = IS[j].value._data();
	  int *cur0 = &(cur[nVS+j]);
	  if (x > x0[*cur0+1]){
		for (int i = *cur0+1; i < len-1; i++){
		  if (x <= x0[i+1]){
			*cur0 = i;
			u_col[nVS+j] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+j][*cur0];
			break;
		  }
		}
		if (x > x0[len-1]){
		  *cur0 = len-1;
		  u_col[nVS+j] = y0[len-1];
		}
	  }else{
		u_col[nVS+j] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+j][*cur0];
	  }
	}	
  }
}

void interp_next_step3(double x, Source *IS, int* var_ii, int nv, int* cur, vec* slope, int nVS, vec &u_col)
{
  for(int k = 0; k < nv; ++k){
	int j = var_ii[k];
	int len = IS[j].time.size();
	if (cur[nVS+j] == len-1){
	   u_col[nVS+j] = IS[j].value[len-1];
	}else{
	  double *x0 = IS[j].time._data();
	  double *y0 = IS[j].value._data();
	  int *cur0 = &(cur[nVS+j]);
	  if (x > x0[*cur0+1]){
		for (int i = *cur0+1; i < len-1; i++){
		  if (x <= x0[i+1]){
			*cur0 = i;
			u_col[nVS+j] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+j][*cur0];
			break;
		  }
		}
		if (x > x0[len-1]){
		  *cur0 = len-1;
		  u_col[nVS+j] = y0[len-1];
		}
	  }else{
		u_col[nVS+j] = y0[*cur0] + (x - x0[*cur0]) * slope[nVS+j][*cur0];
	  }
	}	
  }
}

void form_vec(vec &v, double start, double step, double end){
  if (step != 0){
	v.set_size(floor_i((end-start)/step) + 1 + 1);
	for (int i = 0; i < v.size(); i++){
	  v(i) = start + i*step;
	}
	if (v.get(v.length()-1) > end)
	  v.set(v.length()-1, end);
  }else{
	v.set_size(1);
	v(0) = start;
  }
}
