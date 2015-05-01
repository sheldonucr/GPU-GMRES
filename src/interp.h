/*
*******************************************************

    Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: interp.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:07:58 $
 *    Authors: Duo Li
 *
 *    Functions: Interpolation header
 *
 */

#ifndef INTERP_H
#define INTERP_H

#include <itpp/base/vec.h>
#include "etbr.h"

using namespace itpp;

void find_nextpos(const vec &x0, double x, int &a, int &b, int cur);

// void interp1(const vec &x0, const vec &y0, const double &x, double &y);

void interp1(const vec &x0, const vec &y0, double x, double &y, int &cur);

void interp1(const vec &x0, const vec &y0, const vec &x, vec &y);

void interp1(const vec &x0, const vec &y0, double x, double &y, int &cur, vec &slope);

void interp1(int len, double *x0, double *y0, double x, double &y, int &cur, double *slope);

void interp1(int len, Source &s, double x, double &y, int &cur, double *slope);

void interp_next_step(double x, Source *IS, vector<int> &var_i, int* cur, vec* slope, int nVS, vec &u_col);

void interp_next_step2(double x, Source *IS, vector<int> &var_i, int* cur, vec* slope, int nVS, vec &u_col);

void interp_next_step3(double x, Source *IS, int* var_ii, int nv, int* cur, vec* slope, int nVS, vec &u_col);

void form_vec(vec &v, double start, double step, double end);

#endif
