/*!	\file
  	\brief use texture memroy to cache the data
 */

/*
 * IBM Sparse Matrix-Vector Multiplication Toolkit for Graphics Processing Units
 * (c) Copyright IBM Corp. 2008, 2009.  All Rights Reserved.
 */ 

#ifndef __CACHE_H__
#define __CACHE_H__

texture<float,1> tex_y_float;

void bind_y(const float * y)
{   cudaBindTexture(NULL, tex_y_float, y);   }

void unbind_y(const float * y)
{   cudaUnbindTexture(tex_y_float); }

__device__ float fetch_y(const int& i, const float * y)
{
    #if CACHE
        return tex1Dfetch(tex_y_float, i);
    #else
        return y[i];
    #endif
}

#endif /* __CACHE_H__ */

