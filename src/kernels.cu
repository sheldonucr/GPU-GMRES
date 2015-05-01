#include "gpuData.h"
#include "kernels.h"

__global__ void
myMemcpyD2Sdev(float *vecS, double *vecD, int q)
{
  int tid=threadIdx.x;
  if(tid < q)
    vecS[tid] = vecD[tid];
}

void myMemcpyD2Sdev_wrapper(float *vecS, double *vecD, int q)
{
  myMemcpyD2Sdev<<<1,q>>>(vecS, vecD, q);
}
//------------------------------------------
__global__ void
myMemcpyS2Ddev(double *vecD, float *vecS, int q)
{
  int tid=threadIdx.x;
  if(tid < q)
    vecD[tid] = vecS[tid];
}

void myMemcpyS2Ddev_wrapper(double *vecD, float *vecS, int q)
{
  myMemcpyS2Ddev<<<1,q>>>(vecD, vecS, q);
}
//------------------------------------------
__global__ void
permute_kernel(double *x, int *ipiv, int q)
{
  __shared__ double val[MAX_THREADS];  //__shared__ int ind[MAX_THREADS];
  int tid=threadIdx.x, ind;

  if(tid<q) {
    val[tid] = x[tid];
    ind = ipiv[tid];
  }
  __syncthreads();
  double myval=val[ind];
  if(tid<q)
    x[tid] = myval;
}

void permute_kernel_wrapper(double *x, int *ipiv, int q)
{
  permute_kernel<<<1,q>>>(x, ipiv, q);
}

//------------------------------------------
__global__ void
permute_single_kernel(float *x, int *ipiv, int q)
{
  __shared__ float val[MAX_THREADS];
  int tid=threadIdx.x, ind;

  if(tid<q) {
    val[tid] = x[tid];
    ind = ipiv[tid];
  }
  __syncthreads();
  float myval=val[ind];
  if(tid<q)
    x[tid] = myval;
}

void permute_single_kernel_wrapper(float *x, int *ipiv, int q)
{
  permute_single_kernel<<<1,q>>>(x, ipiv, q);
}
//------------------------------------------------

__global__ void
gen_dcVt_kernel(double *ut, double *dcVt, int p, int ldUt)
{
  int VSidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  double value=dcVt[VSidx];

  if(idxt < p)
    ut[VSidx*ldUt + idxt] = value;
}

void
gen_dcVt_kernel_wrapper(double *ut, double *dcVt, int p, int ldUt,
			dim3 grid, dim3 block)
{
  gen_dcVt_kernel<<<grid, block>>>(ut, dcVt, p, ldUt);
}
//------------------------------------------------

__global__ void
gen_dcVt_single_kernel(float *ut, float *dcVt, int p, int ldUt)
{
  int VSidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  float value=dcVt[VSidx];

  if(idxt < p)
    ut[VSidx*ldUt + idxt] = value;
}

void gen_dcVt_single_kernel_wrapper(float *ut, float *dcVt, int p, int ldUt,
                                    dim3 grid, dim3 block)
{
  gen_dcVt_single_kernel<<<grid, block>>>(ut, dcVt, p, ldUt);
}

//------------------------------------------------

__global__ void
gen_dcVt_part_kernel(double *ut, double *dcVt, int p, int ldUt, int shift)
{
  int VSidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  double value=dcVt[VSidx];

  if(idxt+shift < p)
    ut[VSidx*ldUt + idxt] = value;
}

void
gen_dcVt_part_kernel_wrapper(double *ut, double *dcVt, int p, int ldUt, int shift,
                             dim3 grid, dim3 block)
{
  gen_dcVt_part_kernel<<<grid, block>>>
    (ut, dcVt, p, ldUt, shift); //??????
}
//------------------------------------------------

__global__ void
gen_dcVt_part_single_kernel(float *ut, float *dcVt, int p, int ldUt, int shift)
{
  int VSidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  float value=dcVt[VSidx];

  if(idxt+shift < p)
    ut[VSidx*ldUt + idxt] = value;
}

void
gen_dcVt_part_single_kernel_wrapper(float *ut, float *dcVt, int p, int ldUt, int shift,
                                    dim3 grid, dim3 block)
{
  gen_dcVt_part_single_kernel<<<grid, block>>>
    (ut, dcVt, p, ldUt, shift); // ??????
}

//---------------------------------------------------

__global__ void
gen_PWLut_kernel(double *ut, double *PWLtime, double *PWLval, int *PWLnumPts,
		 double tstep, int p, int ldUt)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid, totalPts=PWLnumPts[ISidx];
  __shared__ double t[MAX_PWL_PTS], v[MAX_PWL_PTS];
  if(tid < totalPts) {
    t[tid] = PWLtime[ISidx*MAX_PWL_PTS+tid];
    v[tid] = PWLval[ISidx*MAX_PWL_PTS+tid];
  }
  __syncthreads();
  double mytime=(idxt)*tstep, value=v[0]; // +1
  int i;
  for(i=0; i<totalPts; i++) {
    if(mytime < t[i]) {
      value = v[i] - (t[i]-mytime) * (v[i]-v[i-1])/(t[i]-t[i-1]);
      break;
    }
  }
  if(i==totalPts) value = v[totalPts-1];

  if(idxt < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PWLut_kernel_wrapper(double *ut, double *PWLtime, double *PWLval, int *PWLnumPts,
			 double tstep, int p, int ldUt,
			 dim3 grid, dim3 block)
{
  gen_PWLut_kernel<<<grid, block>>>(ut, PWLtime, PWLval, PWLnumPts,
				    tstep, p, ldUt);
}
//---------------------------------------------------
__global__ void
gen_PWLut_single_kernel(float *ut, float *PWLtime, float *PWLval, int *PWLnumPts,
		 double tstep, int p, int ldUt)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid, totalPts=PWLnumPts[ISidx];
  __shared__ float t[MAX_PWL_PTS], v[MAX_PWL_PTS];
  if(tid < totalPts) {
    t[tid] = PWLtime[ISidx*MAX_PWL_PTS+tid];
    v[tid] = PWLval[ISidx*MAX_PWL_PTS+tid];
  }
  __syncthreads();
  float mytime=(idxt)*tstep, value=v[0]; // +1
  int i;
  for(i=0; i<totalPts; i++) {
    if(mytime < t[i]) {
      value = v[i] - (t[i]-mytime) * (v[i]-v[i-1])/(t[i]-t[i-1]);
      break;
    }
  }
  if(i==totalPts) value = v[totalPts-1];

  if(idxt < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PWLut_single_kernel_wrapper(float *ut, float *PWLtime, float *PWLval, int *PWLnumPts,
                                double tstep, int p, int ldUt,
                                dim3 grid, dim3 block)
{
  gen_PWLut_single_kernel<<<grid, block>>>
    (ut, PWLtime, PWLval, PWLnumPts, tstep, p, ldUt);
}
//-----------------------------------------
__global__ void
gen_PULSEut_kernel(double *ut, double *PULSEtime, double *PULSEval,
		   double tstep, int p, int ldUt)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  __shared__ double v[2], t[5]; // vlo, vhi, td, tr, tf, tw, tp
  if(tid < 5) t[tid] = PULSEtime[ISidx*5+tid];
  if(tid < 2) v[tid] = PULSEval[ISidx*2+tid];
  __syncthreads();
  double td=t[0], tr=t[1], tf=t[2], tw=t[3], tp=t[4], vlo=v[0], vhi=v[1];
  double mytime=(idxt)*tstep, value=vlo; // +1
  mytime = mytime - floor(mytime/tp)*tp;

  if(mytime < td) value = vlo;
  else if(mytime < td+tr) value = vlo + (mytime-td)*(vhi-vlo)/tr;
  else if(mytime < td+tr+tw) value = vhi;
  else if(mytime < td+tr+tw+tf) value = vhi - (mytime-td-tr-tw)*(vhi-vlo)/tf;
  else value = vlo;

  if(idxt < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PULSEut_kernel_wrapper(double *ut, double *PULSEtime, double *PULSEval,
			   double tstep, int p, int ldUt,
			   dim3 grid, dim3 block)
{
  gen_PULSEut_kernel<<<grid, block>>>(ut, PULSEtime, PULSEval,
				      tstep, p, ldUt);
}

//-----------------------------------------
__global__ void
gen_PULSEut_single_kernel(float *ut, float *PULSEtime, float *PULSEval,
		   double tstep, int p, int ldUt)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  __shared__ float t[5], v[2]; // vlo, vhi, td, tr, tf, tw, tp
  if(tid < 5) t[tid] = PULSEtime[ISidx*5+tid];
  if(tid < 2) v[tid] = PULSEval[ISidx*2+tid];
  __syncthreads();
  float td=t[0], tr=t[1], tf=t[2], tw=t[3], tp=t[4], vlo=v[0], vhi=v[1];
  float mytime=(idxt)*tstep, value=vlo; // +1
  mytime = mytime - floor(mytime/tp)*tp;

  if(mytime < td) value = vlo;
  else if(mytime < td+tr) value = vlo + (mytime-td)*(vhi-vlo)/tr;
  else if(mytime < td+tr+tw) value = vhi;
  else if(mytime < td+tr+tw+tf) value = vhi - (mytime-td-tr-tw)*(vhi-vlo)/tf;
  else value = vlo;

  if(idxt < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PULSEut_single_kernel_wrapper(float *ut, float *PULSEtime, float *PULSEval,
                                  double tstep, int p, int ldUt,
                                  dim3 grid, dim3 block)
{
  gen_PULSEut_single_kernel<<<grid, block>>>
    (ut, PULSEtime, PULSEval, tstep, p, ldUt);
}
//-----------------------------------------
__global__ void
gen_PULSEut_part_kernel(double *ut, double *PULSEtime, double *PULSEval,
			double tstep, int p, int ldUt, int shift)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  __shared__ double v[2], t[5]; // vlo, vhi, td, tr, tf, tw, tp
  if(tid < 5) t[tid] = PULSEtime[ISidx*5+tid];
  if(tid < 2) v[tid] = PULSEval[ISidx*2+tid];
  __syncthreads();
  double td=t[0], tr=t[1], tf=t[2], tw=t[3], tp=t[4], vlo=v[0], vhi=v[1];
  double mytime=(idxt+shift)*tstep, value=vlo; // +1
  mytime = mytime - floor(mytime/tp)*tp;

  if(mytime < td) value = vlo;
  else if(mytime < td+tr) value = vlo + (mytime-td)*(vhi-vlo)/tr;
  else if(mytime < td+tr+tw) value = vhi;
  else if(mytime < td+tr+tw+tf) value = vhi - (mytime-td-tr-tw)*(vhi-vlo)/tf;
  else value = vlo;

  if(idxt+shift < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PULSEut_part_kernel_wrapper(double *ut, double *PULSEtime, double *PULSEval,
                                double tstep, int p, int ldUt, int shift,
                                dim3 grid, dim3 block)
{
  gen_PULSEut_part_kernel<<<grid, block>>>
    (ut, PULSEtime, PULSEval, tstep, p, ldUt, shift);
}
//-----------------------------------------
__global__ void
gen_PULSEut_part_single_kernel(float *ut, float *PULSEtime, float *PULSEval,
			       double tstep, int p, int ldUt, int shift)
{
  int ISidx=blockIdx.x, blkIdx=blockIdx.y, blkSize=blockDim.x,
    tid=threadIdx.x, idxt=blkIdx*blkSize+tid;
  __shared__ float t[5], v[2]; // vlo, vhi, td, tr, tf, tw, tp
  if(tid < 5) t[tid] = PULSEtime[ISidx*5+tid];
  if(tid < 2) v[tid] = PULSEval[ISidx*2+tid];
  __syncthreads();
  float td=t[0], tr=t[1], tf=t[2], tw=t[3], tp=t[4], vlo=v[0], vhi=v[1];
  float mytime=(idxt+shift)*tstep, value=vlo; // +1
  mytime = mytime - floor(mytime/tp)*tp;

  if(mytime < td) value = vlo;
  else if(mytime < td+tr) value = vlo + (mytime-td)*(vhi-vlo)/tr;
  else if(mytime < td+tr+tw) value = vhi;
  else if(mytime < td+tr+tw+tf) value = vhi - (mytime-td-tr-tw)*(vhi-vlo)/tf;
  else value = vlo;

  if(idxt+shift < p)
    ut[ISidx*ldUt + idxt] = value;
}

void
gen_PULSEut_part_single_kernel_wrapper(float *ut, float *PULSEtime, float *PULSEval,
                                       double tstep, int p, int ldUt, int shift,
                                       dim3 grid, dim3 block)
{
  gen_PULSEut_part_single_kernel<<<grid, block>>>
    (ut, PULSEtime, PULSEval, tstep, p, ldUt, shift);
}
