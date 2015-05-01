#include <cuda.h>
#include <stdio.h>

void setGPUdevice()
{
  cudaDeviceProp  prop;
  memset( &prop, 0, sizeof( cudaDeviceProp ) );
  int dev;
  // int count;
  // cudaGetDeviceCount( &count );
  // for (int i=0; i< count; i++) {
  //   cudaGetDeviceProperties( &prop, i );
  //   printf( "Dev %d: Name:  %s\n", i, prop.name );
  //   printf( "        Compute capability:  %d.%d\n", prop.major, prop.minor );
  // }
  // cudaGetDevice( &dev );
  // printf( "ID of current CUDA device:  %d\n", dev );

  // prop.major = 2;
  // prop.minor = 0;
  // cudaChooseDevice( &dev, &prop );
  // printf( "ID of CUDA device closest to revision %d.%d:  %d\n",
  //         prop.major, prop.minor, dev );
  dev = 1;
  cudaSetDevice( dev );
  cudaGetDeviceProperties( &prop, dev );
  printf( "Dev %d: Name:  %s", dev, prop.name );
  printf( "        Compute capability:  %d.%d\n", prop.major, prop.minor );
}
