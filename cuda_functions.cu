#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include "Classes.hpp"

extern cusparseHandle_t cusparseHandle;

//#define EMPTY_KERNELS

// number of thread per block in each dimension - TPB^2 is the number of threads per block
const int TPB = 32;
const int n_sm = 20; // depends on hardware

// vars for xzebra
extern tfloat* d;
extern tfloat* dl;
extern tfloat* du;
extern void* pBuffer;


void CheckCudaError(cudaError_t cudaStatus, const char *const file, int const line);
int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}
/*
static void CheckCudaError(cudaError_t cudaStatus)
//void CheckCudaError(cudaError_t cudaStatus, char const *const func, const char *const file, int const line)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaStatus: %d\n", cudaStatus);
		throw "Cuda Error!";

		/*fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(cudaStatus), _cudaGetErrorEnum(cudaStatus), func);* /

	}
}*/

__constant__ tfloat c_stencil[9];
__constant__ tfloat c_residual_stencils[2*MAX_LEVELS][9];
__constant__ tfloat c_jacobi_stencils[MAX_LEVELS][9];
__constant__ tfloat c_jacobi_stencil[9];
// the first MAX_LEVELS entries of c_crv are for x-zebra, the last MAX_LEVELS entries are for y-zebra
__constant__ struct CyclicReductionValues c_crv[2*MAX_LEVELS][MAX_LEVELS];

__global__ void zeroKernel(tfloat* __restrict__ dest, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
		dest[i] = 0;
}

__global__ void zeroKernelLoop(tfloat* __restrict__ dest, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (; i < count; i += stride)
		dest[i] = 0;
}

__global__ void addKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
		dest[i] += src[i];
}

__global__ void addKernel2(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; i < count; i += stride)
	{
		dest[i] += src[i];
	}
}

__global__ void addKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, tfloat scale, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
		dest[i] += scale * src[i];
	//dest[i] = 1.0;
}

__global__ void multiplyKernel(tfloat* __restrict__ p, tfloat scale, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
		p[i] *= scale;
}

// dest = A*x
__global__ void applyStencillKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ x, int stride, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	int idx = i + j * stride;
	if (i == 0 || i >= n || j >= n)
		return;
	const tfloat* stencil = c_residual_stencils[0];
	dest[idx] = ( stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
			    + stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
			    + stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
}


// dest = rhs-A*x
__global__ void residualKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = rhs[idx] -
		      ( stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
		      + stencil[3] * x[idx - 1]          + stencil[4] * x[idx]          + stencil[5] * x[idx + 1]
		      + stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
}

// dest += w * (rhs-A*x)
__global__ void jacobiKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n, tfloat w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = w * rhs[idx] -
		(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
		+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
		+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
}

// dest = rhs-A*x
// optimized version 1
/*__global__ void residualKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = __ldg(&rhs[idx]) -
		(stencil[0] * __ldg(&x[idx - stride - 1]) + stencil[1] * __ldg(&x[idx - stride]) + stencil[2] * __ldg(&x[idx - stride + 1])
			+ stencil[3] * __ldg(&x[idx - 1]) + stencil[4] * __ldg(&x[idx]) + stencil[5] * __ldg(&x[idx + 1])
			+ stencil[6] * __ldg(&x[idx + stride - 1]) + stencil[7] * __ldg(&x[idx + stride]) + stencil[8] * __ldg(&x[idx + stride + 1]));
}*/

// dest = rhs-A*x
// optimized version 2
/*__global__ void residualKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = __ldg(&rhs[idx]) -
		(__ldg(&stencil[0]) * __ldg(&x[idx - stride - 1]) + __ldg(&stencil[1]) * __ldg(&x[idx - stride]) + __ldg(&stencil[2]) * __ldg(&x[idx - stride + 1])
			+ __ldg(&stencil[3]) * __ldg(&x[idx - 1]) + __ldg(&stencil[4]) * __ldg(&x[idx]) + __ldg(&stencil[5]) * __ldg(&x[idx + 1])
			+ __ldg(&stencil[6]) * __ldg(&x[idx + stride - 1]) + __ldg(&stencil[7]) * __ldg(&x[idx + stride]) + __ldg(&stencil[8]) * __ldg(&x[idx + stride + 1]));
}*/

// dStride - dest stride
// sStride - src stride
__global__ void restrictKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip first row (zeros)
	int didx = i + j * dStride;
	int sidx = 2*i + 2*j * sStride;

	if (i >= nDest - 1 || j >= nDest - 1)
		return;

	dest[didx] = 0.25 * (
		src[sidx] +
		0.5 * (src[sidx-1] + src[sidx+1] + src[sidx - sStride] + src[sidx + sStride]) +
		0.25 * (src[sidx - sStride - 1] + src[sidx - sStride + 1] + src[sidx + sStride - 1] + src[sidx + sStride + 1])
		);

}

// dStride - dest stride
// sStride - src stride
__global__ void restrictKernelLoop(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip first row (zeros)

	for (; j < nDest - 1; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nDest - 1; i += gridDim.x * blockDim.x)
		{
			int didx = i + j * dStride;
			int sidx = 2*i + 2*j * sStride;
			if (i > 0)
			{
				dest[didx] = 0.25 * (
					src[sidx] +
					0.5 * (src[sidx-1] + src[sidx+1] + src[sidx - sStride] + src[sidx + sStride]) +
					0.25 * (src[sidx - sStride - 1] + src[sidx - sStride + 1] + src[sidx + sStride - 1] + src[sidx + sStride + 1])
					);
			}
		}
	}
}

// dStride - dest stride
// sStride - src stride
__global__ void restrictyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip first row (zeros)

	for (; j < nyDest - 1; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx - 1; i += gridDim.x * blockDim.x)
		{
			int didx = i + j * dStride;
			int sidx = i + 2 * j * sStride;
			if (i > 0)
			{
				dest[didx] = 0.5 * (
					src[sidx] +
					0.5 * (src[sidx - sStride] + src[sidx + sStride])
					);
			}
		}
	}
}

// dStride - dest stride
// sStride - src stride
//__global__ void prolongKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nSrc)
// TODO: read each src element only once!
__global__ void prolongKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int didx = 2 * i + 2 * j * dStride;
	int sidx = i + j * sStride;

	if (2 * i >= nDest - 1 || 2 * j >= nDest - 1)
		return;

	dest[didx] = src[sidx];
	if (2 * i + 1 < nDest - 1)
		dest[didx + 1] = 0.5 * (src[sidx] + src[sidx + 1]);
	if (2 * j + 1 < nDest - 1)
		dest[didx + dStride] = 0.5 * (src[sidx] + src[sidx + sStride]);
	if (2 * i + 1 < nDest - 1 && 2 * j + 1 < nDest - 1)
		dest[didx + dStride + 1] = 0.25 * (src[sidx] + src[sidx + 1] + src[sidx + sStride] + src[sidx + sStride + 1]);
}

// dStride - dest stride
// sStride - src stride
__global__ void prolongyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int didx = i + 2 * j * dStride;
	int sidx = i + j * sStride;

	if (i >= nx - 1 || 2 * j >= nyDest - 1)
		return;

	dest[didx] = src[sidx];
	if (2 * j + 1 < nyDest - 1)
		dest[didx + dStride] = 0.5 * (src[sidx] + src[sidx + sStride]);
}

// dStride - dest stride
// sStride - src stride
//__global__ void prolongKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nSrc)
// TODO: read each src element only once!
__global__ void prolongKernelLoop(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	for (; 2 * j < nDest - 1; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 2 * i < nDest - 1; i += gridDim.x * blockDim.x)
		{
			int didx = 2 * i + 2 * j * dStride;
			int sidx = i + j * sStride;
			//tfloat src_0 = src[sidx];
			//tfloat src_1 = 0.0;
			//tfloat src_n = 0.0;
			dest[didx] = src[sidx];
			//if (2 * i + 1 < nDest - 1) // TODO: nDest is an odd number, so this 'if' may be unnecesery, need to verify that
			{
				//src_1 = src[sidx+1];
				dest[didx + 1] = 0.5*(src[sidx] + src[sidx + 1]);
			}
			//if (2 * j + 1 < nDest - 1)
			{
				//src_n = src[sidx+sStride];
				dest[didx + dStride] = 0.5*(src[sidx] + src[sidx + sStride]);
			}
			//if (2 * i + 1 < nDest - 1 && 2 * j + 1 < nDest - 1)
			{
				dest[didx + dStride + 1] = 0.25*(src[sidx] + src[sidx + 1] + src[sidx + sStride] + src[sidx + sStride + 1]);
			}
		}
	}
}

// dStride - dest stride
// sStride - src stride
//__global__ void prolongKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nSrc)
// reading each src element only once per thread, no if
__global__ void prolongKernelLoop1(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	for (; 2 * j < nDest - 1; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 2 * i < nDest - 1; i += gridDim.x * blockDim.x)
		{
			int didx = 2 * i + 2 * j * dStride;
			int sidx = i + j * sStride;
			tfloat src_0 = src[sidx];
			tfloat src_1 = src[sidx+1];
			tfloat src_n = src[sidx + sStride];
			dest[didx] = src_0;
			src_1 = src_1;
			dest[didx + 1] = 0.5*(src_0 + src_1);
			dest[didx + dStride] = 0.5*(src_0 + src_n);
			dest[didx + dStride + 1] = 0.25*(src_0 + src_1 + src_n + src[sidx + sStride + 1]);
		}
	}
}


__global__ void incKernel(tfloat* __restrict__ p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
		p[idx]++;
}

__global__ void noneKernel(tfloat* __restrict__ p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0)
		p[idx] = 0;
}

// dest = rhs-A*x
// for 0 <= i,j < n
__global__ void residualKernelConst(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = rhs[idx] -
		(c_stencil[0] * x[idx - stride - 1] + c_stencil[1] * x[idx - stride] + c_stencil[2] * x[idx - stride + 1]
			+ c_stencil[3] * x[idx - 1] + c_stencil[4] * x[idx] + c_stencil[5] * x[idx + 1]
			+ c_stencil[6] * x[idx + stride - 1] + c_stencil[7] * x[idx + stride] + c_stencil[8] * x[idx + stride + 1]);
}

// dest = rhs-A*x
// for 0 < i,j < n
// The function used 30 registers, 356 bytes cmem[0]
__global__ void residualKernelConstCoalescingArray(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, int level)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
		return;
	dest[idx] = rhs[idx] -
		(c_residual_stencils[level][0] * x[idx - stride - 1] + c_residual_stencils[level][1] * x[idx - stride] + c_residual_stencils[level][2] * x[idx - stride + 1]
		+ c_residual_stencils[level][3] * x[idx - 1] + c_residual_stencils[level][4] * x[idx] + c_residual_stencils[level][5] * x[idx + 1]
		+ c_residual_stencils[level][6] * x[idx + stride - 1] + c_residual_stencils[level][7] * x[idx + stride] + c_residual_stencils[level][8] * x[idx + stride + 1]);
}

// dest = rhs-A*x
// for 0 < i,j < n

// 08/02/2020 - with -G compiler flag
//1>ptxas info : Compiling entry function '_Z18residualKernelLoopPdPKdS1_iii' for 'sm_61'
//1>ptxas info : Function properties for _Z18residualKernelLoopPdPKdS1_iii
//1>    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
//1>ptxas info : Used 23 registers, 356 bytes cmem[0]
// 08/02/2020 - without -G compiler flag
//1>ptxas info : Compiling entry function '_Z18residualKernelLoopPdPKdS1_iii' for 'sm_61'
//1>ptxas info : Function properties for _Z18residualKernelLoopPdPKdS1_iii
//1>    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
//1>ptxas info : Used 40 registers, 356 bytes cmem[0]


__global__ void residualKernelLoop(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, int level)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	//int k = threadIdx.x + threadIdx.y;

	const tfloat* stencil = c_residual_stencils[level];

	//int xThreads = gridDim.x * blockDim.x;
	//int yThreads = gridDim.y * blockDim.y;
	for (; j < n; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
		{
			int idx = i + j * stride;
			if (i > 0)
			{
				/*dest[idx] = rhs[idx] -
					(c_residual_stencils[level][0] * x[idx - stride - 1] + c_residual_stencils[level][1] * x[idx - stride] + c_residual_stencils[level][2] * x[idx - stride + 1]
					+ c_residual_stencils[level][3] * x[idx - 1] + c_residual_stencils[level][4] * x[idx] + c_residual_stencils[level][5] * x[idx + 1]
					+ c_residual_stencils[level][6] * x[idx + stride - 1] + c_residual_stencils[level][7] * x[idx + stride] + c_residual_stencils[level][8] * x[idx + stride + 1]);*/
				dest[idx] = rhs[idx] -
					(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
					+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
					+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			}
		}
	}
}

__global__ void
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__launch_bounds__(1024, 2)
// (1024, 2) => 32 registers
//2>ptxas info : Compiling entry function '_Z27residualKernelLoopRegsLimitPdPKdS1_iii' for 'sm_61'
//2>ptxas info : Function properties for _Z27residualKernelLoopRegsLimitPdPKdS1_iii
//2>    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
//2>ptxas info : Used 32 registers, 356 bytes cmem[0]
residualKernelLoopRegsLimit(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, int level)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)

	const tfloat* stencil = c_residual_stencils[level];

	for (; j < n; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
		{
			int idx = i + j * stride;
			if (i > 0)
			{
				/*dest[idx] = rhs[idx] -
				(c_residual_stencils[level][0] * x[idx - stride - 1] + c_residual_stencils[level][1] * x[idx - stride] + c_residual_stencils[level][2] * x[idx - stride + 1]
				+ c_residual_stencils[level][3] * x[idx - 1] + c_residual_stencils[level][4] * x[idx] + c_residual_stencils[level][5] * x[idx + 1]
				+ c_residual_stencils[level][6] * x[idx + stride - 1] + c_residual_stencils[level][7] * x[idx + stride] + c_residual_stencils[level][8] * x[idx + stride + 1]);*/
				dest[idx] = rhs[idx] -
					(  stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
					 + stencil[3] * x[idx - 1]          + stencil[4] * x[idx]          + stencil[5] * x[idx + 1]
					 + stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			}
		}
	}
}

__global__ void
__launch_bounds__(1024, 2)
// (1024, 2) => 32 registers
residualKernelLoopRegsLimit(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int nx, int ny, int level)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)

	const tfloat* stencil = c_residual_stencils[level];

	for (; j < ny; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx; i += gridDim.x * blockDim.x)
		{
			int idx = i + j * stride;
			if (i > 0)
			{
				dest[idx] = rhs[idx] -
					(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
						+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
						+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			}
		}
	}
}


// dest = rhs-A*x
// for 0 < i,j < n
__global__ void residualKernelConstCoalescing(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
		return;
	dest[idx] = rhs[idx] -
		(c_stencil[0] * x[idx - stride - 1] + c_stencil[1] * x[idx - stride] + c_stencil[2] * x[idx - stride + 1]
		+ c_stencil[3] * x[idx - 1] + c_stencil[4] * x[idx] + c_stencil[5] * x[idx + 1]
		+ c_stencil[6] * x[idx + stride - 1] + c_stencil[7] * x[idx + stride] + c_stencil[8] * x[idx + stride + 1]);
}

// dest += w * (rhs-A*x)
// for 0 <= i,j < n
__global__ void jacobiKernelConst(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i >= n || j >= n)
		return;
	dest[idx] = w * rhs[idx] -
		(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
		+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
		+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);
}

// dest += w * (rhs-A*x)
// for 0 < i,j < n
__global__ void jacobiKernelConstCoalescing(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
	//if (i == 0 || i >= n || j >= n)
			return;
	dest[idx] = w * rhs[idx] -
		(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
		+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
		+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);
}

// dest += w * (rhs-A*x)
// for 0 < i,j < n
__global__ void jacobiKernelLevel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w, int level)
{
	const tfloat* stencil = c_jacobi_stencils[level];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
	//if (i == 0 || i >= n || j >= n)
			return;
	dest[idx] = w * rhs[idx] -
		(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
		+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
		+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
}

// dest += w * (rhs-A*x)
// for 0 < i < nx, 0 < j < ny
__global__ void jacobiKernelLevel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int nx, int ny, tfloat w, int level)
{
	const tfloat* stencil = c_jacobi_stencils[level];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= nx || j >= ny)
		//if (i == 0 || i >= n || j >= n)
		return;
	dest[idx] = w * rhs[idx] -
		(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
			+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
			+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
}

/*__global__ void jacobiKernelLevelSingleBlock(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, tfloat w, int level, int n_times)
{
	const tfloat* stencil = c_jacobi_stencils[level];
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int idx = i + j * stride;
	bool bUse = i > 0 && j > 0 && i < nx&& j < ny;
	for (int iIter = 0; iIter < n_times; iIter++)
	{
		tfloat val = 0.0;
		if (bUse)
		{
			val = w * rhs[idx] -
				(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
					+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
					+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
		}
		__syncthreads();
		if (bUse)
			x[idx] = val;
	}
}

/**/

__global__ void jacobiKernelLevelSingleBlock(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, tfloat w, int level, int n_times)
{
	//const int SIZE = 33;
	const int SIZE = 17;
	const tfloat* stencil = c_jacobi_stencils[level];
	int ii = threadIdx.x + 1;
	int jj = threadIdx.y + 1;
	int idx = ii + jj * stride;
	bool bUse = ii > 0 && jj > 0 && ii < nx && jj < ny;
	__shared__ tfloat xx[SIZE][SIZE];
	tfloat rr = w * rhs[idx];
	if (threadIdx.y == 0)
	{
		xx[0][threadIdx.x] = 0.0;
		xx[SIZE-1][threadIdx.x] = 0.0;
		xx[threadIdx.x][0] = 0.0;
		xx[threadIdx.x][SIZE-1] = 0.0;
		if (threadIdx.x == 0)
		{
			xx[SIZE-1][SIZE-1] = 0.0;
		}
	}
	tfloat val = 0.0;
	if (bUse)
	{
		tfloat val = x[idx];
		xx[jj][ii] = val;
	}
	__syncthreads();
	for (int iIter = 0; iIter < n_times; iIter++)
	{
		if (bUse)
		{
			//tfloat Ax = (stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
				//+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
				//+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			tfloat Ax = (stencil[0] * xx[jj-1][ii-1] + stencil[1] * xx[jj-1][ii] + stencil[2] * xx[jj-1][ii+1]
				+ stencil[3] * xx[jj][ii-1] + stencil[4] * xx[jj][ii] + stencil[5] * xx[jj][ii+1]
				+ stencil[6] * xx[jj + 1][ii-1] + stencil[7] * xx[jj + 1][ii] + stencil[8] * xx[jj + 1][ii+1]);
			//tfloat Ax = 0.2 * (xx[jj - 1][ii - 1] + xx[jj - 1][ii] + xx[jj - 1][ii + 1]
				//+ xx[jj][ii - 1] + xx[jj][ii] + xx[jj][ii + 1]
				//+ xx[jj + 1][ii - 1] + xx[jj + 1][ii] + xx[jj + 1][ii + 1]);
			val = rr - Ax;
		}
		__syncthreads();
		if (bUse)
		{
			xx[jj][ii] = val;
			//x[idx] = val;
		}
		__syncthreads();
	}
	if (bUse)
		x[idx] = val;
}

/*__device__  void dCyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n)
{
	const int MAX_LEVELS = 13;
	const int MAX_N = 8 * 1024 + 2;
	tfloat a_a[MAX_LEVELS], b_a[MAX_LEVELS], c_a[MAX_LEVELS];
	tfloat aa, bb, cc;
	tfloat ff[MAX_N], uu[MAX_N];
	tfloat* = pf = ff;

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		for (int ii = 0, i = 1; ii < nn; ii++, i += 2)
		{
			ff[ii] = f[i - 1] * b / a - f[i] + f[i + 1] * c / a;
		}
		a_a[iLevel] = a;
		b_a[iLevel] = b;
		c_a[iLevel] = c;

		bb = b * b / a;
		cc = c * c / a;
		aa = 2 * b * c / a - a;
		tfloat* cf = pf + n + 2;

		iLevel++;
		n = nn;
		a = aa;
		b = bb;
		c = cc;
	}

	if (n == 1)
	{
		u[0] = 0.0;
		u[1] = f[0] / a;
		u[2] = 0.0;
		return;
	}

	return;

	while (iLevel >= 0)
	{
		for (int ii = 0, i = 0; ii < nn + 2; ii++, i += 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			u[i] = uu[ii];
		}
		for (int ii = 0, i = 1; ii < nn + 1; ii++, i += 2)
		{
			u[i] = (f[i - 1] - b * uu[ii] - c * uu[ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}

		iLevel--;
	}

}*/

__device__ void dIterativeCyclicReduction_1thread(tfloat a, tfloat b, tfloat c, tfloat* fs, tfloat* u, int n)
{
	const int MAX_LEVELS = 14;
	//const int MAX_N = 2 * 8 * 1024 + 2;
	const int MAX_N = 2 * 1024 + 2;
	tfloat as[MAX_LEVELS], bs[MAX_LEVELS], cs[MAX_LEVELS];
	tfloat aa, bb, cc;
	__shared__ tfloat us[MAX_N];

	int s = 0; // current level start index
	int ic = n + 2; //coarser level start index

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		for (int ii = 0, i = 1; ii < nn; ii++, i += 2)
		{
			fs[ic + ii] = fs[s + i - 1] * b / a - fs[s + i] + fs[s + i + 1] * c / a;
		}
		as[iLevel] = a;
		bs[iLevel] = b;
		cs[iLevel] = c;

		// prepare next level:
		bb = b * b / a;
		cc = c * c / a;
		aa = 2 * b * c / a - a;

		iLevel++;
		n = nn;
		a = aa;
		b = bb;
		c = cc;

		s = ic;
		ic += nn + 2;
	}

	if (n == 1) // must be true here
	{
		us[s + 0] = 0.0;
		us[s + 1] = fs[s + 0] / a;
		us[s + 2] = 0.0;
	}

	while (--iLevel >= 0)
	{
		int nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;
		a = as[iLevel];
		b = bs[iLevel];
		c = cs[iLevel];
		for (int ii = 0, i = 0; ii < nn + 2; ii++, i += 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			us[s + i] = us[ic + ii];
		}
		for (int ii = 0, i = 1; ii < nn + 1; ii++, i += 2)
		{
			us[s + i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}
	}

	for (int i = 0; i < n + 2; i++)
		u[i] = us[i];
}


__global__ void xzebraKernel_1threadPerBlock(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstRow)
{
	const tfloat* stencil = c_residual_stencils[level];
	int i = threadIdx.x;
	int iRow = firstRow + 2 * blockIdx.x;
	tfloat* u = x + iRow * stride;
	const tfloat* f = rhs + iRow * stride;
	const int MAX_N = 2 * 1024 + 2;
	__shared__ tfloat fs[MAX_N];

	tfloat a = stencil[4];
	tfloat b = stencil[3];
	tfloat c = stencil[5];

	if (iRow < ny)
	{
		if (threadIdx.x == 0)
		{
			for (int i = 1; i <= nx; i++)
			{
				int idx = i + iRow * stride;
				// r is the sum of weighted values for the row before and the row after i
				tfloat r = 0.0;
				r += stencil[0] * x[idx - stride - 1];
				r += stencil[1] * x[idx - stride];
				r += stencil[2] * x[idx - stride + 1];
				r += stencil[6] * x[idx + stride - 1];
				r += stencil[7] * x[idx + stride];
				r += stencil[8] * x[idx + stride + 1];
				fs[i] = rhs[idx] - r;
			}

			//dIterativeCyclicReduction(a, b, c, f + 1, u, nx);
			dIterativeCyclicReduction_1thread(a, b, c, fs + 1, u, nx);
		}
	}
}

/*
from: Cyclic Reduction Tridiagonal Solvers on GPUs Applied to Mixed-Precision Multigrid - section 5.3

On the CPU, tridiagonal solvers are implemented using the
Thomas algorithm, which is essentially Gaussian elimination
and requires 6M operations for a system with M unknowns.
Cyclic reduction is also linear in the number of unknowns,
but is more expensive, and performs approximately 23M
arithmetic operations.

in my code (a special case):
3 float divisions per unknown at the first level
8 float multiplications per unknown at the first level
*/
__device__ void dIterativeCyclicReduction(tfloat a, tfloat b, tfloat c, tfloat* __restrict__ fs, tfloat* __restrict__ u, int n)
{
	const int MAX_LEVELS = 14;
	//const int MAX_N = 2 * 8 * 1024 + 2;
	const int MAX_N = 2 * 1024 + 2;
	tfloat as[MAX_LEVELS], bs[MAX_LEVELS], cs[MAX_LEVELS];
	__shared__ tfloat us[MAX_N];
	int ii = threadIdx.x;

	int s = 0; // current level start index
	int ic = n + 2; //coarser level start index

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		//a = 1 / a;
		tfloat fl = b / a;
		tfloat fu = c / a;
		//tfloat fl = b * a;
		//tfloat fu = c * a;
		if (ii < nn)
		{
			int i = 1 + 2 * ii;
			fs[ic + ii] = fl * fs[s + i - 1] - fs[s + i] + fu * fs[s + i + 1];
		}

		__syncthreads();

		as[iLevel] = a;
		bs[iLevel] = b;
		cs[iLevel] = c;

		// prepare next level:
		a = 2 * b * fu - a; // OK only for constant values of a, b, c
		b = fl * b;
		c = fu * c;

		iLevel++;
		n = nn;

		s = ic;
		ic += nn + 2;
	}

	//if (n == 1) // must be true here
	if (threadIdx.x == 0)
	{
		us[s + 0] = 0.0;
		us[s + 1] = fs[s + 0] / a;
		us[s + 2] = 0.0;
	}
	__syncthreads();

	while (--iLevel >= 0)
	{
		int nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;
		a = as[iLevel];
		b = bs[iLevel];
		c = cs[iLevel];
		if (ii < nn + 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			int i = 2 * ii;
			us[s + i] = us[ic + ii];
		}
		if (ii < nn + 1)
		{
			int i = 1 + 2 * ii;
			us[s + i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}
		__syncthreads();
	}

	if (ii < n + 2)
		u[ii] = us[ii];
	__syncthreads();
}

/*
in my code (a special case):
1 float divisions per unknown at the first level
4 float multiplications per unknown at the first level
*/
__device__ void dIterativeCyclicReductionConst(tfloat* __restrict__ fs, tfloat* __restrict__ u, int n, int c_crv_index)
{
	const int MAX_LEVELS = 14;
	//const int MAX_N = 2 * 8 * 1024 + 2;
	const int MAX_N = 2 * 1024 + 2;
	__shared__ tfloat us[MAX_N];
	int ii = threadIdx.x;

	int s = 0; // current level start index
	int ic = n + 2; //coarser level start index

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		if (ii < nn)
		{
			tfloat fl = c_crv[c_crv_index][iLevel].fl;
			tfloat fu = c_crv[c_crv_index][iLevel].fu;
			int i = 1 + 2 * ii;
			fs[ic + ii] = fl * fs[s + i - 1] - fs[s + i] + fu * fs[s + i + 1];
		}

		__syncthreads();

		iLevel++;
		n = nn;

		s = ic;
		ic += nn + 2;
	}

	//if (n == 1) // must be true here
	if (threadIdx.x == 0)
	{
		us[s + 0] = 0.0;
		us[s + 1] = fs[s + 0] / c_crv[c_crv_index][iLevel].a;
		us[s + 2] = 0.0;
	}
	__syncthreads();

	while (--iLevel >= 0)
	{
		int nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;

		if (ii < nn + 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			int i = 2 * ii;
			us[s + i] = us[ic + ii];
		}
		if (ii < nn + 1)
		{
			tfloat a = c_crv[c_crv_index][iLevel].a;
			tfloat b = c_crv[c_crv_index][iLevel].b;
			tfloat c = c_crv[c_crv_index][iLevel].c;
			int i = 1 + 2 * ii;
			us[s + i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}
		__syncthreads();
	}

	if (ii < n + 2)
		u[ii] = us[ii];
	//__syncthreads(); // not needed?
}

__device__ void dIterativeCyclicReductionHuge(tfloat* __restrict__ fs, tfloat* __restrict__ u, tfloat* us, int n, int c_crv_index)
{
	const int MAX_LEVELS = 14;
	//const int MAX_N = 2 * 8 * 1024 + 2;
	const int MAX_N = 2 * 16*1024 + 2;
	//int ii = threadIdx.x;

	int s = 0; // current level start index
	int ic = n + 2; //coarser level start index

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		for (int ii = threadIdx.x; ii < nn; ii += blockDim.x)
		{
			tfloat fl = c_crv[c_crv_index][iLevel].fl;
			tfloat fu = c_crv[c_crv_index][iLevel].fu;
			int i = 1 + 2 * ii;
			fs[ic + ii] = fl * fs[s + i - 1] - fs[s + i] + fu * fs[s + i + 1];
		}

		__syncthreads();

		iLevel++;
		n = nn;

		s = ic;
		ic += nn + 2;
	}

	//if (n == 1) // must be true here
	if (threadIdx.x == 0)
	{
		us[s + 0] = 0.0;
		us[s + 1] = fs[s + 0] / c_crv[c_crv_index][iLevel].a;
		us[s + 2] = 0.0;
	}
	__syncthreads();

	while (--iLevel >= 0)
	{
		int nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;

		//if (ii < nn + 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		for (int ii = threadIdx.x; ii < nn + 2; ii += blockDim.x)
		{
			int i = 2 * ii;
			us[s + i] = us[ic + ii];
		}
		//if (ii < nn + 1)
		for (int ii = threadIdx.x; ii < nn + 1; ii += blockDim.x)
		{
			tfloat a = c_crv[c_crv_index][iLevel].a;
			tfloat b = c_crv[c_crv_index][iLevel].b;
			tfloat c = c_crv[c_crv_index][iLevel].c;
			int i = 1 + 2 * ii;
			us[s + i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}
		__syncthreads();
	}

	for (int ii = threadIdx.x; ii < n + 2; ii += blockDim.x)
		u[ii] = us[ii];
	//__syncthreads(); // not needed?
}

// B is the right hand side of the equation system as input and the solution as output
// n is the number of vars, NOT including the 2 zeros at the start and at the end
// n+1 is a power of 2 (n/2 + 1 is also a power of 2)
__device__ void dIterativeCyclicReductionHuge2(tfloat* __restrict__ B, tfloat* __restrict__ buffer, int n, int c_crv_index)
{
	int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
	for (int ii = threadIdx.x; ii < nn; ii += blockDim.x)
	{
		tfloat fl = c_crv[c_crv_index][0].fl;
		tfloat fu = c_crv[c_crv_index][0].fu;
		int i = 1 + 2 * ii + 1; // the additional +1 is because the first non-zero value is at index 1 in B
		buffer[ii] = fl * B[i - 1] - B[i] + fu * B[i + 1];
	}
	__syncthreads();

	int s = 0; // current level start index
	int ic = nn + 2; //coarser level start index
	int iLevel = 1;
	while (nn > 1)
	{
		nn = nn / 2; // nn is an odd number, so nn is rounded down, nn(new)*2+1=nn(old)
		for (int ii = threadIdx.x; ii < nn; ii += blockDim.x)
		{
			tfloat fl = c_crv[c_crv_index][iLevel].fl;
			tfloat fu = c_crv[c_crv_index][iLevel].fu;
			int i = 1 + 2 * ii;
			buffer[ic + ii] = fl * buffer[s + i - 1] - buffer[s + i] + fu * buffer[s + i + 1];
		}

		__syncthreads();

		iLevel++;

		s = ic;
		ic += nn + 2;
	}

	int stride = (n+1)/2;
	//if (n == 1) // must be true here
	if (threadIdx.x == 0)
	{
		//B[0] = 0.0;
		B[(n+1)/2] = buffer[s + 0] / c_crv[c_crv_index][iLevel].a;
		//B[n + 1] = 0.0;
		//printf("nn=%d, stride=%d\n", nn, stride); //nn=1, stride=1024
		buffer[s + 0] = 0.0;
	}
	__syncthreads();

	n = 1;
	while (--iLevel >= 1)
	{
		nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;

		//if (ii < nn + 1)
		for (int ii = threadIdx.x; ii < nn + 1; ii += blockDim.x)
		{
			tfloat a = c_crv[c_crv_index][iLevel].a;
			tfloat b = c_crv[c_crv_index][iLevel].b;
			tfloat c = c_crv[c_crv_index][iLevel].c;
			int i = 1 + 2 * ii;
			B[stride/2 + ii*stride] = (buffer[s + i - 1] - b * B[ii*stride] - c * B[(ii+1)*stride]) / a; // u[1] is the first value in u, while f[0] is the first in f
			buffer[s + i - 1] = 0.0;
		}
		stride /= 2;
		__syncthreads();
	}
	//if (threadIdx.x == 0)
		//printf("s=%d\n", s); //s=0


	nn = n;
	//printf("nn=%d, stride=%d\n", nn, stride); //nn=1023, stride=2
	for (int ii = threadIdx.x; ii < nn + 1; ii += blockDim.x)
	{
		tfloat a = c_crv[c_crv_index][iLevel].a;
		tfloat b = c_crv[c_crv_index][iLevel].b;
		tfloat c = c_crv[c_crv_index][iLevel].c;
		B[stride / 2 + ii * stride] = (B[stride / 2 + ii * stride] - b * B[ii * stride] - c * B[(ii + 1) * stride]) / a; // u[1] is the first value in u, while f[0] is the first in f
	}

	/*if (threadIdx.x == 0)
	{
		printf("B=[%lf, %lf, %lf]\n", B[0], B[1], B[2]);
	}*/
	// buffer is a matrix that needs zero boundraies.
	// It is a little ugly to make it here, but should work for now.
	/*if (threadIdx.x == 0)
	{
		buffer[0] = 0.0;
		buffer[(nn+1)*2] = 0.0; // orignal n and nn values have been changed here, (nn+1)*2 should be equal to original n value
		buffer[stride] = 0.0;
	}*/
}

// B is the right hand side of the equation system as input and the solution as output
// n is the number of vars, NOT including the 2 zeros at the start and at the end
// n+1 is a power of 2 (n/2 + 1 is also a power of 2)
__device__ void dIterativeCyclicReductionInPlace(tfloat* __restrict__ B, int n, int c_crv_index)
{

	int nn = n;
	int stride = 1;
	int iLevel = 0;
	while (nn > 1)
	{
		nn = nn / 2; // nn is an odd number, so nn is rounded down, nn(new)*2+1=nn(old)
		tfloat fl = c_crv[c_crv_index][iLevel].fl;
		tfloat fu = c_crv[c_crv_index][iLevel].fu;
		for (int ii = threadIdx.x; ii < nn; ii += blockDim.x)
		{
			int i = (ii+1) * 2 * stride; // the additional +1 is because the first non-zero value is at index 1 in B
			B[i] = fl * B[i - stride] - B[i] + fu * B[i + stride];
		}
		__syncthreads();
		iLevel++;
		stride *= 2;
	}

	//if (nn == 1) // must be true here
	if (threadIdx.x == 0)
	{
		B[stride] = B[stride] / c_crv[c_crv_index][iLevel].a;
		//printf("n=%d, nn=%d, stride=%d\n", n, nn, stride); //n=2047, nn=1, stride=1024
	}
	__syncthreads();

	while (--iLevel >= 0)
	{
		stride /= 2;
		tfloat a = c_crv[c_crv_index][iLevel].a;
		tfloat b = c_crv[c_crv_index][iLevel].b;
		tfloat c = c_crv[c_crv_index][iLevel].c;
		for (int ii = threadIdx.x; ii < nn + 1; ii += blockDim.x)
		{
			int i = stride + 2 * stride * ii;
			B[i] = (B[i] - b * B[i-stride] - c * B[i+stride]) / a;
		}
		__syncthreads();
		nn = nn * 2 + 1;
	}
	/*if (threadIdx.x == 0)
	{
		printf("n=%d, nn=%d, stride=%d\n", n, nn, stride); //n=2047, nn=2047, stride=1
	}*/
}

// xzebraKernel has shared mem, and it calls dIterativeCyclicReductionConst which also has shared mem
// shared memory: (2 * 1024 + 2) * 2 * sizeof(tfloat) = 4100 * sizeof(tfloat)
// for sizeof(tfloat) = 8 => shared mem = 32,800 bytes

__global__ void xzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstRow)
{
	const tfloat* stencil = c_residual_stencils[level];
	int i = threadIdx.x;
	int iRow = firstRow + 2 * blockIdx.x;
	//tfloat* __restrict__ u = x + iRow * stride;
	const int MAX_N = 2 * 1024 + 2;
	__shared__ tfloat fs[MAX_N];

	tfloat a = stencil[4];
	tfloat b = stencil[3];
	tfloat c = stencil[5];

	//while (iRow < ny)
	if (iRow < ny) // for some reason, if is faster than while here
	{
		if (i >= 1 && i <= nx)
		{
			int idx = i + iRow * stride;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[1] * x[idx - stride];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[7] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			fs[i] = rhs[idx] - r;
		}

		__syncthreads();

		//dIterativeCyclicReduction(a, b, c, fs + 1, u, nx);
		dIterativeCyclicReductionConst(fs + 1, x + iRow * stride, nx, level);

		iRow += 2 * gridDim.x;
	}
}

// tmp and tmp2 must have at least twice the space of x in each row
__global__ void xzebraKernelHuge(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, tfloat* __restrict__ tmp, tfloat* __restrict__ tmp2, int stride, int nx, int ny, int level, int firstRow, int tmp_stride)
{
	const tfloat* stencil = c_residual_stencils[level];
	int i = threadIdx.x;
	int iRow = firstRow + 2 * blockIdx.x;

	tfloat a = stencil[4];
	tfloat b = stencil[3];
	tfloat c = stencil[5];

	//while (iRow < ny)
	if (iRow < ny) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x; i < nx; i += blockDim.x)
		{
			int idx = 1 + i + iRow * stride;
			int tmp_idx = i + iRow * tmp_stride;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[1] * x[idx - stride];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[7] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			tmp[tmp_idx] = rhs[idx] - r;
		}

		__syncthreads();

		dIterativeCyclicReductionHuge(tmp + iRow * tmp_stride, x + iRow * stride, tmp2 + iRow * tmp_stride, nx, level);

		iRow += 2 * gridDim.x;
	}
}

__global__ void prepareXzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstRow)
{
	const tfloat* stencil = c_residual_stencils[level];
	int i = threadIdx.x;
	int iRow = firstRow + 2 * blockIdx.x;

	tfloat a = stencil[4];
	tfloat b = stencil[3];
	tfloat c = stencil[5];

	//while (iRow < ny)
	if (iRow < ny) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x; i < nx; i += blockDim.x)
		{
			int idx = 1 + i + iRow * stride;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[1] * x[idx - stride];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[7] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			x[idx] = rhs[idx] - r;
		}

		//iRow += 2 * gridDim.x;
	}
}

__global__ void prepareYzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstColumn, tfloat* __restrict__ tmp)
{
	const tfloat* stencil = c_residual_stencils[level];
	//int i = threadIdx.x;
	int iCol = firstColumn + 2 * blockIdx.x;

	//while (iRow < ny)
	if (iCol < nx) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x + 1; i <= ny; i += blockDim.x)
		{
			int idx = i * stride + iCol;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[3] * x[idx - 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[5] * x[idx + 1];
			r += stencil[8] * x[idx + stride + 1];
			int t_idx = iCol * stride + i; // tmp is transposed
			tmp[t_idx] = rhs[idx] - r; // tmp is transposed
		}
	}
}

/*
__global__ void yzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstColumn, tfloat* __restrict__ tmp)
{
	//const tfloat* stencil = c_residual_stencils[level];
	//int i = threadIdx.x;
	int iCol = firstColumn + 2 * blockIdx.x;

	//tfloat a = stencil[4];
	//tfloat b = stencil[1];
	//tfloat c = stencil[7];

	//while (iRow < ny)
	if (iCol < nx) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x; i < ny; i += blockDim.x)
		{
			int idx = 1 + i * stride + iCol;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[1] * x[idx - stride];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[7] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			int t_idx = 1 + iCol * stride + i; // tmp is transposed
			tmp[t_idx] = rhs[idx] - r; // tmp is transposed
		}

		//__syncthreads();

		//dIterativeCyclicReductionHuge2(x + iRow * stride, tmp + iRow * stride, nx, level);
		//dIterativeCyclicReductionInPlace(x + iRow * stride, nx, level);
	}
}*/

__global__ void inPlaceXzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstRow)
{
	const tfloat* stencil = c_residual_stencils[level];
	//int i = threadIdx.x;
	int iRow = firstRow + 2 * blockIdx.x;

	//tfloat a = stencil[4];
	//tfloat b = stencil[3];
	//tfloat c = stencil[5];

	//while (iRow < ny)
	if (iRow < ny) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x; i < nx; i += blockDim.x)
		{
			int idx = 1 + i + iRow * stride;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[1] * x[idx - stride];
			r += stencil[2] * x[idx - stride + 1];
			r += stencil[6] * x[idx + stride - 1];
			r += stencil[7] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			x[idx] = rhs[idx] - r;
		}

		__syncthreads();

		//dIterativeCyclicReductionHuge2(x + iRow * stride, tmp + iRow * stride, nx, level);
		dIterativeCyclicReductionInPlace(x + iRow * stride, nx, level);
		// tmp is a matrix that needs zero boundraies.
		// It is a little ugly to make it here, but should work for now.
		/*if (threadIdx.x == 0)
		{
			tmp[iRow * stride] = 0.0;
			tmp[iRow * stride + nx+1] = 0.0; // orignal n and nn values have been changed here
			tmp[iRow * stride + stride] = 0.0;
		}
		if (iRow == ny - 1)
		{
			for (int i = threadIdx.x; i < nx + 2; i += blockDim.x)
			{
				int idx = i + iRow * stride;
				tmp[idx + stride] = 0.0;
			}
		}*/

		//iRow += 2 * gridDim.x;
	}
}

// the x input is transposed, so this function actually acts on the rows (like xzebra)
// nx and ny are counts of x and y, for the transposed matrix
// the matrix rhs is not transposed, so its indices are different
__global__ void inPlaceYzebraKernel(tfloat* __restrict__ x, const tfloat* __restrict__ rhs, int stride, int nx, int ny, int level, int firstRow)
{
	const tfloat* stencil = c_residual_stencils[level];
	int iRow = firstRow + 2 * blockIdx.x;

	//while (iRow < ny)
	if (iRow < ny) // for some reason, if is faster than while here
	{
		for (int i = threadIdx.x + 1; i <= nx; i += blockDim.x)
		{
			int idx = iRow * stride + i;
			// r is the sum of weighted values for the row before and the row after i
			tfloat r = 0.0;
			r += stencil[0] * x[idx - stride - 1];
			r += stencil[3] * x[idx - stride];
			r += stencil[6] * x[idx - stride + 1];
			r += stencil[2] * x[idx + stride - 1];
			r += stencil[5] * x[idx + stride];
			r += stencil[8] * x[idx + stride + 1];
			int rhs_idx = i * stride + iRow; // x is transposed, but rhs is not
			x[idx] = rhs[rhs_idx] - r;
		}

		__syncthreads();

		dIterativeCyclicReductionInPlace(x + iRow * stride, nx, MAX_LEVELS+level);
	}
}
// the x input is transposed, so this function actually acts on the rows (like xzebra)
// nx and ny are counts of x and y, for the transposed matrix
// the matrix should already contain the right side as input (in x)
__global__ void inPlaceTransposeYzebraKernel(tfloat* __restrict__ x, int stride, int nx, int ny, int level, int firstRow)
{
	//const tfloat* stencil = c_residual_stencils[level];
	int iRow = firstRow + 2 * blockIdx.x;

	if (iRow < ny) // for some reason, if is faster than while here
	{

		dIterativeCyclicReductionInPlace(x + iRow * stride, nx, MAX_LEVELS+level);
	}
}

__global__ void transposeKernel(tfloat* __restrict__ x, tfloat* __restrict__ dest, int stride, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // skip first col (zeros)
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	if (i < n && j < n)
	{
		int s_idx = i + j * stride;
		int d_idx = i * stride + j;
		dest[d_idx] = x[s_idx];
	}
}

// for bandwidth test only:
/*__global__ void jacobiKernelConstCoalescing(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
	//if (i >= n || j >= n)
			return;
	//dest[idx] = w * rhs[idx] - c_jacobi_stencil[4] * x[idx];
	dest[idx] = rhs[idx] - x[idx];
}

// for bandwidth test only:
__global__ void jacobiKernelConstCoalescing(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	int idx = i + j * stride;
	if (i == 0 || j == 0 || i >= n || j >= n)
		//if (i >= n || j >= n)
		return;
	//dest[idx] = w * rhs[idx] - c_jacobi_stencil[4] * x[idx];
	dest[idx] = rhs[idx] - x[idx];
}*/

// Call with pointers to the begginings of the arrays (including padding), n - column size-1
// Call with number of thread blocks = number of rows - 2
__global__ void jacobiKernelConstCoalescingLoop(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int rowIndex = (blockIdx.x + 1) * stride; // skip first row (zeros)
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		int idx = rowIndex + i;
		if (i > 0) // skip first column (zeros)
		{
			dest[idx] = w * rhs[idx] -
				(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
					+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
					+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);
		}
	}
}

// Call with pointers to the begginings of the arrays (including padding), n = column size-1
//Used 32 registers, 360 bytes cmem[0]
// This function must be called with a 1-dimension thread-block and a 1-dimension grid. 
// Each thread block is responsible for an entire row, or more than one row. 
// The inner loop is responsible to handle the rest of the row, in case the number of threads per block is smaller than the row size.
//  The outer loop is responsible to handle more rows, in case the number of thread blocks is smaller
// than the number of rows.
// !!!
__global__ void jacobiKernelConstCoalescingLoop2(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
//__global__ void jacobiKernelConstCoalescingLoop2(tfloat* dest, const tfloat* rhs, const tfloat* x, int stride, int n, tfloat w) // this version wouldn't use the L1 cache
{
	for (int j = blockIdx.x + 1; j < n;  j += gridDim.x) // skip first row (zeros)
	{
		int rowIndex = j * stride;
		for (int i = threadIdx.x; i < n; i += blockDim.x)
		{
			int idx = rowIndex + i;
			if (i > 0) // skip first column (zeros)
			{
				dest[idx] = w * rhs[idx] -
					(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
					+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
					+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);

				// for bandwidth tests only:
				//dest[idx] = rhs[idx] - x[idx];
			}
		}
	}
}

// Call with pointers to the begginings of the arrays (including padding), n = column size-1
// This function meant to be called with a 2-dimensional grid and a 2-dimensional thread-block.
// !!!
__global__ void
jacobiKernelConstCoalescingLoop3(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	for (; j < n; j += gridDim.y * blockDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
		{
			int idx = i + j * stride;
			if (i > 0) // skip first column (zeros)
			{
				dest[idx] = w * rhs[idx] -
					(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
						+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
						+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);

				// for bandwidth tests only:
				//dest[idx] = rhs[idx] - x[idx];
			}
		}
	}
}

// Call with pointers to the begginings of the arrays (including padding), n = column size-1
// This function meant to be called with a 2-dimensional grid and a 2-dimensional thread-block.
// !!!
__global__ void
jacobiKernelLoop4(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w, int rowMultiplyer, int rowJump)
{
	int j = rowMultiplyer * blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	int nRows = min(n, j + rowMultiplyer);
	for (; j < nRows; j += rowJump)
	{
		int i = threadIdx.x;
		//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
		{
			int idx = i + j * stride;
			if (i > 0) // skip first column (zeros)
			{
				dest[idx] = w * rhs[idx] -
					(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
						+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
						+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);

				// for bandwidth tests only:
				//dest[idx] = rhs[idx] - x[idx];
			}
		}
	}
}

const int THREADS_PER_BLOCK = 256;

__global__ void
//__launch_bounds__(64, 32)
//jacobiKernelLoopShared(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w, int rowMultiplyer, int rowJump)
jacobiKernelLoopShared(tfloat* dest, const tfloat* rhs, const tfloat* x, int stride, int n, tfloat w, int rowMultiplyer, int rowJump)
{
	__shared__ tfloat x_buf_up[THREADS_PER_BLOCK+2];
	__shared__ tfloat x_buf_middle[THREADS_PER_BLOCK+2];
	__shared__ tfloat x_buf_down[THREADS_PER_BLOCK+2];

	tfloat* __restrict__ p_up = x_buf_up;
	tfloat* __restrict__ p_middle = x_buf_middle;
	tfloat* __restrict__ p_down = x_buf_down;

	// threadIdx.y is 0 for 1 dim blocks
	int j = rowMultiplyer * blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = i + j * stride;

	if (i < n)
	{

		// Read first 2 rows:
		x_buf_up[1 + threadIdx.x] = x[idx - stride];
		x_buf_middle[1 + threadIdx.x] = x[idx];

		if (threadIdx.x == 0)
		{
			x_buf_up[0] = 0;
			x_buf_middle[0] = 0;
			x_buf_up[THREADS_PER_BLOCK+1] = 0;
			x_buf_middle[THREADS_PER_BLOCK+1] = 0;
		}

		if (blockIdx.x > 0)
		{
			if (threadIdx.x == 0)
			{
				x_buf_up[0] = x[idx - stride - 1];
				x_buf_middle[0] = x[idx - 1];
			}
		}

		if (threadIdx.x == THREADS_PER_BLOCK-1 && i < n)
		{
			x_buf_up[THREADS_PER_BLOCK+1] = x[idx - stride + 1];
			x_buf_middle[THREADS_PER_BLOCK+1] = x[idx + 1];
		}
	}

	int nRows = min(n, j + rowMultiplyer);
	for (; j < nRows; j += rowJump)
	{
		//int i = threadIdx.x;
		//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
		{
			tfloat right;
			if (i < n)
			{
				idx = i + j * stride;
				// Read next row:
				//p_down[1 + threadIdx.x] = x[idx + stride];
				p_down[1 + threadIdx.x] = x[idx + stride];
				right = rhs[idx]; // The performance is improved if this line is above the __syncthreads()
				if (blockIdx.x > 0)
				{
					if (threadIdx.x == 0)
					{
						p_down[0] = x[idx + stride - 1];
					}
				}
				if (threadIdx.x == THREADS_PER_BLOCK-1)
				{
					p_down[THREADS_PER_BLOCK+1] = x[idx + stride + 1];
				}
			}

			__syncthreads();
			if (i > 0 && i < n) // skip first column (zeros)
			{

				/* * /
				tfloat Ax =
					(c_jacobi_stencil[0] * x[idx - stride - 1] + c_jacobi_stencil[1] * x[idx - stride] + c_jacobi_stencil[2] * x[idx - stride + 1]
						+ c_jacobi_stencil[3] * x[idx - 1] + c_jacobi_stencil[4] * x[idx] + c_jacobi_stencil[5] * x[idx + 1]
						+ c_jacobi_stencil[6] * x[idx + stride - 1] + c_jacobi_stencil[7] * x[idx + stride] + c_jacobi_stencil[8] * x[idx + stride + 1]);
				/* * /
				// huge slowdown if tfloat is single float and the constants are doubles:
				tfloat Ax =
					(2.0f * x[idx - stride - 1] + 3.0f * x[idx - stride] + 4.0f * x[idx - stride + 1]
						+ 5.0f * x[idx - 1] + 6.0f * x[idx] + 7.0f * x[idx + 1]
						+ 8.0f * x[idx + stride - 1] + 9.0f * x[idx + stride] + 10.0f * x[idx + stride + 1]);
				/* * /
				tfloat Ax =
					(x[idx - stride - 1] + x[idx - stride] + x[idx - stride + 1]
						+ x[idx - 1] + x[idx] + x[idx + 1]
						+ x[idx + stride - 1] + x[idx + stride] + x[idx + stride + 1]);
				/*
				tfloat Ax =
					(c_jacobi_stencil[0] * p_up[threadIdx.x] + c_jacobi_stencil[1] * p_up[threadIdx.x+1] + c_jacobi_stencil[2] * p_up[threadIdx.x+2]
						+ c_jacobi_stencil[3] * p_middle[threadIdx.x] + c_jacobi_stencil[4] * p_middle[threadIdx.x+1] + c_jacobi_stencil[5] * p_middle[threadIdx.x+2]
						+ c_jacobi_stencil[6] * p_down[threadIdx.x] + c_jacobi_stencil[7] * p_down[threadIdx.x+1] + c_jacobi_stencil[8] * p_down[threadIdx.x+2]);
				/* */

				tfloat Ax =
					(c_jacobi_stencil[0] * p_up[threadIdx.x] + c_jacobi_stencil[1] * p_up[threadIdx.x + 1] + c_jacobi_stencil[2] * p_up[threadIdx.x + 2]
						+ c_jacobi_stencil[3] * p_middle[threadIdx.x] + c_jacobi_stencil[4] * p_middle[threadIdx.x + 1] + c_jacobi_stencil[5] * p_middle[threadIdx.x + 2]
					+ c_jacobi_stencil[6] * p_down[threadIdx.x] + c_jacobi_stencil[7] * p_down[threadIdx.x + 1] + c_jacobi_stencil[8] * p_down[threadIdx.x + 2]);
				/* */

				// for bandwidth tests only:
				//tfloat Ax = x[idx];
				//tfloat Ax = p_middle[threadIdx.x + 1];

				// Write the result to dest in global memory:
				dest[idx] = w * right - Ax;
			}

			tfloat* __restrict__ p_tmp = p_up;
			p_up = p_middle;
			p_middle = p_down;
			p_down = p_tmp; // p_down contents will be overwritten in next iteration
		}
	}
}
//bandwidth tests, 10 levels: 157.462 ms runtime
//bandwidth tests, 10 levels: 170.885  ms runtime for single precision floats, THREADS_PER_BLOCK=512
// probably the function version with best performance, 10 levels, single: 165.315 ms runtime

// for now, assuming blocks of size (1024, 1), grid size (1, yy)
// using shared memory
__global__ void
__launch_bounds__(1024, 2)
jacobiKernelLoop5(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
//jacobiKernelLoop5(tfloat*  dest, const tfloat*  rhs, const tfloat*  x, int stride, int n, tfloat w)
{
	__shared__ double x_buf_up[1026];
	__shared__ double x_buf_middle[1026];
	__shared__ double x_buf_down[1026];

	int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip first row (zeros)
	for (; j < n; j += gridDim.y * blockDim.y)
	{
		int i = threadIdx.x;
		//for (int i = threadIdx.x; i < n; i += 1024)
		{
			int idx = i + j * stride;

			x_buf_up[1 + threadIdx.x] = x[idx - stride];
			x_buf_middle[1 + threadIdx.x] = x[idx];
			x_buf_down[1 + threadIdx.x] = x[idx + stride];

			__syncthreads();

			if (i > 0) // skip first column (zeros)
			{
				/*tfloat Ax = c_jacobi_stencil[0] * x_buf_up[threadIdx.x] + c_jacobi_stencil[1] * x_buf_up[1 + threadIdx.x] + c_jacobi_stencil[2] * x_buf_up[2 + threadIdx.x];
				Ax += (c_jacobi_stencil[3] * x_buf_middle[threadIdx.x] + c_jacobi_stencil[4] * x_buf_middle[1 + threadIdx.x] + c_jacobi_stencil[5] * x_buf_middle[2 + threadIdx.x]);
				Ax += (c_jacobi_stencil[6] * x_buf_down[threadIdx.x] + c_jacobi_stencil[7] * x_buf_down[1 + threadIdx.x] + c_jacobi_stencil[8] * x_buf_down[2 + threadIdx.x]);*/
				/*tfloat Ax = c_jacobi_stencil[0] * x_buf_up[threadIdx.x];
				Ax += c_jacobi_stencil[1] * x_buf_up[1 + threadIdx.x];
				Ax += c_jacobi_stencil[2] * x_buf_up[2 + threadIdx.x];
				Ax += c_jacobi_stencil[3] * x_buf_middle[threadIdx.x];
				Ax += c_jacobi_stencil[4] * x_buf_middle[1 + threadIdx.x];
				Ax += c_jacobi_stencil[5] * x_buf_middle[2 + threadIdx.x];
				Ax += c_jacobi_stencil[6] * x_buf_down[threadIdx.x];
				Ax += c_jacobi_stencil[7] * x_buf_down[1 + threadIdx.x];
				Ax += c_jacobi_stencil[8] * x_buf_down[2 + threadIdx.x];*/
				tfloat Ax = 2 * x_buf_up[threadIdx.x];
				Ax += 3 * x_buf_up[1 + threadIdx.x];
				Ax += 4 * x_buf_up[2 + threadIdx.x];
				//Ax += 5 * x_buf_middle[threadIdx.x];
				//Ax += 6 * x_buf_middle[1 + threadIdx.x];
				//Ax += 7 * x_buf_middle[2 + threadIdx.x];
				//Ax += 8 * x_buf_down[threadIdx.x];
				//Ax += 9 * x_buf_down[1 + threadIdx.x];
				//Ax += 10 * x_buf_down[2 + threadIdx.x];

				// for bandwidth tests only:
				//dest[idx] = rhs[idx] - x[idx];
				//result -= x[idx] + x_buf_up[threadIdx.x] + x_buf_middle[threadIdx.x] + x_buf_down[threadIdx.x];
				//result += (c_jacobi_stencil[0] + c_jacobi_stencil[1] + c_jacobi_stencil[2] + c_jacobi_stencil[3] + c_jacobi_stencil[4] + c_jacobi_stencil[5] + c_jacobi_stencil[6] + c_jacobi_stencil[7] + c_jacobi_stencil[8] + c_jacobi_stencil[9]);

				dest[idx] = w * rhs[idx] - Ax;
			}
			__syncthreads();
		}
	}
}


// for bandwidth test only:
/*__global__ void jacobiKernelConstCoalescingLoop(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int rowIndex = (blockIdx.x+1) * stride;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		int idx = rowIndex + i;
		if (i > 0)
			dest[idx] = rhs[idx] - x[idx];
	}
}*/

// Only for 1 var
__global__ void solveKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, tfloat w)
{
	dest[0] = w * rhs[0];
}


#ifdef EMPTY_KERNELS

__global__ void emptyKernel(tfloat* __restrict__ dest, int count)
{

}

#define zeroKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define applyStencillKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, tfloat scale, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define addKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define residualKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define restrictKernel emptyKernel
#define prolongKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, const tfloat* __restrict__ stencil, int stride, int n, tfloat w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define jacobiKernel emptyKernel

__global__ void emptyKernel(tfloat* __restrict__ dest, const tfloat* __restrict__ rhs, const tfloat* __restrict__ x, int stride, int n, tfloat w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
#define jacobiKernelConstCoalescing emptyKernel

#endif

bool CudaZero(tfloat* dest, int count)
{
	int threadsPerBlock = TPB * TPB;
	int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
	//zeroKernel << <numBlocks, threadsPerBlock >> >(dest, count);
	zeroKernelLoop << <min(64, (count - 1) / 1024 + 1), 1024 >> >(dest, count);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaAdd(tfloat* dest, const tfloat* src, int count)
{
	int threadsPerBlock = TPB * TPB;
	threadsPerBlock = 1024;
	int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
	addKernel << <numBlocks, threadsPerBlock >> >(dest, src, count);
	//addKernel2 <<<min(2*n_sm, numBlocks), threadsPerBlock >>>(dest, src, count);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaAdd(tfloat* dest, const tfloat* src, int count, tfloat scale)
{
	int threadsPerBlock = TPB * TPB;
	int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
	addKernel << <numBlocks, threadsPerBlock >> >(dest, src, scale, count);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaMultiply(tfloat* p, int count, tfloat scale)
{
	int threadsPerBlock = TPB * TPB;
	int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
	multiplyKernel << <numBlocks, threadsPerBlock >> >(p, scale, count);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, int level)
{
	int max_index = n - 1; // col and row at index n are all zero
	static bool copied = false;
	//n -= 2;
	n--;
	int numBlocksDim = (max_index + TPB - 1) / TPB;

	dim3 threadsPerBlock(TPB, TPB);

	//numBlocksDim = min(numBlocksDim, 5); // only when using residualKernelLoopRegsLimit!

	dim3 numBlocks(numBlocksDim, numBlocksDim);

	if (!copied)
	{
		//cudaMemcpyToSymbolAsync(c_stencil, stencil, 9 * sizeof(tfloat), 0, cudaMemcpyDeviceToDevice);
		//copied = true;
	}

	//residualKernel << <numBlocks, threadsPerBlock >> >(dest + stride + 1, rhs + stride + 1, x + stride + 1, stencil, stride, n);
	//residualKernelConst << <numBlocks, threadsPerBlock >> >(dest + stride + 1, rhs + stride + 1, x + stride + 1, stride, n);
	//residualKernelConstCoalescing << <numBlocks, threadsPerBlock >> >(dest, rhs, x, stride, max_index);
	//residualKernelConstCoalescingArray << <numBlocks, threadsPerBlock >> >(dest, rhs, x, stride, max_index, level);

	//residualKernelLoop <<<numBlocks, threadsPerBlock >>>(dest, rhs, x, stride, max_index, level);
	residualKernelLoopRegsLimit << <numBlocks, threadsPerBlock >> >(dest, rhs, x, stride, max_index, level);

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, int level)
{
	dim3 threadsPerBlock(TPB, TPB);

	dim3 numBlocks((nx + TPB - 1 - 1) / TPB, (ny + TPB - 1 - 1) / TPB); // last col and last row are zero

	residualKernelLoopRegsLimit << <numBlocks, threadsPerBlock >> > (dest, rhs, x, stride, nx-1, ny-1, level);

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool MemCopyJacobiStencil(const tfloat* stencil)
{
	cudaMemcpyToSymbolAsync(c_jacobi_stencil, stencil, 9 * sizeof(tfloat), 0, cudaMemcpyDeviceToDevice);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool MemCopyJacobiStencil(const tfloat* stencil, int iLevel)
{
	cudaMemcpyToSymbolAsync(c_jacobi_stencils, stencil, 9 * sizeof(tfloat), iLevel * 9 * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool MemCopyResidualStencil(const tfloat* stencil, int iLevel)
{
	cudaMemcpyToSymbolAsync(c_residual_stencils, stencil, 9 * sizeof(tfloat), iLevel * 9 * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	//cudaError_t cudaStatus = cudaGetLastError();
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);

	tfloat stencil_t[9];
	stencil_t[0] = stencil[0];
	stencil_t[1] = stencil[3];
	stencil_t[2] = stencil[6];
	stencil_t[3] = stencil[1];
	stencil_t[4] = stencil[4];
	stencil_t[5] = stencil[7];
	stencil_t[6] = stencil[2];
	stencil_t[7] = stencil[5];
	stencil_t[8] = stencil[8];
	iLevel += MAX_LEVELS;
	cudaMemcpyToSymbolAsync(c_residual_stencils, stencil_t, 9 * sizeof(tfloat), iLevel * 9 * sizeof(tfloat));
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);

	return cudaStatus == cudaSuccess;
}

bool CreateCyclicReductionValues(tfloat a, tfloat b, tfloat c, int iLevel)
{
	struct CyclicReductionValues crv[MAX_LEVELS];

	for (int i = 0; i < MAX_LEVELS; i++)
	{
		tfloat fl = b / a;
		tfloat fu = c / a;

		crv[i].a = a;
		crv[i].b = b;
		crv[i].c = c;
		crv[i].ra = 1.0 / a;
		crv[i].fl = fl;
		crv[i].fu = fu;

		// prepare next level:
		a = 2 * b * fu - a; // OK only for constant values of a, b, c
		b = fl * b;
		c = fu * c;
	}

	cudaMemcpyToSymbolAsync(c_crv, crv, sizeof(crv), iLevel * sizeof(crv));
	//cudaError_t cudaStatus = cudaGetLastError();
	cudaError_t cudaStatus = cudaDeviceSynchronize(); // local memory, need a synch
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CreateCyclicReductionValues(tfloat* stencil, int iLevel)
{
	cudaError_t cudaStatus = cudaDeviceSynchronize(); // synch before reading stencil
	CUDA_ERROR(cudaStatus);

	tfloat a = stencil[4];
	tfloat b = stencil[3];
	tfloat c = stencil[5];

	if (!CreateCyclicReductionValues(a, b, c, iLevel)) // for x-zebra
		return false;
	a = stencil[4];
	b = stencil[1];
	c = stencil[7];
	if (!CreateCyclicReductionValues(a, b, c, MAX_LEVELS+iLevel)) // for y-zebra
		return false;
	return true;
}

bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, tfloat w, int level)
{
	int max_index = n - 1; // col and row at index n are all zero
	int numBlocksDim = (max_index + TPB - 1) / TPB;

	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(numBlocksDim, numBlocksDim);

	//dim3 threadsPerBlock(1024, 1);
	//dim3 numBlocks((max_index + 1024 - 1) / 1024, max_index);

	//cudaMemcpyToSymbolAsync(c_jacobi_stencil, stencil, 9 * sizeof(tfloat), 0, cudaMemcpyDeviceToDevice);
	//jacobiKernelConst << <numBlocks, threadsPerBlock >> >(dest + stride + 1, rhs + stride + 1, x + stride + 1, stride, max_index, w);
	//jacobiKernelConstCoalescing <<<numBlocks, threadsPerBlock >>>(dest, rhs, x, stride, max_index, w); // currently used for the results in the paper

	//jacobiKernelConstCoalescingLoop << <max_index-1, min(256, max_index) >> >(dest, rhs, x, stride, max_index, w);
	//jacobiKernelConstCoalescingLoop2 <<<max_index - 1, min(1024, max_index)>>>(dest, rhs, x, stride, max_index, w); //currently tested
	//jacobiKernelConstCoalescingLoop2 <<<min(max_index - 1, 80), min(512, max_index)>>>(dest, rhs, x, stride, max_index, w);

	/* * /
	// 10 levels tests:
	//jacobiKernelConstCoalescingLoop2 <<<10, 1024>>>(dest, rhs, x, stride, max_index, w);
	//dim3 ts(1024, 1);
	dim3 bs(max(max_index / THREADS_PER_BLOCK, 1), 20);
	//jacobiKernelConstCoalescingLoop3 <<<bs, ts>>>(dest, rhs, x, stride, max_index, w); //!!!
	//jacobiKernelLoop4 << <bs, ts >> >(dest, rhs, x, stride, max_index, w, 1024 / 20 + 1, 1);
	//jacobiKernelLoop4 << <bs, 960 >> >(dest, rhs, x, stride, max_index, w, 1024 / 20 + 1, 1);
	jacobiKernelLoopShared <<<bs, THREADS_PER_BLOCK >>>(dest, rhs, x, stride, max_index, w, max_index / 20 + 1, 1);
	//dim3 bs(1, 20);
	//jacobiKernelLoop5 << <bs, 1024 >> >(dest, rhs, x, stride, max_index, w);
	/* */

	//jacobiKernelConstCoalescingLoop3 <<<numBlocks, threadsPerBlock>>>(dest, rhs, x, stride, max_index, w); //!!!

	//jacobiKernel << <numBlocks, threadsPerBlock >> >(dest + stride + 1, rhs + stride + 1, x + stride + 1, stencil, stride, max_index, w);

	if (level == 0)
		jacobiKernelConstCoalescing <<<numBlocks, threadsPerBlock >>>(dest, rhs, x, stride, max_index, w); // currently used for the results in the paper
	else
	{
		jacobiKernelLevel <<<numBlocks, threadsPerBlock >>>(dest, rhs, x, stride, max_index, w, level);
	}

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// GPU Max Clock rate: 1709 MHz (1.71 GHz)
// Memory Clock rate : 4004 Mhz

//peak theoretical memory bandwidth = 4.004 * (192 / 8) * 2 = 192.192 GB / s (home computer)

// arithmetic operations:
// 128*10 = 1280 cores for single precisions
// 128*10 = 1280 single precision operations per cycle
// 4*10 = 40 double precision operations per cycle
// 1280*1.71 = 2188.8 Gflops/s - single precision (3855 GFLOPS in wikipedia)
// 40*1.71 = 68.4 Gflops/s - double precision (120 GFLOPS in wikipedia)
// https://en.wikipedia.org/wiki/GeForce_10_series
// The reason for the different values in wikipedia compared to my calculation are probably:
// 1) clock speed of 1506 (1708 is boosted clock rate).
// 2) counting each FMA instruction as 2 floating point operations (each core is counted as able to do 2 operations in a cycle)

// peak of 68.4 * 8 = 547.2 GB / s of double precision arithmetic operations
// => 547.2 / 192.192 = 2.8
// => 2.8 of arithmetic operations per var to reach a balance between the memory bandwidth and the double precision arithmetic unit speed

bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level)
{
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks((nx + TPB - 1 - 1) / TPB, (ny + TPB - 1 - 1) / TPB); // last col and last row are zero

	jacobiKernelLevel << <numBlocks, threadsPerBlock >> > (dest, rhs, x, stride, nx-1, ny-1, w, level); // 144 ms with 10 levels
	//addKernel << <stride * ny / 1024, 1024 >> > (dest, x, stride * ny); // for bandwidth tests - 154.125 ms with 10 levels
	//xzebraKernel <<< ny-2, 1024 >>> (dest, rhs, x, stride, nx - 1, ny - 1, 1.0, level);

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaJacobiSingleBlock(tfloat* x, const tfloat* rhs, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level, int n_times)
{
	//dim3 threadsPerBlock(TPB, min(TPB, ny-2));
	dim3 threadsPerBlock(min(TPB, nx-1), min(TPB, ny - 2));

	jacobiKernelLevelSingleBlock <<<1, threadsPerBlock >>> (x, rhs, stride, nx - 1, ny - 1, w, level, n_times);

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* tmp1, tfloat* tmp2, int tmp_stride)
{
	if (nx-1 > 1024)
	{
		xzebraKernelHuge<<<(ny-1)/2, 1024>>> (x, rhs, tmp1, tmp2, stride, nx - 2, ny - 1, level, 1, tmp_stride);
		xzebraKernelHuge<<<(ny-1)/2, 1024>>> (x, rhs, tmp1, tmp2, stride, nx - 2, ny - 1, level, 2, tmp_stride);
	}
	else
	{
		//xzebraKernel_1threadPerBlock << <ny - 1, 1 >> > (x, rhs, stride, nx - 2, ny - 1, level, 1);
		//xzebraKernel_1threadPerBlock << <ny - 1, 1 >> > (x, rhs, stride, nx - 2, ny - 1, level, 2);
		//xzebraKernel << <min(10, ny - 1), nx >> > (x, rhs, stride, nx - 2, ny - 1, level, 1);
		//xzebraKernel << <min(10, ny - 1), nx >> > (x, rhs, stride, nx - 2, ny - 1, level, 2);
		xzebraKernel <<< (ny-1)/2, nx-1 >> > (x, rhs, stride, nx - 2, ny - 1, level, 1);
		xzebraKernel <<< (ny-1)/2, nx-1 >> > (x, rhs, stride, nx - 2, ny - 1, level, 2);
	}

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaXZebraWithCusparse(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* buffer, int tmp_stride)
{
	cusparseStatus_t status;
	cudaError_t cudaStatus;

	prepareXzebraKernel <<<(ny - 1) / 2, 1024 >>> (x, rhs, stride, nx - 2, ny - 1, level, 1);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	status = cusparse_gtsv2_nopivot(cusparseHandle, nx - 2, (ny - 1) / 2, dl, d, du, x + 1 + stride, stride * 2, pBuffer);
	if (CUSPARSE_STATUS_SUCCESS != status)
	{
		cout << "Error in CudaXZebraWithCusparse: " << status << endl;
		throw "Error in CudaXZebraWithCusparse";
	}

	prepareXzebraKernel <<<(ny - 1) / 2, 1024 >>> (x, rhs, stride, nx - 2, ny - 1, level, 2);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	status = cusparse_gtsv2_nopivot(cusparseHandle, nx - 2, (ny - 2) / 2, dl, d, du, x + 1 + 2 * stride, stride * 2, pBuffer);
	if (CUSPARSE_STATUS_SUCCESS != status)
	{
		cout << "Error in CudaXZebraWithCusparse: " << status << endl;
		throw "Error in CudaXZebraWithCusparse";
	}

	//xzebraKernelHuge << <(ny - 1) / 2, 1024 >> > (x, rhs, tmp1, tmp2, stride, nx - 2, ny - 1, level, 1, tmp_stride);
	//xzebraKernelHuge << <(ny - 1) / 2, 1024 >> > (x, rhs, tmp1, tmp2, stride, nx - 2, ny - 1, level, 2, tmp_stride);

	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;


}

bool CudaInPlaceXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level)
{
	cudaError_t cudaStatus;

	inPlaceXzebraKernel <<<(ny - 1) / 2, min(1024, nx-2) >>> (x, rhs, stride, nx - 2, ny - 1, level, 1);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	if ((ny - 2) / 2 > 0)
	{
		inPlaceXzebraKernel <<<(ny - 2) / 2, min(1024, nx-2) >>> (x, rhs, stride, nx - 2, ny - 1, level, 2);
		cudaStatus = cudaGetLastError();
		CUDA_ERROR(cudaStatus);
	}
	return cudaStatus == cudaSuccess;
}

bool CudaYZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* tmp)
{
	cudaError_t cudaStatus;

	if (nx != ny)
		throw "Error in CudaYZebra - nx != ny";
	int n = nx;

	prepareYzebraKernel <<<(n - 1) / 2, min(1024, n-2) >>> (x, rhs, stride, n - 1, n - 2, level, 1, tmp);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	inPlaceTransposeYzebraKernel << <(n - 1) / 2, min(1024, n - 2) >> >(tmp, stride, n - 2, n - 1, level, 1);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	if ((n - 2) / 2 > 0)
	{
		inPlaceYzebraKernel <<<(n - 2) / 2, min(1024, n-2) >>> (tmp, rhs, stride, n - 2, n - 1, level, 2);
		cudaStatus = cudaGetLastError();
		CUDA_ERROR(cudaStatus);
	}
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(divUp(n - 2, TPB), divUp(n - 2, TPB));
	transposeKernel << <numBlocks, threadsPerBlock >> > (tmp, x, stride, n - 1);
	cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}


bool CudaSolve(tfloat* dest, const tfloat* rhs, int stride, tfloat w)
{
	solveKernel <<<1, 1 >>>(dest + stride + 1, rhs + stride + 1, w);

	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// currently only for finest level (ignores stencil argument)
bool CudaApplyStencill(tfloat* dest, const tfloat* x, const tfloat* stencil, int n, int stride)
{
	int max_index = n - 1; // col and row at index n are all zero
	int numBlocksDim = (max_index + TPB - 1) / TPB;

	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(numBlocksDim, numBlocksDim);
	applyStencillKernel << <numBlocks, threadsPerBlock >> >(dest, x, stride, max_index);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// dStride - dest stride
// sStride - src stride
// n - dest matrix size is n*n
bool CudaRestrict(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest)
{
	int numBlocksDim = ((nDest - 2) + TPB - 1) / TPB;
	dim3 threadsPerBlock(TPB, TPB);

	//numBlocksDim = min(numBlocksDim, 5); // only when using restrictKernelLoop!

	dim3 numBlocks(numBlocksDim, numBlocksDim);
	//dim3 numBlocks(min(numBlocksDim, 5), min(numBlocksDim, 4)); // only when using restrictKernelLoop!
	//dim3 numBlocks(min(numBlocksDim, 16), min(numBlocksDim, 16)); // only when using restrictKernelLoop!

	//restrictKernel << <numBlocks, threadsPerBlock >> >(dest, src, dStride, sStride, nDest);
	restrictKernelLoop <<<numBlocks, threadsPerBlock >>>(dest, src, dStride, sStride, nDest);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// dStride - dest stride
// sStride - src stride
// nSrc - src matrix size is n*n
// nDest - dest matrix size is n*n
bool CudaProlong(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest, int nSrc)
{
	int numBlocksDim = ((nSrc - 0) + TPB - 1) / TPB;
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(numBlocksDim, numBlocksDim);

	//prolongKernel << <numBlocks, threadsPerBlock >> >(dest, src, dStride, sStride, nSrc);
	//prolongKernel << <numBlocks, threadsPerBlock >> >(dest, src, dStride, sStride, nDest);
	prolongKernelLoop1 <<<numBlocks, threadsPerBlock >>>(dest, src, dStride, sStride, nDest);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// dStride - dest stride
// sStride - src stride
// n - dest matrix size is n*n
bool CudaRestrictY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx)
{
	int numyBlocks = ((nyDest - 2) + TPB - 1) / TPB;
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(((nx-2) + TPB - 1) / TPB, numyBlocks);

	restrictyKernel << <numBlocks, threadsPerBlock >> > (dest, src, dStride, sStride, nyDest, nx);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

// dStride - dest stride
// sStride - src stride
// nSrc - src matrix size is n*n
// nDest - dest matrix size is n*n
bool CudaProlongY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx)
{
	int numBlocksDim = ((nx - 0) + TPB - 1) / TPB;
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks(numBlocksDim, numBlocksDim);

	prolongyKernel <<<numBlocks, threadsPerBlock >>> (dest, src, dStride, sStride, nyDest, nx);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}

bool CudaInc(tfloat* __restrict__ p, int nBlocks, int nThreads, cudaStream_t stream)
{
	if (stream == nullptr)
		incKernel << <nBlocks, nThreads >> >(p);
	else
		incKernel << <nBlocks, nThreads, 0, stream >> >(p);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}


bool CudaNone(tfloat* __restrict__ p, int nBlocks, int nThreads, cudaStream_t stream)
{
	if (stream == nullptr)
		noneKernel << <nBlocks, nThreads >> >(p);
	else
		noneKernel << <nBlocks, nThreads, 0, stream >> >(p);
	cudaError_t cudaStatus = cudaGetLastError();
	CUDA_ERROR(cudaStatus);
	return cudaStatus == cudaSuccess;
}


/*
from helper_cuda.h

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

*/


/*
 OpenCV files:
 opencv\modules\cudafilters\src\cuda\filter2d.cu
 https://github.com/opencv/opencv_contrib/blob/master/modules/cudafilters/src/cuda/filter2d.cu
*/
