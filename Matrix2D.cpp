#ifdef HAVE_CUDA

#include "cuda_runtime.h"
#include <cublas_v2.h>

#else

#include "cuda_runtime_dummy.h"

#endif

#include "Classes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <memory.h>
#include <iostream>

#ifdef HAVE_CUDA
bool CudaZero(tfloat* dest, int count);
bool CudaAdd(tfloat* dest, const tfloat* src, int count);
bool CudaAdd(tfloat* dest, const tfloat* src, int count, tfloat scale);
//void CudaZeroBorder(tfloat* p, int n, int stride);
bool CudaMultiply(tfloat* p, int count, tfloat scale);
#else
bool CudaZero(tfloat* dest, int count) {return false;}
bool CudaAdd(tfloat* dest, const tfloat* src, int count) {return false;}
bool CudaAdd(tfloat* dest, const tfloat* src, int count, tfloat scale) {return false;}
bool CudaMultiply(tfloat* p, int count, tfloat scale) {return false;}
#endif

extern cublasHandle_t cublasHandle;


bool Matrix2D::AllocateMemory(int nx, int ny, int stride, bool bGpu, int access_policy)
{
	if (p != nullptr)
	{
		cout << "Error in Matrix2D::AllocateMemory(): p != nullptr" << endl;
		throw "Error in Matrix2D::AllocateMemory(): p != nullptr";
	}
	if (access_policy == -1)
	{
		access_policy = bGpu ? UM_ACCESS_GPU : UM_ACCESS_CPU;
	}
	size_t size = stride*ny;
	tfloat* p = (tfloat*)um_malloc(sizeof(tfloat)*size, access_policy);
	if (p == nullptr)
		return false;

	//if (bGpu)
	if (access_policy == UM_ACCESS_GPU)
		cudaMemset(p, 0, size * sizeof(tfloat));
	else
		memset(p, 0, size * sizeof(tfloat));

	this->bGpu = bGpu;
	this->nx = nx;
	this->ny = ny;
	this->stride = stride;
	this->count = size;
	this->p = p;
	this->access_policy = access_policy;

	return true;
}

Matrix2D::~Matrix2D()
{
	if (p != nullptr)
		um_free(p, access_policy);
}


void Matrix2D::MakeZero()
{
	if (bGpu)
	{
		CudaZero(p, count); // seems to be much faster than cudaMemset
		//CudaZero(p, 1); // for speed tests
		//cudaMemset(p, 0, count * sizeof(tfloat));
	}
	else
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaDeviceSynchronize();

		memset(p, 0, count * sizeof(tfloat));

		cudaStatus = cudaDeviceSynchronize();
	}
}

// may need to call ZeroBorder() after MakeRandom()
void Matrix2D::MakeRandom()
{
	tfloat* pp = p;
	for (int i = 0; i < count; i++)
	{
		pp[i] = ((tfloat)rand()) / RAND_MAX;
	}
}


void Matrix2D::Add(const Matrix2D& R)
{
	if (bGpu)
		CudaAdd(p, R.p, count);
	else
	{
		tfloat* dest = p;
		const tfloat* src = R.p;
		for (int i = 0; i < count; i++)
			dest[i] += src[i];
	}
}

void Matrix2D::Add(const Matrix2D& R, tfloat scale)
{
	if (bGpu)
		CudaAdd(p, R.p, count, scale);
	else
	{
		tfloat* __restrict__ dest = p;
		const tfloat* __restrict__ src = R.p;
		for (int i = 0; i < count; i++)
			dest[i] += scale * src[i];
	}
}

void Matrix2D::Multiply(tfloat scale)
{
	if (bGpu)
		CudaMultiply(p, count, scale);
	else
	{
		tfloat* pp = p;
		for (int i = 0; i < count; i++)
			pp[i] *= scale;
	}
}


void Matrix2D::ZeroBorder()
{
	//if (bGpu)
		//CudaZeroBorder(p, n, n);
	tfloat* pp = p;
	for (int i = 0; i < ny; i++)
	{
		pp[i*stride] = 0.0;
		pp[nx - 1 + i*stride] = 0.0;

		// in case stride > nx:
		for (int j = nx; j < stride; j++)
			pp[j + i*stride] = 0.0;
	}

	for (int i = 0; i < nx; i++)
	{
		pp[i] = 0.0;
		pp[(ny - 1) * stride + i] = 0.0;
	}
}

// "to" must be a matrix of the same size as "this"
void Matrix2D::CoptyTo(Matrix2D& to) const
{
	if (bGpu)
	{
		// should handle all kinds of copy (GPU to GPU, CPU to CPU and combinations):
		cudaError_t cudaStatus = cudaMemcpy(to.p, p, count * sizeof(tfloat), cudaMemcpyDefault);
	}
	else
		memcpy(to.p, p, count * sizeof(tfloat));
}

// "to" must be a matrix of the same size as "this"
void Matrix2D::CoptyBorderTo(Matrix2D& to) const
{
	const tfloat* pSrc = p;
	tfloat* pDest = to.p;
	for (int i = 0; i < ny; i++)
	{
		pDest[i] = pSrc[i];
		pDest[(nx - 1)*stride + i] = pSrc[(nx - 1)*stride + i];

		// in case stride > nx:
		for (int j = nx; j < stride; j++)
			pDest[j + i*stride] = pSrc[j + i*stride];
	}

	for (int i = 0; i < nx; i++)
	{
		pDest[i * stride] = pSrc[i * stride];
		pDest[ny - 1 + i * stride] = pSrc[ny - 1 + i * stride];
	}
}

// This norm is for the whole matrix, includind the boundaries
// The norm is a vector norm, and is not normalized according to the number of values
tfloat Matrix2D::Norm() const
{
	tfloat result = 0.0;
	tfloat* pp = p;

	if (bGpu)
	{
#ifdef USE_DOUBLE
		cublasDnrm2(cublasHandle, count, pp, 1, &result); // for double
#else
		cublasSnrm2(cublasHandle, count, pp, 1, &result); // for float
#endif
	}
	else
	{
		for (int i = 0; i < count; i++)
			result += pp[i] * pp[i];
		result = sqrt(result);
	}

	return result;
}

tfloat Matrix2D::DotProduct(const Matrix2D& R) const
{
	tfloat result = 0.0;
	tfloat* p1 = p;
	tfloat* p2 = R.p;

	if (bGpu)
	{
#ifdef USE_DOUBLE
		cublasDdot(cublasHandle, count, p1, 1, p2, 1, &result); // for double
#else
		cublasSdot(cublasHandle, count, p1, 1, p2, 1, &result); // for float
#endif
	}
	else
	{
		for (int i = 0; i < count; i++)
			result += p1[i] * p2[i];
	}

	return result;

}

void Matrix2D::Swap(Matrix2D& other)
{
	if (other.nx != nx || other.ny != ny || other.count != count)
		throw "Error in Matrix2D::Swap() - the matrices are of different sizes";
	tfloat* pTemp = other.p;
	other.p = this->p;
	this->p = pTemp;
}

void Matrix2D::Transpose(Matrix2D& dest) const
{
	if (dest.nx != nx || dest.ny != ny || dest.count != count)
		throw "Error in Matrix2D::Transpose() - the matrices are not in compatible sizes";
#ifdef HAVE_CUDA
	tfloat alpha = 1.0;
	tfloat beta = 0.0;
#ifdef USE_DOUBLE
	cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nx, ny, &alpha, p, stride, &beta, nullptr, stride, dest.p, stride);
#else
	cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nx, ny, &alpha, p, stride, &beta, nullptr, stride, dest.p, stride);
#endif
#endif
	// TODO: add cpu implementation
}


#if 0

#define MAX_BLOCK_SZ 512

/*-------------------------------------------------------------------
* Function:    Dev_dot  (kernel)
* Purpose:     Implement a dot product of floating point vectors
*              using atomic operations for the global sum
* In args:     x, y, n
* Out arg:     z
*
*/
__global__ void Dev_dot(float x[], float y[], float z[], int n) {
	/* Use tmp to store products of vector components in each block */
	/* Can't use variable dimension here                            */
	__shared__ float tmp[MAX_BLOCK_SZ];
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	int loc_t = threadIdx.x;

	if (t < n) tmp[loc_t] = x[t] * y[t];
	__syncthreads();

	/* This uses a tree structure to do the addtions */
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if (loc_t < stride)
			tmp[loc_t] += tmp[loc_t + stride];
		__syncthreads();
	}

	/* Store the result from this cache block in z[blockIdx.x] */
	if (threadIdx.x == 0) {
		z[blockIdx.x] = tmp[0];
	}
}  /* Dev_dot */

   /*-------------------------------------------------------------------
   * Function:  Dot_wrapper
   * Purpose:   CPU wrapper function for GPU dot product
   * Note:      Assumes x_d, y_d have already been
   *            allocated and initialized on device.  Also
   *            assumes z_d has been allocated.
   */
float Dot_wrapper(float x_d[], float y_d[], float z_d[],
	int n, int blocks, int threads) {
	int i;
	float dot = 0.0;
	float z_h[blocks];

	/* Invoke kernel */
	Dev_dot << <blocks, threads >> >(x_d, y_d, z_d, n);
	cudaThreadSynchronize();

	cudaMemcpy(&z_h, z_d, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < blocks; i++)
		dot += z_h[i];
	return dot;
}  /* Dot_wrapper */

#endif
